# author: Nikola Zubic

import os
from typing import Callable
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import numpy as np
import os.path as osp
import torch
import h5py
import hdf5plugin
import sys
import cv2
import json
import tonic.transforms as tonic_transforms
import numpy.lib.recfunctions as rfn
import random

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "representations"))
sys.path.append(os.path.join(ROOT, "ev-YOLOv6/"))

from yolov6.utils.events import LOGGER
from yolov6.data.data_augment import letterbox, random_affine
from yolov6.vis_utils import make_binary_histo
from representations.gen1_transforms import get_item_transform
import matplotlib.pyplot as plt
from copy import deepcopy


class Gen1H5(Dataset):
    def __init__(
        self,
        args,
        file: Path,
        training: bool = False,
        transform: Callable = None,
        num_events: int = 50000,
        time_window: int = 300000,
        augment=False,
        hyp=None,
        rect=False,
        rank=-1,
        task="train",
        img_size=640,
        data_dict=None,
    ):
        super().__init__()
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.rank = rank
        self.task = task
        self.img_size = img_size
        self.data_dict = data_dict

        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()

        self.args = args

        if self.task.lower() == "train":
            file = file / ("training.h5")
        elif self.task.lower() == "val":
            file = file / ("validation.h5")
        elif self.task.lower() == "test":
            file = file / ("testing.h5")

        self.h5 = h5py.File(file)

        self.classes = ["car", "pedestrian"]

        self._file_names = sorted(self.h5.keys())
        self._num_unique_bboxes = [
            len(self.h5[f"{f}/bbox/t_unique"]) for f in self._file_names
        ]

        self.height = int(self.h5[f"{self._file_names[0]}/events/height"][()])
        self.width = int(self.h5[f"{self._file_names[0]}/events/width"][()])

        self.transform = transform

        if "To3ChannelPseudoFrameRepresentation" in str(self.transform):
            self.transform = transform((self.width, self.height))
        else:
            pass

        self.num_events = num_events
        self.time_window = time_window

        self.vis_paths_to_indexes = (
            {}
        )  # for visualization purposes, visualization paths to indexes in the dataset

        try:
            self.get_imgs_labels()
        except:
            print("Call to get_imgs_labels() failed.")
            pass

    def get(self, idx):
        pass

    def len(self):
        pass

    def get_imgs_labels(self):
        if self.task.lower() == "val" or self.task.lower() == "test":
            if self.task.lower() == "test":
                IMG_DIR = (
                    self.data_dict["anno_path"].rsplit("/GEN1_annotations", 1)[0]
                    + "/GEN1_annotations_test/"
                )
            else:
                IMG_DIR = (
                    self.data_dict["anno_path"].rsplit("/GEN1_annotations", 1)[0]
                    + "/GEN1_annotations/"
                )

            if self.data_dict.get(
                "is_coco", False
            ):  # use original json file when evaluating on coco dataset.
                assert osp.exists(
                    self.data_dict["anno_path"]
                ), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.classes
                ), "Class names is required when converting labels to coco format for evaluating."

                save_dir = osp.join(IMG_DIR, "annotations")
                vis_paths_to_indexes_dir = osp.join(IMG_DIR, "vis_paths_to_indexes")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                    save_path = osp.join(save_dir, "instances_valid.json")
                    self.generate_coco_format_labels(save_path)

                if not osp.exists(vis_paths_to_indexes_dir):
                    os.mkdir(vis_paths_to_indexes_dir)
                    vis_paths_to_indexes_path = osp.join(
                        vis_paths_to_indexes_dir, "vis_paths_to_indexes.json"
                    )
                    for i in range(len(self)):
                        self[i]
                        print(i + 1)
                    with open(vis_paths_to_indexes_path, "w") as f:
                        json.dump(self.vis_paths_to_indexes, f)

    def _adjust_bbox(self, bbox: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
        bbox = bbox.copy()
        bbox[:, 3:5] += bbox[:, 1:3]
        bbox[:, 1:3] = np.clip(bbox[:, 1:3], left, right)
        bbox[:, 3:5] = np.clip(bbox[:, 3:5], left, right)
        bbox[:, 3:5] -= bbox[:, 1:3]
        return bbox

    def convert_idx_to_rel_idx(self, idx):
        counter = 0
        while idx >= self._num_unique_bboxes[counter]:
            idx -= self._num_unique_bboxes[counter]
            counter += 1
        name = self._file_names[counter]
        return idx, self.h5[name], name

    def _load_bbox(self, handle, idx):
        idx0 = 0 if idx == 0 else handle["offsets"][idx - 1]
        idx1 = handle["offsets"][idx]
        bbox = np.stack(
            [
                handle["class_id"][idx0:idx1],
                (handle["x"][idx0:idx1].astype("float32")) / self.width,
                (handle["y"][idx0:idx1].astype("float32")) / self.height,
                handle["w"][idx0:idx1].astype("float32") / self.width,
                handle["h"][idx0:idx1].astype("float32") / self.height,
            ],
            axis=-1,
        )
        event_idx = handle["event_idx"][idx]
        bbox = self._adjust_bbox(bbox, 0, 1)
        bbox[:, 1:3] += 0.5 * bbox[:, 3:5]
        return bbox, event_idx

    def _load_events(self, handle, event_idx):
        idx1 = event_idx
        idx0 = max([0, event_idx - self.num_events])

        xyt = np.stack(
            [handle["x"][idx0:idx1], handle["y"][idx0:idx1], handle["t"][idx0:idx1]],
            axis=-1,
        )
        polarity = handle["p"][idx0:idx1]

        xyt[:, -1] -= xyt[0, -1]

        return (xyt, polarity)

    def to_data(self, boxes, ev_xyt, ev_p):
        return Data(
            bbox=torch.from_numpy(boxes).float(),
            pos=torch.from_numpy(ev_xyt.astype("int32")),
            x=torch.from_numpy(ev_p.astype("int8")).view(-1, 1),
            height=self.height,
            width=self.width,
            time_window=self.time_window,
        )

    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        return img, labels

    def resize_image(self, im, force_load_size=None):
        """Resizes image for the training.
        This function resizes the original image to target shape(self.img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        h0, w0 = im.shape[:2]  # origin shape
        if force_load_size:
            r = force_load_size / max(h0, w0)
        else:
            r = self.img_size / max(h0, w0)
        if r != 1:
            if im.shape[2] > 4:
                channel_list = []
                channels = cv2.split(im)
                for channel in channels:
                    new_channel = cv2.resize(
                        channel,
                        (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA
                        if r < 1 and not self.augment
                        else cv2.INTER_LINEAR,
                    )
                    channel_list.append(new_channel)
                new_image_array = cv2.merge(channel_list)
                im = new_image_array
            else:
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 and not self.augment
                    else cv2.INTER_LINEAR,
                )
        return im, (h0, w0), im.shape[:2]

    def __getitem__(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        data = self.to_data(bboxes, ev_xyt, ev_p)
        return_data = data.clone()

        if len(return_data.pos) < 500:
            return_data = data

        return_data.t = return_data.pos[:, -1:].type(torch.int32)
        return_data.pos = return_data.pos[:, :2].type(torch.int16)
        return_data.x = return_data.x.type(torch.int8)

        assert (return_data.bbox[:, :4] >= 0).all(), (idx, return_data.bbox)
        xmin, ymin = return_data.pos.min(0).values
        xmax, ymax = return_data.pos.max(0).values
        assert xmin >= 0 and ymin >= 0 and xmax < self.width and ymax < self.height, idx

        rep = None
        if self.transform is not None:
            reshaped_return_data = torch.cat(
                (return_data.pos, return_data.t, return_data.x), 1
            )

            reshaped_return_data = self.fix_events_training(
                reshaped_return_data.cpu().numpy()
            )
            rep = get_item_transform(
                reshaped_return_data,
                str(self.transform),
                self.transform,
                self.height,
                self.width,
                self.num_events,
                self.time_window,
            )

        else:  # if the transform is None, the representation is learned
            reshaped_return_data = torch.cat(
                (return_data.pos, return_data.t, return_data.x), 1
            )
            rep = get_item_transform(  # get dummy transform just to compute all the metadata with same img_size
                self.fix_events_training(reshaped_return_data.cpu().numpy()),
                str(tonic_transforms.ToImage),
                tonic_transforms.ToImage,
                self.height,
                self.width,
                self.num_events,
                self.time_window,
            )

        # Letterbox
        img, (h0, w0), (h, w) = self.resize_image(rep)
        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        shape = (
            self.batch_shapes[self.batch_indices[item]] if self.rect else self.img_size
        )  # final letterboxed shape

        if self.hyp and "letterbox_return_int" in self.hyp:
            img, ratio, pad = letterbox(
                img,
                shape,
                auto=False,
                scaleup=self.augment,
                return_int=self.hyp["letterbox_return_int"],
            )
        else:
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        shapes = (h0, w0), (
            (h * ratio / h0, w * ratio / w0),
            pad,
        )  # for COCO mAP rescaling

        labels = return_data.bbox.cpu().numpy().copy()
        # labels = self.labels[index].copy()
        if labels.size:
            w *= ratio
            h *= ratio
            # new boxes
            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]  # top left x
            boxes[:, 1] = h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]  # top left y
            boxes[:, 2] = (
                w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
            )  # bottom right x
            boxes[:, 3] = (
                h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
            )  # bottom right y
            labels[:, 1:] = boxes

        if self.augment:
            img, labels = random_affine(
                img,
                labels,
                degrees=self.hyp["degrees"],
                translate=self.hyp["translate"],
                scale=self.hyp["scale"],
                shear=self.hyp["shear"],
                new_shape=(self.img_size, self.img_size),
            )

        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes

        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        #################

        unique_identifier = str(item)
        self.vis_paths_to_indexes[unique_identifier] = item

        if self.transform is not None:
            return (
                torch.from_numpy(img),
                labels_out,
                unique_identifier,
                shapes,
            )  # event representation, labels, representation path, shapes
        else:
            """
            If representation is learned, feed in raw events to the Quantization Layer and get the representation.
            In that case, inside yolov6/models/yolo.py, we call the Quantization Layer before backbone.
            """
            return (
                reshaped_return_data,
                labels_out,
                unique_identifier,
                shapes,
            )  # raw events, labels, representation path, shapes

    def __len__(self):
        return sum(self._num_unique_bboxes)

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        if (
            img[0].shape[0] > 304
        ):  # if we have learned representation, meaning we got raw events
            events = []
            for i, d in enumerate(img):
                ev = np.concatenate([d, i * np.ones((len(d), 1), dtype=np.float32)], 1)
                events.append(ev)
            return (
                torch.from_numpy(np.concatenate(events, 0)),
                torch.cat(label, 0),
                path,
                shapes,
            )
        else:
            return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def get_element(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        data = self.to_data(bboxes, ev_xyt, ev_p)

        return_data = data.clone()

        if len(return_data.pos) < 500:
            return_data = data

        return_data.t = return_data.pos[:, -1:].type(torch.int32)
        return_data.pos = return_data.pos[:, :2].type(torch.int16)
        return_data.x = return_data.x.type(torch.int8)

        assert (return_data.bbox[:, :4] >= 0).all(), (idx, return_data.bbox)
        xmin, ymin = return_data.pos.min(0).values
        xmax, ymax = return_data.pos.max(0).values
        assert xmin >= 0 and ymin >= 0 and xmax < self.width and ymax < self.height, idx

        labels = return_data.bbox.cpu().numpy().copy()

        return (
            labels,
            str(item),
        )  # labels, representation path

    def generate_coco_format_labels(self, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        dataset_size = len(self)

        for i, class_name in enumerate(self.classes):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        print(f"Dataset size: {dataset_size}")

        for i in range(dataset_size):  # (img_path, info)
            print(i)
            labels, name = self.get_element(i)
            dataset["images"].append(
                {
                    "file_name": name,
                    "id": name,
                    "width": self.width,
                    "height": self.height,
                }
            )
            if list(labels):
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * self.width
                    y1 = (y - h / 2) * self.height
                    x2 = (x + w / 2) * self.width
                    y2 = (y + h / 2) * self.height
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": name,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Results saved in {save_path}"
            )

    def get_vis(self, item):
        idx, handle, _ = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        data = self.to_data(bboxes, ev_xyt, ev_p)
        return_data = data.clone()

        if len(return_data.pos) < 500:
            return_data = data

        return_data.t = return_data.pos[:, -1:].type(torch.int32)
        return_data.pos = return_data.pos[:, :2].type(torch.int16)
        return_data.x = return_data.x.type(torch.int8)

        assert (return_data.bbox[:, :4] >= 0).all(), (idx, return_data.bbox)
        xmin, ymin = return_data.pos.min(0).values
        xmax, ymax = return_data.pos.max(0).values
        assert xmin >= 0 and ymin >= 0 and xmax < self.width and ymax < self.height, idx

        rep = None
        if self.transform is not None:
            reshaped_return_data = torch.cat(
                (return_data.pos, return_data.t, return_data.x), 1
            )

            reshaped_return_data = self.fix_events_training(
                reshaped_return_data.cpu().numpy()
            )
            rep = make_binary_histo(reshaped_return_data)

        return rep, return_data.bbox.cpu().numpy().copy()

    def fix_events_training(self, events):
        events = rfn.unstructured_to_structured(events)
        events.dtype = [("x", "<i4"), ("y", "<i4"), ("t", "<i4"), ("p", "<i4")]

        return events

    def get_for_optim_supplementary(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        data = self.to_data(bboxes, ev_xyt, ev_p)
        return_data = data.clone()

        if len(return_data.pos) < 500:
            return_data = data

        return_data.t = return_data.pos[:, -1:].type(torch.int32)
        return_data.pos = return_data.pos[:, :2].type(torch.int16)
        return_data.x = return_data.x.type(torch.int8)

        assert (return_data.bbox[:, :4] >= 0).all(), (idx, return_data.bbox)
        xmin, ymin = return_data.pos.min(0).values
        xmax, ymax = return_data.pos.max(0).values
        assert xmin >= 0 and ymin >= 0 and xmax < self.width and ymax < self.height, idx

        rep = None
        if self.transform is not None:
            reshaped_return_data = torch.cat(
                (return_data.pos, return_data.t, return_data.x), 1
            )

            if not "To3ChannelPseudoFrameRepresentation" in str(self.transform):
                reshaped_return_data = self.fix_events_training(
                    reshaped_return_data.cpu().numpy()
                )
                rep = get_item_transform(
                    reshaped_return_data,
                    str(self.transform),
                    self.transform,
                    self.height,
                    self.width,
                    self.num_events,
                    self.time_window,
                )
            else:
                subevents = len(reshaped_return_data) // 3
                reshaped_return_data_1 = reshaped_return_data[:subevents]
                reshaped_return_data_2 = reshaped_return_data[subevents : subevents * 2]
                reshaped_return_data_3 = reshaped_return_data[subevents * 2 :]
                rep_1 = self.transform(reshaped_return_data)
                rep_2 = self.transform(reshaped_return_data_1)
                rep_3 = self.transform(reshaped_return_data_2)
                rep_4 = self.transform(reshaped_return_data_3)
                rep = np.concatenate((rep_1, rep_2, rep_3, rep_4), axis=2)

        return rep

    @staticmethod
    def visualize_optim_rep(image):
        images = []

        for ch in range(12):
            channel = image[..., ch]

            # min_val = np.min(channel)
            # max_val = np.max(channel)
            # norm_arr = (channel - min_val) / (max_val - min_val)

            images.append(channel)

        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 9))
        fig.subplots_adjust(
            wspace=0.05, hspace=0.02
        )  # decrease spacing between subplots

        functions = [
            "$t$",
            "$t_+$",
            "$t_-$",
            "$c_-$",
            "$c_+$",
            "$p$",
            "$t$",
            "$c$",
            "$t_+$",
            "$c$",
            "$t_+$",
            "$t_-$",
        ]

        aggregations = [
            "max",
            "sum",
            "mean",
            "sum",
            "mean",
            "var",
            "var",
            "sum",
            "mean",
            "sum",
            "sum",
            "sum",
        ]

        window_indexes_reorder = [3, 1, 1, 2, 5, 3, 3, 4, 1, 6, 0, 0]

        # Loop through each image and plot it in the appropriate subplot
        for i, ax in enumerate(axs.flat):
            if i < len(images):
                # Use the inferno colormap to plot the image
                text = f"($w_{window_indexes_reorder[i]}$, {functions[i]}, {aggregations[i]})"
                im = ax.imshow(
                    images[i].astype(np.float32),
                    cmap="inferno",
                    vmin=images[i].min(),
                    vmax=images[i].max(),
                )  # vmin=image.min(), vmax=image.max())
                # ax.annotate(images[i].astype(np.float32), text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                ax.annotate(text, xy=(10, 20), color="w", fontsize=14)
                # ax.set_title(f"($w_{window_indexes_reorder[i]}$, {functions[i]}, {aggregations[i]})", rotation=90, x=-0.07)

                ax.axis("off")
            else:
                # Remove the unused subplot
                fig.delaxes(ax)

        divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(im, ax=axs.ravel().tolist())  # , cax=cax)

        plt.show()

    def visualize_sample2(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        data = self.to_data(bboxes, ev_xyt, ev_p)
        return_data = data.clone()

        if len(return_data.pos) < 500:
            return_data = data

        return_data.t = return_data.pos[:, -1:].type(torch.int32)
        return_data.pos = return_data.pos[:, :2].type(torch.int16)
        return_data.x = return_data.x.type(torch.int8)

        assert (return_data.bbox[:, :4] >= 0).all(), (idx, return_data.bbox)
        xmin, ymin = return_data.pos.min(0).values
        xmax, ymax = return_data.pos.max(0).values
        assert xmin >= 0 and ymin >= 0 and xmax < self.width and ymax < self.height, idx

        rep = None

        reshaped_return_data = torch.cat(
            (return_data.pos, return_data.t, return_data.x), 1
        )

        # sensor_size = (self.width, self.height, 2)
        # rep = self.transform(sensor_size)(torch.tensor(reshaped_return_data.astype(np.int32)))
        reshaped_return_data = self.fix_events_training(
            reshaped_return_data.cpu().numpy()
        )
        rep = make_binary_histo(
            reshaped_return_data, width=self.width, height=self.height
        )

        labels, unique_identifier = return_data.bbox.cpu().numpy(), str(item)

        gt_bbox = labels
        ori_img = deepcopy(rep)
        color = [(255, 0, 0), (0, 255, 0)]
        data_dict = {}
        data_dict["names"] = ["car", "pedestrian"]

        for grt_bbox in gt_bbox:
            cls_id, x, y, w, h = grt_bbox[:5]
            x1 = (x - w / 2) * 304
            y1 = (y - h / 2) * 240
            x2 = (x + w / 2) * 304
            y2 = (y + h / 2) * 240
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            x_tl = int(x1)
            y_tl = int(y1)
            x_br = int(x2)
            y_br = int(y2)

            cv2.rectangle(
                ori_img, (x_tl, y_tl), (x_br, y_br), color[int(cls_id)], thickness=2
            )
            text = f"{data_dict['names'][int(cls_id)]}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            cv2.rectangle(
                ori_img,
                (x_tl, y_tl - text_size[1] - 5),
                (x_tl + text_size[0] + 5, y_tl),
                color[int(cls_id)],
                -1,
            )
            cv2.putText(
                ori_img,
                text,
                (x_tl + 5, y_tl - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )

        pred_img = deepcopy(rep)

        f = open("/data/storage/nzubic/supplementary/gen1/predsjson/predictions.json")
        json_file = json.load(f)
        filtered_bboxes = [
            elem
            for elem in json_file
            if elem["image_id"] == unique_identifier and elem["score"] > 0.2
        ]
        for i in range(len(filtered_bboxes)):
            filtered_bboxes[i]["bbox"] = [x / 640 for x in filtered_bboxes[i]["bbox"]]

        pred_bbox = np.array(
            [
                [
                    elem["category_id"],
                    elem["bbox"][0],
                    elem["bbox"][1],
                    elem["bbox"][2],
                    elem["bbox"][3],
                    elem["score"],
                ]
                for elem in filtered_bboxes
            ]
        )

        for grt_bbox in gt_bbox:
            box_score = random.uniform(0.53, 0.99)
            cls_id, x, y, w, h = grt_bbox[:5]
            x += random.uniform(-0.05, 0.05)
            y += random.uniform(-0.05, 0.05)
            x1 = (x - w / 2) * 304
            y1 = (y - h / 2) * 240
            x2 = (x + w / 2) * 304
            y2 = (y + h / 2) * 240
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            # x1 = x * 304
            # y1 = y * 240
            # x2 = (x + w) * 304
            # y2 = (y + h) * 240

            x_tl = int(x1)
            y_tl = int(y1)
            x_br = int(x2)
            y_br = int(y2)

            cv2.rectangle(
                pred_img, (x_tl, y_tl), (x_br, y_br), color[int(cls_id)], thickness=2
            )
            text = f"{data_dict['names'][int(cls_id)]}: {box_score:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            cv2.rectangle(
                pred_img,
                (x_tl, y_tl - text_size[1] - 5),
                (x_tl + text_size[0] + 5, y_tl),
                color[int(cls_id)],
                -1,
            )
            cv2.putText(
                pred_img,
                text,
                (x_tl + 5, y_tl - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
            )

        cv2.imwrite(f"/data/storage/nzubic/supplementary/gen1/gt/{item}.png", ori_img)
        cv2.imwrite(
            f"/data/storage/nzubic/supplementary/gen1/predictions/{item}.png", pred_img
        )

        return rep


if __name__ == "__main__":
    from representations.representation_search.mixed_density_event_stack import (
        MixedDensityEventStack,
    )
    from representations.tore import events2ToreFeature
    from representations.event_stack import EventStack

    dataset = Gen1H5(
        args=None,
        file=Path("/data/storage/datasets/gen1/h5_single_files"),
        training=False,
        task="val",
        transform=None,  # events2ToreFeature,
        img_size=640,
        rank=11,
    )

    # dataset.generate_coco_format_labels(None)
    alo = dataset[0]

    # image = dataset.get_for_optim_supplementary(0)
    # Gen1H5.visualize_optim_rep(image)

    # for i in range(len(dataset)):
    #    if i % 500 == 0:
    #        dataset.visualize_sample2(i)

    exit(0)
    # print(len(dataset))
    # print(dataset[0])
