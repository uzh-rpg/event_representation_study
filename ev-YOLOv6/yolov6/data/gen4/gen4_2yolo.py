import os
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
import cv2
from typing import Callable
import os.path as osp
import json
import sys
import h5py
import hdf5plugin
import random
import warnings

warnings.filterwarnings("ignore")

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "representations"))
sys.path.append(os.path.join(ROOT, "ev-YOLOv6/"))

from representations.gen4_transforms import get_item_transform
from yolov6.data.data_augment import letterbox, random_affine


def getDataloader(name):
    dataset_dict = {"Prophesee": Prophesee}
    return dataset_dict.get(name)


class Prophesee:
    def __init__(
        self,
        args,
        root,
        transform: Callable = None,
        object_classes=["pedestrian", "two wheeler", "car"],
        height=720,
        width=1280,
        mode="train",
        img_size=640,
        augment=False,
        hyp=None,
        rect=False,
        rank=-1,
        task="Train",
        data_dict=None,
    ):
        """
        Creates an iterator over the Prophesee object recognition dataset.
        :param root: path to dataset root
        :param object_classes: list of string containing objects or "all" for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param mode: "train", "val" or "test"
        """
        self.args = args
        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.transform = transform

        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.rank = rank
        self.task = task
        self.img_size = img_size
        self.data_dict = data_dict

        self.main_process = self.rank in (-1, 0)

        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.classes = object_classes

        self.max_nr_bbox = 60

        if not root.endswith("/"):
            root += "/"
        self.hf = f"{root}{self.mode}"
        self.reps = os.path.join(self.hf, "reps")
        self.labels = os.path.join(self.hf, "labels")

        self.files = self.load_data_files()

        self.object_classes = object_classes
        self.nr_classes = len(self.object_classes)  # 7 classes

        self.vis_paths_to_indexes = (
            {}
        )  # for visualization purposes, visualization paths to indexes in the dataset

        self.get_imgs_labels()

    def get_imgs_labels(self):
        if self.task.lower() == "val" or self.task.lower() == "test":
            if self.task.lower() == "test":
                IMG_DIR = (
                    self.data_dict["anno_path"].rsplit("/GEN4_annotations", 1)[0]
                    + "/GEN4_annotations_test/"
                )
            else:
                IMG_DIR = (
                    self.data_dict["anno_path"].rsplit("/GEN4_annotations", 1)[0]
                    + "/GEN4_annotations/"
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
                        self.fill_vis(i)
                        print(i + 1)
                    with open(vis_paths_to_indexes_path, "w") as f:
                        json.dump(self.vis_paths_to_indexes, f)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def resize_image(self, im, force_load_size=None):
        """Resizes image for the training.
        This function resizes the original image to target shape(self.img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        try:
            h0, w0 = im.shape[:2]  # origin shape
            if force_load_size:
                r = force_load_size / max(h0, w0)
            else:
                r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 and not self.augment
                    else cv2.INTER_LINEAR,
                )
        except:
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

    def toh5pyfiles(self):
        hf = h5py.File(f"/data/storage/datasets/gen4_final/{self.mode}.h5", "w")
        print(len(self) // 10)

        for idx in range(len(self) // 10):
            print(idx)

            bbox_file = os.path.join(
                self.root, self.mode, "labels", self.label_files[idx]
            )
            event_file = os.path.join(
                self.root, self.mode, "events", self.event_files[idx]
            )
            parts = event_file.split("/")
            group_name = "/".join([parts[6], parts[8], parts[9].split(".")[0]])
            g = hf.create_group(group_name)

            labels_np = np.load(bbox_file)
            events_np = np.load(event_file)
            for npz_num in range(len(labels_np)):
                try:
                    ev_npz = "e" + str(npz_num)
                    lb_npz = "l" + str(npz_num)
                    events_np_ = events_np[ev_npz]
                    labels_np_ = labels_np[lb_npz]
                except:  # avoid error: Bad CRC-32 for file
                    ev_npz = "e" + str(npz_num - 1)
                    lb_npz = "l" + str(npz_num - 1)
                    events_np_ = events_np[ev_npz]
                    labels_np_ = labels_np[lb_npz]

                mask = (events_np_["x"] < 1280) * (
                    events_np_["y"] < 720
                )  # filter events which are out of bounds
                events_np_ = events_np_[mask]

                labels = rfn.structured_to_unstructured(labels_np_)[
                    :, [5, 1, 2, 3, 4]
                ]  # (class_id, x, y, w, h)

                events = rfn.structured_to_unstructured(events_np_)[
                    :, [1, 2, 0, 3]
                ]  # (x, y, t, p)

                labels = self.cropToFrame(labels)
                labels = self.filter_boxes(labels, 60, 20)  # filter small boxes

                labels[:, 1] /= self.width
                labels[:, 2] /= self.height
                labels[:, 3] /= self.width
                labels[:, 4] /= self.height
                labels[:, 1:3] += 0.5 * labels[:, 3:5]
                labels = labels[~(labels[:, 0] > 2)]  # filter out class_id > 2

                g.create_dataset(f"{ev_npz}", data=events)
                g.create_dataset(f"{lb_npz}", data=labels)

        # a = h5.get('trainfilelist00/moorea_2019-02-18_000_td_427500000_487500000/ev040/e0') ili l0
        hf.close()

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

    def process_dataset(self):
        NUMBER_OF_EVENTS = 70000
        hf = h5py.File(f"/data/storage/datasets/gen4/{self.mode}.h5", "w")

        counter = 0
        events_buffer = np.empty((0, 5))  # (0, 4) for events, +1 for labels idx

        for idx in range(len(self)):
            print(idx)
            event_file = self.event_files[idx]
            events = np.array(self.hf.get(event_file))
            labels = np.array(self.hf.get(self.label_files[idx]))

            if len(list(labels[:, 0].astype("int"))) == 0:
                continue

            if idx == 0:
                events_extended = np.c_[events, np.ones(events.shape[0]) * idx].astype(
                    "uint32"
                )
                events_buffer = events_extended
            else:
                events_extended = np.c_[events, np.ones(events.shape[0]) * idx].astype(
                    "uint32"
                )
                events_buffer = np.concatenate((events_buffer, events_extended), axis=0)

            division = len(events_buffer) / NUMBER_OF_EVENTS

            if division > 1:
                for _ in range(int(division)):
                    events_save = events_buffer[:NUMBER_OF_EVENTS][:, :4]
                    labels_ids = list(set(list(events_buffer[:, 4])))

                    labs_save = np.empty((0, 5))
                    for li, lab_id in enumerate(labels_ids):
                        if li == 0:
                            labs = np.array(self.hf.get(self.label_files[int(lab_id)]))
                            labs_save = labs
                        else:
                            labs = np.array(self.hf.get(self.label_files[int(lab_id)]))
                            labs_save = np.concatenate((labs_save, labs), axis=0)

                    # save it
                    group_name = f"{counter}"
                    g = hf.create_group(group_name)
                    g.create_dataset("events", data=events_save)
                    g.create_dataset("labels", data=labs_save)

                    counter += 1
                    events_buffer = np.delete(
                        events_buffer, np.s_[:NUMBER_OF_EVENTS], axis=0
                    )

        # if events buffer is still not empty, get remaining events and labels and save them
        if len(events_buffer) > 0:
            events_save = events_buffer[:, :4]
            labels_ids = list(set(list(events_buffer[:, 4])))

            labs_save = np.empty((0, 5))
            for li, lab_id in enumerate(labels_ids):
                if li == 0:
                    labs = np.array(self.hf.get(self.label_files[int(lab_id)]))
                    labs_save = labs
                else:
                    labs = np.array(self.hf.get(self.label_files[int(lab_id)]))
                    labs_save = np.concatenate((labs_save, labs), axis=0)

            # save it
            group_name = f"{counter}"
            g = hf.create_group(group_name)
            g.create_dataset("events", data=events_save)
            g.create_dataset("labels", data=labs_save)

        hf.close()

    def fill_vis(self, idx):
        unique_identifier = str(idx)
        self.vis_paths_to_indexes[unique_identifier] = idx

    def downsample_event_stream(self, events):
        events[:, 0] = events[:, 0] / 1280 * self.img_size  # x
        events[:, 1] = events[:, 1] / 720 * self.img_size  # y
        delta_t = events[-1, 2] - events[0, 2]
        events[:, 2] = 4 * (events[:, 2] - events[0, 2]) / delta_t

        _, ev_idx = np.unique(events[:, :2], axis=0, return_index=True)
        downsample_events = events[ev_idx]
        ev = downsample_events[np.argsort(downsample_events[:, 2])]
        return ev

    def __getitem__(self, idx):
        """
        returns events and label, loading events from split .npy files
        :param idx:
        :return: events: (x, y, t, p)
                 boxes: (N, 4), which is consist of (x_min, y_min, x_max, y_max)
                 histogram: (512, 512, 10)
        """
        with h5py.File(
            os.path.join(self.reps, self.files[idx].replace(".npy", ".h5"))
        ) as fh:
            rep = fh["repr"][()]

        labels_path = os.path.join(self.labels, self.files[idx])
        labels = np.load(labels_path)

        # Letterbox
        img, (h0, w0), (h, w) = self.resize_image(rep)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        shape = (
            self.batch_shapes[self.batch_indices[idx]] if self.rect else self.img_size
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

        # labels = labels.copy()

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

        unique_identifier = str(idx)
        self.vis_paths_to_indexes[unique_identifier] = idx

        # return rep, return_data
        return (
            torch.from_numpy(img),
            labels_out,
            unique_identifier,
            shapes,
        )  # event representation, labels, representation path, shapes

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        boxes = []
        for box in np_bbox:
            # if box[2] > 1280 or box[3] > 800:  # filter error label
            if box[3] > 1280:
                continue

            if box[1] < 0:  # x < 0 & w > 0
                box[3] += box[1]
                box[1] = 0
            if box[2] < 0:  # y < 0 & h > 0
                box[4] += box[2]
                box[2] = 0
            if box[1] + box[3] > self.width:  # x+w>1280
                box[3] = self.width - box[1]
            if box[2] + box[4] > self.height:  # y+h>720
                box[4] = self.height - box[2]

            if (
                box[3] > 0
                and box[4] > 0
                and box[1] < self.width
                and box[2] <= self.height
            ):
                boxes.append(box)
        boxes = np.array(boxes).reshape(-1, 5)
        return boxes

    def filter_boxes(self, boxes, min_box_diag=60, min_box_side=20):
        """Filters boxes according to the paper rule.
        To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
        To note: we assume the initial time of the video is always 0
        :param boxes: (np.ndarray)
                     structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence']
                     (example BBOX_DTYPE is provided in src/box_loading.py)
        Returns:
            boxes: filtered boxes
        """
        width = boxes[:, 3]
        height = boxes[:, 4]
        diag_square = width**2 + height**2
        mask = (
            (diag_square >= min_box_diag**2)
            * (width >= min_box_side)
            * (height >= min_box_side)
        )
        return boxes[mask]

    def load_data_files(self):
        files = os.listdir(self.labels)
        files.sort(key=lambda x: int(x.split(".")[0]))

        return files

    def get_element(self, idx):
        labels_path = os.path.join(self.labels, self.files[idx])
        labels = np.load(labels_path)

        return (
            labels,
            str(idx),
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
        print("Convert to COCO format")

        for i in range(dataset_size):  # (img_path, info)
            print(i)
            labels, name = self.get_element(i)
            dataset["images"].append(
                {
                    "file_name": name,
                    "id": name,
                    "width": self.img_size,
                    "height": self.img_size,
                }
            )
            if list(labels):
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * self.img_size
                    y1 = (y - h / 2) * self.img_size
                    x2 = (x + w / 2) * self.img_size
                    y2 = (y + h / 2) * self.img_size
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
            print(f"Convert to COCO format finished. Results saved in {save_path}")

    def get_vis(self, idx):
        rep = None
        with h5py.File(
            os.path.join(self.reps, self.files[idx].replace(".npy", ".h5"))
        ) as fh:
            rep = fh["repr"][()]

        labels = np.load(os.path.join(self.labels, self.files[idx]))

        if rep.shape[2] == 2:
            rep = np.concatenate((rep, rep[:, :, 1:2]), axis=2)

        return rep[:, :, :3], labels  # return_data.bbox.cpu().numpy().copy()