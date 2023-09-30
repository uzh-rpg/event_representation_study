import os
import numpy as np
from numpy.lib import recfunctions as rfn
from torchvision import transforms
import torch
import cv2
from typing import Callable
import os.path as osp
import json
import sys
import h5py
import hdf5plugin
import warnings
import evlicious
import argparse
import tonic.transforms as tonic_transforms

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "representations"))
sys.path.append(os.path.join(ROOT, "ev-YOLOv6/"))

from representations.gen4_transforms import get_item_transform
from yolov6.data.data_augment import letterbox, random_affine
from yolov6.vis_utils import make_binary_histo

warnings.filterwarnings("ignore")


def _compression_opts():
    compression_level = 1  # {0, ..., 9}
    shuffle = 2  # {0: none, 1: byte, 2: bit}
    # From https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L66-L71
    # define BLOSC_BLOSCLZ   0
    # define BLOSC_LZ4       1
    # define BLOSC_LZ4HC     2
    # define BLOSC_SNAPPY    3
    # define BLOSC_ZLIB      4
    # define BLOSC_ZSTD      5
    compressor_type = 5
    compression_opts = (0, 0, 0, 0, compression_level, shuffle, compressor_type)
    return compression_opts


H5_BLOSC_COMPRESSION_FLAGS = dict(
    compression=32001, compression_opts=_compression_opts(), chunks=True  # Blosc
)


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
        self.hf = h5py.File(f"{root}{self.mode}.h5")

        cache_event_files = os.path.join(
            os.path.dirname(__file__), self.mode + "_event_files.npy"
        )
        cache_label_files = os.path.join(
            os.path.dirname(__file__), self.mode + "_label_files.npy"
        )

        if os.path.exists(cache_event_files):
            self.event_files = np.load(cache_event_files).tolist()
            self.label_files = np.load(cache_label_files).tolist()
        else:
            self.event_files, self.label_files = self.load_data_files()
            np.save(cache_event_files, self.event_files)
            np.save(cache_label_files, self.label_files)

        assert len(self.event_files) == len(self.label_files)

        self.object_classes = object_classes
        self.nr_classes = len(self.object_classes)  # 7 classes

        self.vis_paths_to_indexes = (
            {}
        )  # for visualization purposes, visualization paths to indexes in the dataset
        self.get_imgs_labels()

    def get_imgs_labels(self):
        if self.task.lower() == "val":
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
                IMG_DIR = (
                    "/data/nzubic/Projects/event_representation_study/GEN4_annotations/"
                )
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
        return len(self.event_files)

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

    def resize_image_process(self, im, force_load_size=None):
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
                        (self.img_size, self.img_size),
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
                    (self.img_size, self.img_size),
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
        event_file = self.event_files[idx]
        unique_identifier = event_file + "_" + str(idx)
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

    def process_representation(
        self, counter, save_path, event_h5_file, event_file, labels
    ):
        hf = h5py.File(event_h5_file)
        events = np.array(hf.get(event_file))  # [:50000]

        rep = None
        reshaped_return_data = events

        if self.transform is not None:
            if not "To3ChannelPseudoFrameRepresentation" in str(self.transform):
                reshaped_return_data = self.fix_events_training(events)
                rep = get_item_transform(
                    reshaped_return_data,
                    str(self.transform),
                    self.transform,
                    self.height,
                    self.width,
                    events.shape[0],
                )

        rep, _, _ = self.resize_image_process(rep)

        # save it
        rep_file_name = os.path.join(save_path, self.mode, "reps", f"{counter}.h5")
        # np.save(rep_file_name, rep)

        with h5py.File(rep_file_name, "w") as fh:
            fh.create_dataset(
                "repr", data=rep.astype("float32"), shape=rep.shape, dtype="f4"
            )  # , **H5_BLOSC_COMPRESSION_FLAGS)

        np.save(os.path.join(save_path, self.mode, "labels", f"{counter}.npy"), labels)

    def process_representations(self, save_path):
        counter = 0
        hf = self.hf
        self.hf = None

        with evlicious.tools.TaskManager(total=len(self), processes=8) as tm:
            for idx in range(len(self)):
                event_file = self.event_files[idx]
                label_file = self.label_files[idx]

                labels = np.array(hf.get(label_file))

                # If labels are empty, just skip this sample
                if len(labels) == 0:
                    continue

                event_h5_file = f"{self.root}{self.mode}.h5"
                tm.new_task(
                    self.process_representation,
                    counter,
                    save_path,
                    event_h5_file,
                    event_file,
                    labels,
                )
                # self.process_representation(counter, save_path, event_h5_file, event_file, labels)
                tm.pbar.update(1)
                counter += 1

    def __getitem__(self, idx):
        """
        returns events and label, loading events from split .npy files
        :param idx:
        :return: events: (x, y, t, p)
                 boxes: (N, 4), which is consist of (x_min, y_min, x_max, y_max)
                 histogram: (512, 512, 10)
        """
        event_file = self.event_files[idx]
        events = np.array(self.hf.get(event_file))
        labels = np.array(self.hf.get(self.label_files[idx]))

        rep = None
        reshaped_return_data = events

        if self.transform is not None:
            if not "To3ChannelPseudoFrameRepresentation" in str(self.transform):
                reshaped_return_data = self.fix_events_training(events)
                rep = get_item_transform(
                    reshaped_return_data,
                    str(self.transform),
                    self.transform,
                    self.height,
                    self.width,
                    events.shape[0],
                )

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

        unique_identifier = event_file + "_" + str(idx)
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
        event_files = []
        self.hf.visit(
            lambda key: event_files.append(key)
            if isinstance(self.hf[key], h5py.Dataset) and key[-2] == "e"
            else None
        )
        label_files = []
        self.hf.visit(
            lambda key: label_files.append(key)
            if isinstance(self.hf[key], h5py.Dataset) and key[-2] == "l"
            else None
        )

        return event_files, label_files

    def get_element(self, idx):
        event_file = self.event_files[idx]
        labels = np.array(self.hf.get(self.label_files[idx]))

        return (
            labels,
            event_file + "_" + str(idx),
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
            print(f"Convert to COCO format finished. Results saved in {save_path}")

    def get_vis(self, idx):
        events = np.array(self.hf.get(self.event_files[idx]))
        labels = np.array(self.hf.get(self.label_files[idx]))

        rep = None
        reshaped_return_data = events

        if self.transform is not None:
            if not "To3ChannelPseudoFrameRepresentation" in str(self.transform):
                reshaped_return_data = self.fix_events_training(events)
                rep = make_binary_histo(
                    reshaped_return_data, width=self.width, height=self.height
                )
            else:
                sensor_size = (self.width, self.height, 2)
                rep = self.transform(sensor_size)(
                    torch.tensor(reshaped_return_data.astype(np.int32))
                )

        return rep, labels  # return_data.bbox.cpu().numpy().copy()

    def fix_events_training(self, events):
        events = rfn.unstructured_to_structured(events)
        events.dtype = [("x", "<i4"), ("y", "<i4"), ("t", "<i4"), ("p", "<i4")]
        return events


if __name__ == "__main__":
    from representations.representation_search.mixed_density_event_stack import (
        MixedDensityEventStack,
    )
    from representations.tore import events2ToreFeature
    from representations.event_stack import EventStack
    from representations.time_surface import ToTimesurface

    representations_name_to_class = {
        "VoxelGrid": tonic_transforms.ToVoxelGrid,
        "EventHistogram": tonic_transforms.ToImage,
        "TimeSurface": ToTimesurface,
        "EventStack": EventStack,
        "OptimizedRepresentation": MixedDensityEventStack,
        "TORE": events2ToreFeature,
    }

    parser = argparse.ArgumentParser("""Preprocess the gen4 dataset""")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--rep", default="EventHistogram")
    args = parser.parse_args()

    root = "/shares/rpg.ifi.uzh/nzubic/datasets/gen4_final/"

    dataset = Prophesee(
        None, root, transform=representations_name_to_class[args.rep], mode=args.mode
    )

    dataset.process_representations(
        save_path="/shares/rpg.ifi.uzh/nzubic/datasets/gen4/VoxelGrid/"
    )

    print("==END==")
