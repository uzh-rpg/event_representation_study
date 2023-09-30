import os
import numpy as np
from numpy.lib import recfunctions as rfn
from torchvision import transforms
import torch
import cv2
import numba as nb

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # close mulit-processing of open-cv


def getDataloader(name):
    dataset_dict = {"Prophesee": Prophesee}
    return dataset_dict.get(name)


class Prophesee:
    def __init__(
        self,
        root,
        object_classes,
        height,
        width,
        mode="training",
        voxel_size=None,
        max_num_points=None,
        max_voxels=None,
        resize=None,
        num_bins=None,
    ):
        """
        Creates an iterator over the Prophesee object recognition dataset.
        :param root: path to dataset root
        :param object_classes: list of string containing objects or "all" for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param mode: "training", "testing" or "validation"
        :param voxel_size:
        :param max_num_points:
        :param max_voxels:
        :param num_bins:
        """
        if mode == "training":
            mode = "train"
        elif mode == "validation":
            mode = "val"
        elif mode == "testing":
            mode = "test"

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height

        self.voxel_size = voxel_size
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.num_bins = num_bins

        self.resize = resize
        self.max_nr_bbox = 60

        filelist_path = os.path.join(self.root, self.mode)

        self.event_files, self.label_files, self.index_files = self.load_data_files(
            filelist_path, self.root, self.mode
        )

        assert len(self.event_files) == len(self.label_files)

        self.object_classes = object_classes
        self.nr_classes = len(self.object_classes)  # 7 classes

        self.nr_samples = len(self.event_files)
        # self.nr_samples = len(self.event_files) - len(self.index_files)*batch_size

    def __len__(self):
        return len(self.event_files)

    @nb.jit()
    def __getitem__(self, idx):
        """
        returns events and label, loading events from split .npy files
        :param idx:
        :return: events: (x, y, t, p)
                 boxes: (N, 4), which is consist of (x_min, y_min, x_max, y_max)
                 histogram: (512, 512, 10)
        """
        boxes_list, pos_event_list, neg_event_list = [], [], []
        bbox_file = os.path.join(self.root, self.mode, "labels", self.label_files[idx])
        event_file = os.path.join(self.root, self.mode, "events", self.event_files[idx])

        labels_np = np.load(bbox_file)
        events_np = np.load(event_file)
        for npz_num in range(len(labels_np)):
            const_size_box = np.ones([self.max_nr_bbox, 5]) * -1
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
                :, [1, 2, 3, 4, 5]
            ]  # (x, y, w, h, class_id)
            events = rfn.structured_to_unstructured(events_np_)[
                :, [1, 2, 0, 3]
            ]  # (x, y, t, p)

            labels = self.cropToFrame(labels)
            labels = self.filter_boxes(labels, 60, 20)  # filter small boxes

            # select 16.6ms event streams
            # delta_t = (events[-1, 2] - events[0, 2]) / 3
            # flag_t = events[0, 2] + delta_t
            # mask_t = (events[:, 2] < flag_t)
            # events = events[mask_t]  # 16.6ms event streams

            # downsample and resolution=1280x720 -> resolution=512x512
            events = self.downsample_event_stream(events)

            labels[:, 2] += labels[:, 0]
            labels[:, 3] += labels[:, 1]  # [x1, y1, x2, y2, class]
            labels[:, 0] /= 1280
            labels[:, 1] /= 720
            labels[:, 2] /= 1280
            labels[:, 3] /= 720

            labels[:, :4] *= 512
            labels[:, 2] -= labels[:, 0]
            labels[:, 3] -= labels[:, 1]

            labels[:, 2:-1] += labels[:, :2]  # [x_min, y_min, x_max, y_max, class_id]

            pos_events = events[events[:, -1] == 1.0]
            neg_events = events[events[:, -1] == 0.0]
            pos_events = pos_events.astype(np.float32)
            neg_events = neg_events.astype(np.float32)
            if not len(neg_events):  # empty
                neg_events = pos_events
            if not len(pos_events):  # empty
                pos_events = neg_events

            pos_voxels, pos_coordinates, pos_num_points = self.voxel_generator.generate(
                pos_events[:, :3], self.max_voxels
            )
            neg_voxels, neg_coordinates, neg_num_points = self.voxel_generator.generate(
                neg_events[:, :3], self.max_voxels
            )

            boxes = labels.astype(np.float32)
            const_size_box[: boxes.shape[0], :] = boxes
            boxes_list.append(const_size_box.astype(np.float32))
            pos_event_list.append(
                [
                    torch.from_numpy(pos_voxels),
                    torch.from_numpy(pos_coordinates),
                    torch.from_numpy(pos_num_points),
                ]
            )
            neg_event_list.append(
                [
                    torch.from_numpy(neg_voxels),
                    torch.from_numpy(neg_coordinates),
                    torch.from_numpy(neg_num_points),
                ]
            )

        boxes = np.array(boxes_list)
        return boxes, pos_event_list, neg_event_list

    def downsample_event_stream(self, events):
        events[:, 0] = events[:, 0] / 1280 * 512  # x
        events[:, 1] = events[:, 1] / 720 * 512  # y
        delta_t = events[-1, 2] - events[0, 2]
        events[:, 2] = 4 * (events[:, 2] - events[0, 2]) / delta_t

        _, ev_idx = np.unique(events[:, :2], axis=0, return_index=True)
        downsample_events = events[ev_idx]
        ev = downsample_events[np.argsort(downsample_events[:, 2])]
        return ev

    def normalize(self, histogram):
        """standard normalize"""
        nonzero_ev = histogram != 0
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = histogram.sum() / num_nonzeros
            stddev = np.sqrt((histogram**2).sum() / num_nonzeros - mean**2)
            histogram = nonzero_ev * (histogram - mean) / (stddev + 1e-8)
        return histogram

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        boxes = []
        for box in np_bbox:
            # if box[2] > 1280 or box[3] > 800:  # filter error label
            if box[2] > 1280:
                continue

            if box[0] < 0:  # x < 0 & w > 0
                box[2] += box[0]
                box[0] = 0
            if box[1] < 0:  # y < 0 & h > 0
                box[3] += box[1]
                box[1] = 0
            if box[0] + box[2] > self.width:  # x+w>1280
                box[2] = self.width - box[0]
            if box[1] + box[3] > self.height:  # y+h>720
                box[3] = self.height - box[1]

            if (
                box[2] > 0
                and box[3] > 0
                and box[0] < self.width
                and box[1] <= self.height
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
        width = boxes[:, 2]
        height = boxes[:, 3]
        diag_square = width**2 + height**2
        mask = (
            (diag_square >= min_box_diag**2)
            * (width >= min_box_side)
            * (height >= min_box_side)
        )
        return boxes[mask]

    @staticmethod
    @nb.jit()
    def load_data_files(filelist_path, root, mode):
        idx = 0
        event_files = []
        label_files = []
        index_files = []
        filelist_dir = sorted(os.listdir(filelist_path))
        for filelist in filelist_dir:
            event_path = os.path.join(root, mode, filelist, "events")
            label_path = os.path.join(root, mode, filelist, "labels")
            data_dirs = sorted(os.listdir(event_path))

            for dirs in data_dirs:
                event_path_sub = os.path.join(event_path, dirs)
                label_path_sub = os.path.join(label_path, dirs)
                event_path_list = sorted(os.listdir(event_path_sub))
                label_path_list = sorted(os.listdir(label_path_sub))
                idx += len(event_path_list) - 1
                index_files.append(idx)

                for ev, lb in zip(event_path_list, label_path_list):
                    event_root = os.path.join(event_path_sub, ev)
                    label_root = os.path.join(label_path_sub, lb)
                    event_files.append(event_root)
                    label_files.append(label_root)
        return event_files, label_files, index_files

    def file_index(self):
        return self.index_files


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None):
        image = cv2.resize(
            image, (self.size, self.size), interpolation=cv2.INTER_LINEAR
        )
        return image, boxes


if __name__ == "__main__":
    root = "/data/storage/datasets/gen4_clean"
    object_classes = ["pedestrian", "two wheeler", "car"]
    height = 720
    width = 1280
    resize = 512

    dataset = Prophesee(
        root,
        object_classes,
        height,
        width,
        mode="training",
        voxel_size=None,
        max_num_points=None,
        max_voxels=None,
        resize=resize,
        num_bins=None,
    )

    first = dataset[0]

    print("A")
