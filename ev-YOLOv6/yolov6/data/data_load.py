#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

# author: Nikola Zubic

import os
from torch.utils.data import dataloader, distributed
import sys

ROOT = os.getcwd()

from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import torch_distributed_zero_first
from .gen1_2yolo import Gen1H5
from .gen4.gen4_2yolo import Prophesee
import tonic.transforms as tonic_transforms
from pathlib import Path
from representations.event_stack import EventStack
from representations.time_surface import ToTimesurface
from representations.representation_search.mixed_density_event_stack import (
    MixedDensityEventStack,
)
from representations.tore import events2ToreFeature

representations_name_to_class = {
    "VoxelGrid": tonic_transforms.ToVoxelGrid,
    "EventHistogram": tonic_transforms.ToImage,
    "TimeSurface": ToTimesurface,
    "EventStack": EventStack,
    "OptimizedRepresentation": MixedDensityEventStack,
    "TORE": events2ToreFeature,
    "LearnedRepresentation": None,
}


def create_dataloader(
    args,
    path,
    img_size,
    batch_size,
    stride,
    hyp=None,
    augment=False,
    check_images=False,
    check_labels=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    shuffle=False,
    data_dict=None,
    task="Train",
):
    """Create general dataloader.

    Returns dataloader and dataset
    """
    if rect and shuffle:
        LOGGER.warning(
            "WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle = False

    if task == "train":
        gen1_mode = True
    else:
        gen1_mode = False

    if args.dataset == "gen1":
        with torch_distributed_zero_first(rank):
            dataset = Gen1H5(
                args=args,
                file=Path(args.file),
                training=gen1_mode,
                transform=representations_name_to_class[args.representation],
                augment=augment,
                hyp=hyp,
                rect=rect,
                rank=rank,
                task=task,
                img_size=img_size,
                data_dict=data_dict,
            )
            collate_function = Gen1H5.collate_fn

    elif args.dataset == "gen4":
        with torch_distributed_zero_first(rank):
            dataset = Prophesee(
                args=args,
                root=args.file,
                transform=representations_name_to_class[args.representation],
                mode=task,
                img_size=img_size,
                augment=augment,
                hyp=hyp,
                rect=rect,
                rank=rank,
                task=task,
                data_dict=data_dict,
            )
            collate_function = Prophesee.collate_fn

    batch_size = min(batch_size, len(dataset))
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers

    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )

    return (
        TrainValDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=workers,
            sampler=sampler,
            pin_memory=False,
            collate_fn=collate_function,
        ),
        dataset,
    )


class TrainValDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
