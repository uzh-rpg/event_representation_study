#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil
import wandb


def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (rank in (-1, 0)) else logging.WARNING,
    )
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors="ignore") as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, "w") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to wandb log."""
    wandb.log({"val/mAP@0.5": results[0]})
    wandb.log({"val/mAP@0.50:0.95": results[1]})

    wandb.log({"train/iou_loss": losses[0]})
    wandb.log({"train/dist_focalloss": losses[1]})
    wandb.log({"train/cls_loss": losses[2]})

    wandb.log({"x/lr0": results[2]})
    wandb.log({"x/lr1": results[3]})
    wandb.log({"x/lr2": results[4]})


def write_tbimg(tblogger, imgs, step, type="train"):
    """Display train_batch and validation predictions to tensorboard."""
    if type == "train":
        tblogger.add_image(f"train_batch", imgs, step + 1, dataformats="HWC")
    elif type == "val":
        for idx, img in enumerate(imgs):
            wandb.log({f"val_batch/{step}/{idx + 1}": wandb.Image(img.cpu().numpy())})
            # tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning("WARNING: Unknown image type to visualize.\n")
