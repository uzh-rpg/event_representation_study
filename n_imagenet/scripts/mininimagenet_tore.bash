#!/usr/bin/env bash

source activate event_representation

python n_imagenet/real_cnn_model/main.py --config "n_imagenet/real_cnn_model/configs/imagenet/cnn_adam_twelve_channel_big_kernel_random_idx_mini.ini" --override 'loader_type=reshape_then_tore'
