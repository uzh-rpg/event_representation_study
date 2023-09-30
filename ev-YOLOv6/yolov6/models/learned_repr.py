import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=12):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(
        self,
        dim,
        mlp_layers=[1, 100, 100, 1],
        activation=nn.LeakyReLU(negative_slope=0.1),
        image_size=640,
    ):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(
            mlp_layers, activation=activation, num_channels=dim[0]
        ).to("cuda")

        self.dim = dim
        self.image_size = image_size

    @staticmethod
    def letterbox_image_batch(image_batch, size, color=114):
        """
        Resize image batch with unchanged aspect ratio using padding.

        Args:
            image_batch (torch.Tensor): The image batch to be resized.
            size (int): The desired size.

        Returns:
            torch.Tensor: The resized image batch.
        """
        # Assume image_batch shape is (batch, channel, height, width)
        batch_size, c, orig_h, orig_w = image_batch.shape

        # Calculate the scale factor, while maintaining aspect ratio
        scale = min(size / orig_w, size / orig_h)

        # Calculate the new size and resize images
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized_images = torch.nn.functional.interpolate(
            image_batch, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        # Create a new tensor filled with 0 (black) for new images
        new_images = torch.full(
            (batch_size, c, size, size),
            fill_value=color,
            dtype=image_batch.dtype,
            device=image_batch.device,
        )

        # Calculate the position to paste the images on the new canvas
        top = (size - new_h) // 2
        left = (size - new_w) // 2

        # Paste the images onto the canvas
        new_images[:, :, top : top + new_h, left : left + new_w] = resized_images

        return new_images

    def crop_and_resize_to_resolution(self, x):
        x = self.letterbox_image_batch(x, self.image_size)

        return x

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1 + events[-1, -1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full(
            [
                num_voxels,
            ],
            fill_value=0,
        )
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()

        # p = (p + 1) / 2  # maps polarity to 0, 1

        idx_before_bins = x + W * y + 0 + W * H * C * p + W * H * C * 2 * b

        for i_bin in range(C):
            values = t * self.value_layer.forward(t - i_bin / (C - 1))

            # draw in a voxel grid
            idx = idx_before_bins + W * H * i_bin
            idx = torch.clamp(idx.long(), 0, np.prod(vox.shape) - 1)
            vox.put_(idx, values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        vox_cropped = self.crop_and_resize_to_resolution(vox)

        return vox_cropped.cuda().to(dtype=torch.float32)
