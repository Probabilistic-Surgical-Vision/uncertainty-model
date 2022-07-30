from typing import List, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

ImagePyramid = List[Tensor]
PyramidPair = Tuple[ImagePyramid, ImagePyramid]
TensorPair = Tuple[Tensor, Tensor]
Device = Union[torch.device, str]


def l1_loss(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).abs().mean()


def scale_pyramid(x: Tensor, scales: int) -> ImagePyramid:
    _, _, height, width = x.size()

    pyramid = []

    for i in range(scales):
        ratio = 2 ** i

        size = (height // ratio, width // ratio)
        x_resized = F.interpolate(x, size=size, mode='bilinear',
                                  align_corners=True)

        pyramid.append(x_resized)

    return pyramid


def reconstruct(disparity: Tensor, opposite_image: Tensor) -> Tensor:
    batch_size, _, height, width = opposite_image.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width) \
        .repeat(batch_size, height, 1) \
        .type_as(opposite_image)

    y_base = torch.linspace(0, 1, height) \
        .repeat(batch_size, width, 1) \
        .transpose(1, 2) \
        .type_as(opposite_image)

    # Apply shift in X direction
    x_shifts = disparity.squeeze(dim=1)

    # In grid_sample coordinates are assumed to be between -1 and 1
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    flow_field = (2 * flow_field) - 1

    return F.grid_sample(opposite_image, flow_field, mode='bilinear',
                         padding_mode='zeros')


def reconstruct_left_image(left_disparity: Tensor,
                           right_image: Tensor) -> Tensor:

    return reconstruct(-left_disparity, right_image)


def reconstruct_right_image(right_disparity: Tensor,
                            left_image: Tensor) -> Tensor:

    return reconstruct(right_disparity, left_image)


def reconstruct_pyramid(disparities: ImagePyramid, lefts: ImagePyramid,
                        rights: ImagePyramid) -> PyramidPair:

    left_recon_pyramid = []
    right_recon_pyramid = []

    for left, right, disparity in zip(lefts, rights, disparities):
        left_disp, right_disp = torch.split(disparity, [1, 1], 1)

        left_recon = reconstruct_left_image(left_disp, right)
        right_recon = reconstruct_right_image(right_disp, left)

        left_recon_pyramid.append(left_recon)
        right_recon_pyramid.append(right_recon)

    return left_recon_pyramid, right_recon_pyramid


def adjust_disparity_scale(epoch: int, alpha: float = 0.03,
                           beta: float = 0.15, min_scale: float = 0.3,
                           max_scale: float = 1.0) -> float:

    scale = (epoch * alpha) + beta
    return np.clip(scale, min_scale, max_scale)


def to_heatmap(x: Tensor, device: Device = 'cpu', inverse: bool = False,
               colour_map: str = 'inferno') -> Tensor:

    image = x.squeeze(0).cpu().numpy()
    image = 1 - image if inverse else image

    transform = plt.get_cmap(colour_map)
    heatmap = transform(image)[:, :, :3]  # remove alpha channel

    return torch.from_numpy(heatmap).to(device).permute(2, 0, 1)


def combine_disparity(left: Tensor, right: Tensor, device: Device = 'cpu',
                      alpha: float = 20, beta: float = 0.05) -> Tensor:

    left_disp = left.cpu().numpy()
    right_disp = np.fliplr(right.cpu().numpy())

    mean_disp = (left_disp + right_disp) / 2

    _, height, width = mean_disp.shape

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, _ = np.meshgrid(x, y)

    left_mask = 1 - np.clip(alpha * (xv - beta), 0, 1)
    right_mask = np.fliplr(left_mask)

    mean_mask = 1 - (left_mask + right_mask)

    combined_disparity = (right_mask * left_disp) \
        + (left_mask * right_disp) \
        + (mean_mask * mean_disp)

    return torch.from_numpy(combined_disparity).to(device)
