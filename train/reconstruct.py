import torch
import torch.nn.functional as F
from torch import Tensor


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
