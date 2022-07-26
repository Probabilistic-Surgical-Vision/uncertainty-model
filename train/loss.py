from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .reconstruct import reconstruct_left_image, reconstruct_right_image


class MonodepthLoss(nn.Module):

    def __init__(self, scales: int = 4, ssim_weight: float = 0.85,
                 smoothness_weight: float = 1.0,
                 consistency_weight: float = 1.0) -> None:

        super().__init__()

        self.scales = scales

        self.ssim_weight = ssim_weight
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight

        self.l1_weight = 1 - self.ssim_weight

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.repr_loss = 0
        self.con_loss = 0
        self.smooth_loss = 0

        self.disparities = []
        self.reconstructions = []

    def scale_pyramid(self, x: Tensor,
                      scales: Optional[int] = None) -> List[Tensor]:

        scales = self.scales if scales is None else scales
        _, _, height, width = x.size()

        pyramid = []

        for i in range(scales):
            ratio = 2 ** i

            size = (height // ratio, width // ratio)
            x_resized = F.interpolate(x, size=size, mode='bilinear',
                                      align_corners=True)

            pyramid.append(x_resized)

        return pyramid

    def gradient_x(self, x: Tensor) -> Tensor:
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 1, 0, 0), mode='replicate')
        return x[:, :, :, :-1] - x[:, :, :, 1:]

    def gradient_y(self, x: Tensor) -> Tensor:
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        return x[:, :, :-1, :] - x[:, :, 1:, :]

    def apply_disparity(self, x: Tensor, disparity: Tensor):
        batch_size, _, height, width = x.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width) \
            .repeat(batch_size, height, 1) \
            .type_as(x)

        y_base = torch.linspace(0, 1, height) \
            .repeat(batch_size, width, 1) \
            .transpose(1, 2) \
            .type_as(x)

        # Apply shift in X direction
        x_shifts = disparity.squeeze(dim=1)

        # In grid_sample coordinates are assumed to be between -1 and 1
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        flow_field = (2 * flow_field) - 1

        output = F.grid_sample(x, flow_field, mode='bilinear',
                               padding_mode='zeros')

        return output

    def reconstruct_left(self, right: Tensor, disparity: Tensor) -> Tensor:
        return self.apply_disparity(right, -disparity)

    def reconstruct_right(self, left: Tensor, disparity: Tensor) -> Tensor:
        return self.apply_disparity(left, disparity)

    def l1_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return (x - y).abs().mean()

    def ssim(self, x: Tensor, y: Tensor, k1: float = 0.01,
             k2: float = 0.03) -> Tensor:

        luminance_x = self.pool(x)
        luminance_y = self.pool(y)

        luminance_xx = luminance_x * luminance_x
        luminance_yy = luminance_y * luminance_y
        luminance_xy = luminance_x * luminance_y

        contrast_x = self.pool(x * x) - luminance_xx
        contrast_y = self.pool(y * y) - luminance_yy

        contrast_xy = self.pool(x * y) - luminance_xy

        numerator = ((2 * luminance_xy) + k1) * ((2 * contrast_xy) + k2)

        denominator = (luminance_xx + luminance_yy + k1) \
            * (contrast_x + contrast_y + k2)

        # Check whether this outputs a tensor or a float
        return torch.clamp(numerator / denominator, 0, 1)

    def dssim(self, x: Tensor, y: Tensor) -> Tensor:
        return (1 - self.ssim(x, y)) / 2

    def consistency_loss(self, disparity: Tensor,
                         lr_disparity: Tensor) -> Tensor:

        return self.l1_loss(disparity, lr_disparity)

    def smoothness_weights(self, image_gradient: Tensor) -> Tensor:
        return torch.exp(-image_gradient.abs().mean(dim=1, keepdim=True))

    def smoothness_loss(self, disparity: Tensor, image: Tensor) -> Tensor:
        disp_grad_x = self.gradient_x(disparity)
        disp_grad_y = self.gradient_y(disparity)

        image_grad_x = self.gradient_x(image)
        image_grad_y = self.gradient_y(image)

        weights_x = self.smoothness_weights(image_grad_x)
        weights_y = self.smoothness_weights(image_grad_y)

        smoothness_x = disp_grad_x * weights_x
        smoothness_y = disp_grad_y * weights_y

        return smoothness_x.abs() + smoothness_y.abs()

    def total_loss(self, left_image: Tensor, right_image: Tensor,
                   disparity: Tensor) -> Tuple[3 * (float,)]:

        # Split the channels into 4 tensors
        left_disp, right_disp = torch.split(disparity, [1, 1], 1)

        self.disparities.append((left_disp, right_disp))

        # Reconstruct the images using disparity and opposite view
        left_recon = reconstruct_left_image(right_image, left_disp)
        right_recon = reconstruct_right_image(left_image, right_disp)
        self.reconstructions.append((left_recon, right_recon))

        # Reconstruct disparity using disparity and opposite disparity
        left_lr_disp = reconstruct_left_image(right_disp, left_disp)
        right_lr_disp = reconstruct_right_image(left_disp, right_disp)

        # Get L1 Loss between reconstructed and original images
        l1_left_loss = self.l1_loss(left_recon, left_image)
        l1_right_loss = self.l1_loss(right_recon, right_image)

        # Get SSIM between the reconstructed and original images
        ssim_left_loss = self.dssim(left_recon, left_image)
        ssim_right_loss = self.dssim(right_recon, right_image)

        # Combine L1 and SSIM to get Weighted SSIM Metric
        repr_left_loss = (l1_left_loss * self.l1_weight) \
            + (ssim_left_loss * self.ssim_weight)

        repr_right_loss = (l1_right_loss * self.l1_weight) \
            + (ssim_right_loss * self.ssim_weight)

        # Get Consistency Loss
        con_left_loss = self.consistency_loss(left_disp, left_lr_disp)
        con_right_loss = self.consistency_loss(right_disp, right_lr_disp)

        # Get Smoothness Loss
        smooth_left_loss = self.smoothness_loss(left_disp, left_image)
        smooth_right_loss = self.smoothness_loss(right_disp, right_image)

        # Sum losses from both views
        repr_loss = torch.sum(repr_left_loss + repr_right_loss)
        con_loss = torch.sum(con_left_loss + con_right_loss)
        smooth_loss = torch.sum(smooth_left_loss + smooth_right_loss)

        return repr_loss, con_loss, smooth_loss

    def forward(self, left_image: Tensor, right_image: Tensor,
                disparities: Tuple[Tensor]) -> float:

        left_pyramid = self.scale_pyramid(left_image)
        right_pyramid = self.scale_pyramid(right_image)

        self.repr_loss = 0
        self.con_loss = 0
        self.smooth_loss = 0

        self.disparities = []
        self.reconstructions = []

        scales = zip(left_pyramid, right_pyramid, disparities)

        for left, right, disparity in scales:
            (repr_loss, con_loss,
             smooth_loss) = self.total_loss(left, right, disparity)

            self.repr_loss += repr_loss
            self.con_loss += con_loss
            self.smooth_loss += smooth_loss

        return self.repr_loss \
            + (self.con_loss * self.consistency_weight) \
            + (self.smooth_loss * self.smoothness_weight)
