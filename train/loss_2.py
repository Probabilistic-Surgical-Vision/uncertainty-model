from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .reconstruct import reconstruct_left_image, reconstruct_right_image

TensorPair = Tuple[Tensor, Tensor]


class WeightedSSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, k1: float = 0.01,
                 k2: float = 0.03) -> None:

        super().__init__()

        self.alpha = alpha
        self.k1 = k1
        self.k2 = k2

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

    def l1_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return (x - y).abs().mean()

    def ssim(self, x: Tensor, y: Tensor) -> Tensor:

        luminance_x = self.pool(x)
        luminance_y = self.pool(y)

        luminance_xx = luminance_x * luminance_x
        luminance_yy = luminance_y * luminance_y
        luminance_xy = luminance_x * luminance_y

        contrast_x = self.pool(x * x) - luminance_xx
        contrast_y = self.pool(y * y) - luminance_yy

        contrast_xy = self.pool(x * y) - luminance_xy

        numerator = ((2 * luminance_xy) + self.k1) * ((2 * contrast_xy) + self.k2)

        denominator = (luminance_xx + luminance_yy + self.k1) \
            * (contrast_x + contrast_y + self.k2)

        return torch.clamp(numerator / denominator, 0, 1)

    def dssim(self, x: Tensor, y: Tensor) -> Tensor:
        return (1 - self.ssim(x, y)) / 2

    def forward(self, original: TensorPair,
                reconstructed: TensorPair) -> Tensor:

        left_image, right_image = original
        left_recon, right_recon = reconstructed

        left_l1_loss = self.l1_loss(left_image, left_recon)
        right_l1_loss = self.l1_loss(right_image, right_recon)

        left_ssim_loss = self.dssim(left_image, left_recon)
        right_ssim_loss = self.dssim(right_image, right_recon)

        total_l1_loss = torch.sum(left_l1_loss + right_l1_loss)
        total_ssim_loss = torch.sum(left_ssim_loss + right_ssim_loss)

        return (self.alpha * total_ssim_loss) \
            + ((1 - self.alpha) * total_l1_loss)


class ConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def l1_loss(self, x: Tensor, y: Tensor) -> Tensor:
        return (x - y).abs().mean()    

    def forward(self, disparities: TensorPair) -> Tensor:
        left_disp, right_disp = disparities

        left_lr_disp = reconstruct_left_image(left_disp, right_disp)
        right_lr_disp = reconstruct_right_image(right_disp, left_disp)

        left_con_loss = self.l1_loss(left_disp, left_lr_disp)
        right_con_loss = self.l1_loss(right_disp, right_lr_disp)

        return torch.sum(left_con_loss + right_con_loss)


class SmoothnessLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def gradient_x(self, x: Tensor) -> Tensor:
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 1, 0, 0), mode='replicate')
        return x[:, :, :, :-1] - x[:, :, :, 1:]

    def gradient_y(self, x: Tensor) -> Tensor:
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        return x[:, :, :-1, :] - x[:, :, 1:, :]

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

    def forward(self, disparities: TensorPair, images: TensorPair) -> Tensor:
        left_disp, right_disp = disparities
        left_image, right_image = images

        smooth_left_loss = self.smoothness_loss(left_disp, left_image)
        smooth_right_loss = self.smoothness_loss(right_disp, right_image)

        return torch.sum(smooth_left_loss + smooth_right_loss)


class AdversarialLoss(nn.Module):
    def __init__(self, loss: str = 'mse') -> None:
        super().__init__()

        self.loss = nn.MSELoss() \
            if loss == 'mse' else nn.BCELoss()

    def forward(self, pred: Tensor, truth: Tensor) -> Tensor:
        return self.loss(pred, truth)


class PerceptualLoss:
    pass


class MultiScaleLoss(nn.Module):
    def __init__(self, scales: int = 4, wssim_weight: float = 0.85,
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 adversarial_weight: float = 0.85,
                 wssim_alpha: float = 0.85) -> None:

        super().__init__()

        self.wssim = WeightedSSIMLoss(wssim_alpha)

        self.consistency = ConsistencyLoss()
        self.smoothness = SmoothnessLoss()
        self.adversarial = AdversarialLoss()

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