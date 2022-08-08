from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import ImagePyramid

from . import utils as u


class WeightedSSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, k1: float = 0.01,
                 k2: float = 0.03) -> None:

        super().__init__()

        self.alpha = alpha
        self.k1 = k1 ** 2
        self.k2 = k2 ** 2

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

    def ssim(self, x: Tensor, y: Tensor) -> Tensor:

        luminance_x = self.pool(x)
        luminance_y = self.pool(y)

        luminance_xx = luminance_x * luminance_x
        luminance_yy = luminance_y * luminance_y
        luminance_xy = luminance_x * luminance_y

        contrast_x = self.pool(x * x) - luminance_xx
        contrast_y = self.pool(y * y) - luminance_yy

        contrast_xy = self.pool(x * y) - luminance_xy

        numerator = ((2 * luminance_xy) + self.k1) \
            * ((2 * contrast_xy) + self.k2)

        denominator = (luminance_xx + luminance_yy + self.k1) \
            * (contrast_x + contrast_y + self.k2)

        return numerator / denominator

    def dssim(self, x: Tensor, y: Tensor) -> Tensor:
        dissimilarity = (1 - self.ssim(x, y)) / 2
        return torch.clamp(dissimilarity, 0, 1)

    def forward(self, images: Tensor, recon: Tensor) -> Tensor:
        left_l1_loss = u.l1_loss(images[:, 0:3], recon[:, 0:3])
        right_l1_loss = u.l1_loss(images[:, 3:6], recon[:, 3:6])

        left_ssim_loss = self.dssim(images[:, 0:3], recon[:, 0:3])
        right_ssim_loss = self.dssim(images[:, 3:6], recon[:, 3:6])

        ssim_loss = torch.mean(left_ssim_loss + right_ssim_loss)
        l1_loss = left_l1_loss + right_l1_loss

        return (self.alpha * ssim_loss) + ((1 - self.alpha) * l1_loss)


class ConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, disp: Tensor) -> Tensor:
        left_lr_disp = u.reconstruct_left_image(disp[:, 0:1], disp[:, 1:2])
        right_lr_disp = u.reconstruct_right_image(disp[:, 1:2], disp[:, 0:1])

        left_con_loss = u.l1_loss(disp[:, 0:1], left_lr_disp)
        right_con_loss = u.l1_loss(disp[:, 1:2], right_lr_disp)

        return left_con_loss + right_con_loss


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

    def forward(self, disp: Tensor, images: Tensor) -> Tensor:
        smooth_left_loss = self.smoothness_loss(disp[:, 0:1], images[:, 0:3])
        smooth_right_loss = self.smoothness_loss(disp[:, 1:2], images[:, 3:6])

        return torch.mean(smooth_left_loss + smooth_right_loss)


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_pyramid: ImagePyramid,
                recon_pyramid: ImagePyramid, disc: Module) -> Tensor:

        perceptual_loss = 0

        image_maps = disc.features(image_pyramid)
        recon_maps = disc.features(recon_pyramid)

        for image_map, recon_map in zip(image_maps, recon_maps):
            perceptual_loss += u.l1_loss(image_map, recon_map)

        return perceptual_loss


class AdversarialLoss(nn.Module):
    def __init__(self, loss: str = 'mse') -> None:
        super().__init__()

        self.adversarial = nn.MSELoss() \
            if loss == 'mse' else nn.BCELoss()

    def forward(self, recon_pyramid: ImagePyramid, discriminator: Module,
                is_fake: bool = True) -> Tensor:

        predictions = discriminator(recon_pyramid)
        labels = torch.zeros_like(predictions) \
            if is_fake else torch.ones_like(predictions)

        return self.adversarial(predictions, labels)


class ModelLoss(nn.Module):
    def __init__(self, wssim_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 adversarial_weight: float = 0.85,
                 perceptual_weight: float = 0.05,
                 wssim_alpha: float = 0.85,
                 perceptual_start: int = 5,
                 adversarial_loss_type: str = 'mse') -> None:

        super().__init__()

        self.wssim = WeightedSSIMLoss(wssim_alpha)

        self.consistency = ConsistencyLoss()
        self.smoothness = SmoothnessLoss()

        self.adversarial = AdversarialLoss(adversarial_loss_type)
        self.perceptual = PerceptualLoss()

        self.perceptual_start = perceptual_start

        self.wssim_weight = wssim_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, image_pyramid: ImagePyramid,
                disparities: Tuple[Tensor, ...],
                recon_pyramid: ImagePyramid, epoch: Optional[int] = None,
                discriminator: Optional[Module] = None) -> Tensor:

        reprojection_loss = 0
        consistency_loss = 0
        smoothness_loss = 0
        adversarial_loss = 0
        perceptual_loss = 0

        scales = zip(image_pyramid, disparities, recon_pyramid)

        for i, (images, disparity, recon_images) in enumerate(scales):
            reprojection_loss += self.wssim(images, recon_images)
            consistency_loss += self.consistency(disparity)
            smoothness_loss += self.smoothness(disparity, images) / (2 ** i)

        if discriminator is not None:
            adversarial_loss += self.adversarial(recon_pyramid, discriminator)

            if epoch is not None and epoch >= self.perceptual_start:
                perceptual_loss += self.perceptual(image_pyramid,
                                                   recon_pyramid,
                                                   discriminator)

        return reprojection_loss * self.wssim_weight \
            + (consistency_loss * self.consistency_weight) \
            + (smoothness_loss * self.smoothness_weight) \
            + (adversarial_loss * self.adversarial_weight) \
            + (perceptual_loss * self.perceptual_weight)
