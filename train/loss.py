from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from . import utils as u
from .utils import ImagePyramid, PyramidPair, TensorPair


class WeightedSSIMLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, k1: float = 0.01,
                 k2: float = 0.03) -> None:

        super().__init__()

        self.alpha = alpha
        self.k1 = k1
        self.k2 = k2

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

        left_l1_loss = u.l1_loss(left_image, left_recon)
        right_l1_loss = u.l1_loss(right_image, right_recon)

        left_ssim_loss = self.dssim(left_image, left_recon)
        right_ssim_loss = self.dssim(right_image, right_recon)

        total_l1_loss = torch.sum(left_l1_loss + right_l1_loss)
        total_ssim_loss = torch.sum(left_ssim_loss + right_ssim_loss)

        return (self.alpha * total_ssim_loss) \
            + ((1 - self.alpha) * total_l1_loss)


class ConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()    

    def forward(self, disparities: TensorPair) -> Tensor:
        left_disp, right_disp = disparities

        left_lr_disp = u.reconstruct_left_image(left_disp, right_disp)
        right_lr_disp = u.reconstruct_right_image(right_disp, left_disp)

        left_con_loss = u.l1_loss(left_disp, left_lr_disp)
        right_con_loss = u.l1_loss(right_disp, right_lr_disp)

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


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_pyramid: PyramidPair,
                recon_pyramid: PyramidPair, disc: Module) -> Tensor:

        perceptual_loss = 0
        
        left_image_pyramid, right_image_pyramid = image_pyramid
        left_recon_pyramid, right_recon_pyramid = recon_pyramid

        image_maps = disc.features(left_image_pyramid, right_image_pyramid)
        recon_maps = disc.features(left_recon_pyramid, right_recon_pyramid)

        for image_map, recon_map in zip(image_maps, recon_maps):
            perceptual_loss += u.l1_loss(image_map, recon_map)

        return perceptual_loss


class AdversarialLoss(nn.Module):
    def __init__(self, loss: str = 'mse', perceptual_start: int = 5) -> None:
        super().__init__()

        self.adversarial = nn.MSELoss() if loss == 'mse' else nn.BCELoss()
        self.perceptual = PerceptualLoss()

        self.perceptual_start = perceptual_start

    def forward(self, image_pyramid: PyramidPair, recon_pyramid: PyramidPair,
                discriminator: Module, epoch: int) -> Tensor:
        
        predictions = discriminator(*recon_pyramid)
        labels = torch.ones_like(predictions)

        loss = self.adversarial(predictions, labels)

        if epoch >= self.perceptual_start:
            loss += self.perceptual(image_pyramid, recon_pyramid,
                                    discriminator)

        return loss


class GeneratorLoss(nn.Module):
    def __init__(self, scales: int = 4, wssim_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 adversarial_weight: float = 0.85,
                 wssim_alpha: float = 0.85,
                 perceptual_start: int = 5,
                 adversarial_loss_type: str = 'mse') -> None:

        super().__init__()

        self.scales = scales

        self.wssim = WeightedSSIMLoss(wssim_alpha)

        self.consistency = ConsistencyLoss()
        self.smoothness = SmoothnessLoss()

        self.adversarial = AdversarialLoss(adversarial_loss_type,
                                           perceptual_start)

        self.wssim_weight = wssim_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
    
    def forward(self, image_pyramid: PyramidPair,
                disparities: Tuple[Tensor, ...],
                recon_pyramid: PyramidPair, epoch: int,
                discriminator: Optional[Module] = None) -> Tensor:
        
        reprojection_loss = 0
        consistency_loss = 0
        smoothness_loss = 0
        adversarial_loss = 0

        scales = zip(*image_pyramid, disparities, *recon_pyramid)

        for left, right, disparity, left_recon, right_recon in scales:
            left_disp, right_disp = torch.split(disparity, [1, 1], 1)
            
            image_tuple = (left, right)
            disp_tuple = (left_disp, right_disp)
            recon_tuple = (left_recon, right_recon)

            reprojection_loss += self.wssim(image_tuple, recon_tuple)
            consistency_loss += self.consistency(disp_tuple)
            smoothness_loss += self.smoothness(disp_tuple, image_tuple)

        if discriminator is not None:
            adversarial_loss += self.adversarial(image_pyramid, recon_pyramid,
                                                 discriminator, epoch)

        return reprojection_loss * self.wssim_weight \
            + (consistency_loss * self.consistency_weight) \
            + (smoothness_loss * self.smoothness_weight) \
            + (adversarial_loss * self.adversarial_weight)