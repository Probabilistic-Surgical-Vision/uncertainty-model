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

        numerator = ((2 * luminance_xy) + self.k1) \
            * ((2 * contrast_xy) + self.k2)

        denominator = (luminance_xx + luminance_yy + self.k1) \
            * (contrast_x + contrast_y + self.k2)

        return torch.clamp(numerator / denominator, 0, 1)

    def dssim(self, x: Tensor, y: Tensor) -> Tensor:
        return (1 - self.ssim(x, y)) / 2

    def forward(self, images: Tensor, recon: Tensor) -> Tensor:
        left_l1_loss = u.l1_loss(images[:, 0:3], recon[:, 0:3])
        right_l1_loss = u.l1_loss(images[:, 3:6], recon[:, 3:6])

        left_ssim_loss = self.dssim(images[:, 0:3], recon[:, 0:3])
        right_ssim_loss = self.dssim(images[:, 3:6], recon[:, 3:6])

        total_l1_loss = torch.sum(left_l1_loss + right_l1_loss)
        total_ssim_loss = torch.sum(left_ssim_loss + right_ssim_loss)

        return (self.alpha * total_ssim_loss) \
            + ((1 - self.alpha) * total_l1_loss)


class ConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, disp: Tensor) -> Tensor:
        left_lr_disp = u.reconstruct_left_image(disp[:, 0:1], disp[:, 1:2])
        right_lr_disp = u.reconstruct_right_image(disp[:, 1:2], disp[:, 0:1])

        left_con_loss = u.l1_loss(disp[:, 0:1], left_lr_disp)
        right_con_loss = u.l1_loss(disp[:, 1:2], right_lr_disp)

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

    def loss(self, disparity: Tensor, image: Tensor) -> Tensor:
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
        smooth_left_loss = self.loss(disp[:, 0:1], images[:, 0:3])
        smooth_right_loss = self.loss(disp[:, 1:2], images[:, 3:6])

        return torch.sum(smooth_left_loss + smooth_right_loss)


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
    def __init__(self, loss: str = 'mse', perceptual_start: int = 5) -> None:
        super().__init__()

        self.adversarial = nn.MSELoss() if loss == 'mse' else nn.BCELoss()
        self.perceptual = PerceptualLoss()

        self.perceptual_start = perceptual_start

    def forward(self, image_pyramid: ImagePyramid, recon_pyramid: ImagePyramid,
                discriminator: Module, epoch: int) -> Tensor:

        predictions = discriminator(recon_pyramid)
        labels = torch.ones_like(predictions)

        loss = self.adversarial(predictions, labels)

        if epoch >= self.perceptual_start:
            loss += self.perceptual(image_pyramid, recon_pyramid,
                                    discriminator)

        return loss


class L1ReprojectionErrorLoss(nn.Module):
    def __init__(self, include_smoothness: bool = True) -> None:
        super().__init__()

        self.include_smoothness = include_smoothness

        self.smoothness = SmoothnessLoss() \
            if include_smoothness else None
    
    def forward(self, predicted: Tensor, truth: Tensor) -> Tensor:
        left, right = torch.split(truth.detach().clone(), [3, 3], dim=1)
        
        left, right = left.mean(1, keepdim=True), right.mean(1, keepdim=True)
        truth_mean = torch.cat((left, right), dim=1)

        _, _, height, width = predicted.size()

        truth_resized = F.interpolate(truth_mean, size=(height, width),
                                      mode='bilinear', align_corners=True)

        loss = u.l1_loss(predicted, truth_resized)

        if self.include_smoothness:
            smoothness_loss = self.smoothness.loss(predicted, truth_resized)
            loss += torch.sum(smoothness_loss)

        return loss


class BayesianReprojectionErrorLoss(nn.Module):
    def __init__(self, include_smoothness: bool = True) -> None:
        super().__init__()

        self.include_smoothness = include_smoothness

        self.smoothness = SmoothnessLoss() \
            if include_smoothness else None
    
    def forward(self, predicted: Tensor, truth: Tensor) -> Tensor:
        left, right = torch.split(truth.detach().clone(), [3, 3], dim=1)
        
        left, right = left.mean(1, keepdim=True), right.mean(1, keepdim=True)
        truth_mean = torch.cat((left, right), dim=1)

        _, _, height, width = predicted.size()

        truth_resized = F.interpolate(truth_mean, size=(height, width),
                                      mode='bilinear', align_corners=True)


        loss = (truth_resized / predicted) + torch.log(predicted)

        if self.include_smoothness:
            loss += self.smoothness.loss(predicted, truth_resized)

        return torch.sum(loss) / (width * height)


class ModelLoss(nn.Module):
    def __init__(self, wssim_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 adversarial_weight: float = 0.85,
                 predictive_error_weight: float = 1.0,
                 wssim_alpha: float = 0.85,
                 perceptual_start: int = 5,
                 adversarial_loss_type: str = 'mse',
                 error_loss_type = 'l1',
                 error_smoothness: bool = True) -> None:

        super().__init__()

        self.wssim = WeightedSSIMLoss(wssim_alpha)

        self.consistency = ConsistencyLoss()
        self.smoothness = SmoothnessLoss()

        self.adversarial = AdversarialLoss(adversarial_loss_type,
                                           perceptual_start)

        self.predictive_error = L1ReprojectionErrorLoss() \
            if error_loss_type == 'l1' \
            else BayesianReprojectionErrorLoss(error_smoothness)

        self.wssim_weight = wssim_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight

        self.predictive_error_weight = predictive_error_weight

    def forward(self, image_pyramid: ImagePyramid,
                disparities: Tuple[Tensor, ...],
                recon_pyramid: ImagePyramid, epoch: int,
                discriminator: Optional[Module] = None) -> Tensor:

        reprojection_loss = 0
        consistency_loss = 0
        loss = 0
        adversarial_loss = 0

        error_loss = 0

        scales = zip(image_pyramid, disparities, recon_pyramid)

        for i, (images, disparity, recon_images) in enumerate(scales):
            disparity, uncertainty = torch.split(disparity, [2, 2], dim=1)

            reprojection_loss += self.wssim(images, recon_images)
            consistency_loss += self.consistency(disparity) / (2 ** i)
            loss += self.smoothness(disparity, images)

            error_loss += self.predictive_error(uncertainty, recon_images)

        if discriminator is not None:
            adversarial_loss += self.adversarial(image_pyramid, recon_pyramid,
                                                 discriminator, epoch)

        return reprojection_loss * self.wssim_weight \
            + (consistency_loss * self.consistency_weight) \
            + (loss * self.smoothness_weight) \
            + (adversarial_loss * self.adversarial_weight) \
            + (error_loss * self.predictive_error_weight)
