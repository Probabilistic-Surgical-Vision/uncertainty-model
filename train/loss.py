from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import ImagePyramid

from . import utils as u


class WeightedSSIMLoss(nn.Module):
    """Calculate the SSIM/L1 loss between two images.

    Args:
        alpha (float, optional): The weight of the SSIM Loss in the overall
            metric (note that L1 weight is equal to 1 - alpha).
            Defaults to 0.85.
        k1 (float, optional): The first SSIM factor. Defaults to 0.01.
        k2 (float, optional): The second SSIM factor. Defaults to 0.03.
    """
    def __init__(self, alpha: float = 0.85, k1: float = 0.01,
                 k2: float = 0.03) -> None:

        super().__init__()

        self.alpha = alpha
        self.k1 = k1 ** 2
        self.k2 = k2 ** 2

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

    def ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate the per-pixel SSIM between two images.

        Note:
            Both images are average-pooled and therefore smaller.

        Args:
            x (Tensor): The first image to compare.
            y (Tensor): The second image to compare.

        Returns:
            Tensor: The SSIM image (reduced in size by pooling).
        """
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
        """Calculate the Structural Dissimilarity (DSSIM) between two images.

        Note:
            Both images are average-pooled and therefore smaller.

        Args:
            x (Tensor): The first image to compare.
            y (Tensor): The second image to compare.

        Returns:
            Tensor: The per-pixel DSSIM image (reduced in size by pooling).
        """
        dissimilarity = (1 - self.ssim(x, y)) / 2
        return torch.clamp(dissimilarity, 0, 1)

    def forward(self, images: Tensor, recon: Tensor) -> Tensor:
        """Calculate the weighted SSIM loss.

        This is given by:
            loss = (alpha * DSSIM) + ((1 - alpha) * L1)

        Args:
            x (Tensor): The first image to compare.
            y (Tensor): The second image to compare.

        Returns:
            Tensor: The WSSIM loss as a single float.
        """
        left_l1_loss = u.l1_loss(images[:, 0:3], recon[:, 0:3])
        right_l1_loss = u.l1_loss(images[:, 3:6], recon[:, 3:6])

        left_ssim_loss = self.dssim(images[:, 0:3], recon[:, 0:3])
        right_ssim_loss = self.dssim(images[:, 3:6], recon[:, 3:6])

        ssim_loss = torch.mean(left_ssim_loss + right_ssim_loss)
        l1_loss = left_l1_loss + right_l1_loss

        return (self.alpha * ssim_loss) + ((1 - self.alpha) * l1_loss)


class ConsistencyLoss(nn.Module):
    """Calculate the consistency loss between two disparity images.

    This is achieved by reconstructing each view of the disparity from the
    opposite image. By comparing the original disparity with the original
    image, the model learns to output similar disparity maps in both views.

    Based off:
        https://arxiv.org/abs/1609.03677
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, disp: Tensor) -> Tensor:
        """Calculate the consistency loss of the disparity prediction.

        Args:
            disp (Tensor): The stereo disparity prediction.

        Returns:
            Tensor: The consistency loss as a single float.
        """
        left_lr_disp = u.reconstruct_left_image(disp[:, 0:1], disp[:, 1:2])
        right_lr_disp = u.reconstruct_right_image(disp[:, 1:2], disp[:, 0:1])

        left_con_loss = u.l1_loss(disp[:, 0:1], left_lr_disp)
        right_con_loss = u.l1_loss(disp[:, 1:2], right_lr_disp)

        return torch.sum(left_con_loss + right_con_loss)


class SmoothnessLoss(nn.Module):
    """Calculate the smoothness loss from disparity.

    This loss function penalises the model for predicting noisy or jumpy
    disparity maps unnecessarily. Regions in the original image with little
    change in RGB are weighted higher, and multiplied by the gradient in
    disparity.

    Therefore, the loss function only penalises jagged disparity when there
    is no indication of a line or edge in the original image.

    Based off:
        https://arxiv.org/abs/1609.03677
    """
    def __init__(self) -> None:
        super().__init__()

    def gradient_x(self, x: Tensor) -> Tensor:
        """Calculate the image gradient along x."""
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 1, 0, 0), mode='replicate')
        return x[:, :, :, :-1] - x[:, :, :, 1:]

    def gradient_y(self, x: Tensor) -> Tensor:
        """Calculate the image gradient along y."""
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        return x[:, :, :-1, :] - x[:, :, 1:, :]

    def smoothness_weights(self, image_gradient: Tensor) -> Tensor:
        """Evaluate the weightings according the original image gradient."""
        return torch.exp(-image_gradient.abs().mean(dim=1, keepdim=True))

    def smoothness_loss(self, disparity: Tensor, image: Tensor) -> Tensor:
        """Calculate the smoothness loss between an image and the disparity.

        Args:
            disparity (Tensor): The (single-channel) disparity of the image.
            image (Tensor): The original image used to predict disparity.

        Returns:
            Tensor: The per-pixel smoothness loss of the disparity image.
        """
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
        """Calculate the smoothness loss between the images and disparities.

        Args:
            disp (Tensor): The stereo disparity prediction.
            images (Tensor): The original stereo images.

        Returns:
            Tensor: The smoothness loss as a single float.
        """
        smooth_left_loss = self.smoothness_loss(disp[:, 0:1], images[:, 0:3])
        smooth_right_loss = self.smoothness_loss(disp[:, 1:2], images[:, 3:6])

        return torch.mean(smooth_left_loss + smooth_right_loss)


class PerceptualLoss(nn.Module):
    """Calculate the discriminator feature reconstruction loss.

    This loss compares the reconstructed and original images by calculating
    the L1 Loss between their respective feature maps at each encoder stage
    of the discriminator.

    Based off:
        https://tinyurl.com/23jb9tnz
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_pyramid: ImagePyramid,
                recon_pyramid: ImagePyramid, disc: Module) -> Tensor:
        """Calculate the perceptual loss of the reconstructed images.

        Args:
            image_pyramid (ImagePyramid): The original stereo images.
            recon_pyramid (ImagePyramid): The reconstructed stereo images.
            disc (Module): The discriminator model to use.

        Returns:
            Tensor: The total perceptual loss as a single float.
        """
        perceptual_loss = 0

        image_maps = disc.features(image_pyramid)
        recon_maps = disc.features(recon_pyramid)

        for image_map, recon_map in zip(image_maps, recon_maps):
            perceptual_loss += u.l1_loss(image_map, recon_map)

        return perceptual_loss


class GeneratorLoss(nn.Module):
    """Calculate the loss from failing to create realistic looking images.

    The Generator needs to learn to convince the Discriminator that its
    reconstructed images are real.

    Therefore, the ground truth values must all be one. The model is then
    trained on either binary cross-entropy or mean-squared error.
    """
    def __init__(self, loss: str = 'mse') -> None:
        super().__init__()

        self.adversarial = nn.MSELoss() \
            if loss == 'mse' else nn.BCELoss()

    def forward(self, recon_pyramid: ImagePyramid,
                discriminator: Module) -> Tensor:
        """Calculate the generator loss from the reconstructed images.

        Args:
            recon_pyramid (ImagePyramid): The reconstructed stereo images.
            discriminator (Module): The discriminator.

        Returns:
            Tensor: The generator loss as a single float.
        """
        predictions = discriminator(recon_pyramid)
        labels = torch.ones_like(predictions)

        return self.adversarial(predictions, labels)


class TukraLoss(nn.Module):
    """Calculate the total loss of the model.

    For each scale of the pyramid, the loss is calculated for:
    - Reconstruction.
    - Smoothness.
    - Consistency.

    If adversarial, these are also calculated:
    - Generator Loss.
    - Discriminator Feature Reconstruction.

    Code adapted from:
        https://tinyurl.com/23jb9tnz

    Args:
        wssim_weight (float, optional): The weight of the reprojection loss.
            Defaults to 1.0.
        consistency_weight (float, optional): The weight of the consistency
            loss. Defaults to 1.0.
        smoothness_weight (float, optional): The weight of the smooothness
            loss. Defaults to 1.0.
        adversarial_weight (float, optional): The weight of the generator
            loss. Defaults to 0.85.
        perceptual_weight (float, optional): The weight of the discriminator
            feature reconstruction loss. Defaults to 0.05.
        wssim_alpha (float, optional): The weight of SSIM to L1 Loss within
            the reprojection loss. Defaults to 0.85.
        perceptual_start (int, optional): The epoch number to begin
            calculating the discriminator feature reconstruction loss.
            Defaults to 5.
        adversarial_loss_type (str, optional): The type of loss function to
            use for the generator loss. Defaults to 'mse'.
    """
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

        self.adversarial = GeneratorLoss(adversarial_loss_type)
        self.perceptual = PerceptualLoss()

        self.perceptual_start = perceptual_start

        self.wssim_weight = wssim_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, image_pyramid: ImagePyramid, disparities: ImagePyramid,
                recon_pyramid: ImagePyramid, epoch: Optional[int] = None,
                discriminator: Optional[Module] = None) -> Tensor:
        """Calculate the total loss of the model.

        Args:
            image_pyramid (ImagePyramid): The original stereo images.
            disparities (ImagePyramid): The model disparity predictions
            recon_pyramid (ImagePyramid): The reconstructed stereo images.
            epoch (Optional[int], optional): The training epoch (for
                perceptual start). Defaults to None.
            discriminator (Optional[Module], optional): The discriminator (if
                applicable). Defaults to None.

        Returns:
            Tensor: The total loss as a single float.
        """
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
