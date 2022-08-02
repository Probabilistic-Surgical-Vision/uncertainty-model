import os.path
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from . import utils as u
from .utils import Device, ImagePyramid


def save_comparison(comparison: Tensor, directory: str,
                    epoch: Optional[int] = None,
                    is_final: bool = True) -> None:

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    filename = 'final.png' if is_final else f'epoch_{epoch+1}.png'
    filepath = os.path.join(directory, filename)

    save_image(comparison, filepath)


def create_comparison(image_pyramid: ImagePyramid, disparities: ImagePyramid,
                      recon_pyramid: ImagePyramid,
                      device: Device = 'cpu') -> Tensor:

    # Get the largest scale image from the pyramids
    left_image, right_image = torch.split(image_pyramid[0], [3, 3], 1)
    left_disp, right_disp = torch.split(disparities[0], [1, 1], 1)
    left_recon, right_recon = torch.split(recon_pyramid[0], [3, 3], 1)

    left_heat_disp = u.to_heatmap(left_disp[0], device)
    right_heat_disp = u.to_heatmap(right_disp[0], device)

    # Combine disparity in stereo and increase contrast
    disp = u.combine_disparity(left_disp[0], right_disp[0], device)
    scaled_disp = (disp - disp.min()) / (disp.max() - disp.min())

    heat_disp = u.to_heatmap(disp, device)
    scaled_heat_disp = u.to_heatmap(scaled_disp, device)

    grid = torch.stack((
        left_image[0], right_image[0],
        left_heat_disp, right_heat_disp,
        heat_disp, scaled_heat_disp,
        left_recon[0], right_recon[0]))

    return make_grid(grid, nrow=2)


@torch.no_grad()
def evaluate_model(model: Module, loader: DataLoader,
                   loss_function: Module, scale: float = 1.0,
                   save_evaluation_to: Optional[str] = None,
                   epoch: Optional[int] = None, is_final: bool = True,
                   scales: int = 4, device: Device = 'cpu',
                   no_pbar: bool = False) -> float:

    running_loss = 0

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch', disable=no_pbar)

    for i, image_pair in enumerate(tepoch):
        left = image_pair["left"].to(device)
        right = image_pair["right"].to(device)

        images = torch.cat([left, right], dim=1)
        image_pyramid = u.scale_pyramid(images, scales)

        disparities = model(left, scale)

        recon_pyramid = u.reconstruct_pyramid(disparities, image_pyramid)
        loss = loss_function(image_pyramid, disparities, recon_pyramid, i)

        running_loss += loss.item()

        average_loss_per_image = running_loss / ((i+1) * batch_size)
        tepoch.set_postfix(loss=average_loss_per_image)

        if save_evaluation_to is not None and i == 0:
            comparison = create_comparison(image_pyramid, disparities,
                                           recon_pyramid, device)

            save_comparison(comparison, save_evaluation_to, epoch, is_final)

    if no_pbar:
        print(f"{description}:"
              f"\n\tmodel loss: {average_loss_per_image:.2e}"
              f"\n\tdisparity scale: {scale:.2f}")

    return average_loss_per_image
