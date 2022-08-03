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

    filename = 'final.png' if is_final else f'epoch_{epoch+1:03}.png'
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
                   disc: Optional[Module] = None,
                   disc_loss_function: Optional[Module] = None, 
                   save_evaluation_to: Optional[str] = None,
                   epoch_number: Optional[int] = None,
                   is_final: bool = True,
                   scales: int = 4, device: Device = 'cpu',
                   no_pbar: bool = False, rank: int = 0) -> float:

    running_model_loss = 0
    running_disc_loss = 0

    model_loss_per_image = None
    disc_loss_per_image = None

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch',
                       disable=(no_pbar or rank > 0))

    for i, image_pair in enumerate(tepoch):
        left = image_pair["left"].to(device)
        right = image_pair["right"].to(device)

        images = torch.cat([left, right], dim=1)
        image_pyramid = u.scale_pyramid(images, scales)

        disparities = model(left, scale)

        recon_pyramid = u.reconstruct_pyramid(disparities, image_pyramid)
        model_loss = loss_function(image_pyramid, disparities,
                                   recon_pyramid, i, disc)

        if disc is not None:
            disc_loss = u.run_discriminator(image_pyramid, recon_pyramid,
                                            disc, disc_loss_function,
                                            batch_size)

        if rank > 0:
            continue

        running_model_loss += model_loss.item()
        model_loss_per_image = running_model_loss / ((i+1) * batch_size)

        if disc is not None:
            running_disc_loss += disc_loss.item()
            disc_loss_per_image = running_disc_loss / ((i+1) * batch_size)

        tepoch.set_postfix(loss=model_loss_per_image,
                           disc=disc_loss_per_image,
                           scale=scale)

        if save_evaluation_to is not None and i == 0:
            comparison = create_comparison(image_pyramid, disparities,
                                           recon_pyramid, device)

            save_comparison(comparison, save_evaluation_to,
                            epoch_number, is_final)

    if no_pbar and rank == 0:
        print(f"{description}:"
              f"\n\tmodel loss: {model_loss_per_image:.2e}"
              f"\n\tdisparity scale: {scale:.2f}")

    return model_loss_per_image, disc_loss_per_image
