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
                    epoch_number: Optional[int] = None,
                    is_final: bool = True) -> None:

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    filename = 'final.png' if is_final else f'epoch_{epoch_number:03}.png'
    filepath = os.path.join(directory, filename)

    print(f'Saving comparison to:\n\t{filepath}')
    save_image(comparison, filepath)


def create_comparison(image_pyramid: ImagePyramid, disparities: ImagePyramid,
                      recon_pyramid: ImagePyramid,
                      device: Device = 'cpu') -> Tensor:

    # Get the largest scale image from the pyramids
    left_image, right_image = torch.split(image_pyramid[0], [3, 3], 1)
    disparity, uncertainty = torch.split(disparities[0], [2, 2], 1)
    left_disp, right_disp = torch.split(disparity, [1, 1], 1)
    left_error, right_error = torch.split(uncertainty, [1, 1], 1)
    left_recon, right_recon = torch.split(recon_pyramid[0], [3, 3], 1)

    # Take the first images in the batch
    left_image, right_image = left_image[0], right_image[0]
    left_disp, right_disp = left_disp[0], right_disp[0]
    left_recon, right_recon = left_recon[0], right_recon[0]

    # Find max/min for increasing contrast
    max_disp, min_disp = disparity.max(), disparity.min()
    max_error, min_error = uncertainty.max(), uncertainty.min()

    left_disp_heatmap = u.to_heatmap(left_disp, device)
    right_disp_heatmap = u.to_heatmap(right_disp, device)

    scaled_left_disp = (left_disp - min_disp) / (max_disp - min_disp)
    scaled_right_disp = (right_disp - min_disp) / (max_disp - min_disp)
    scaled_left_disp_heatmap = u.to_heatmap(scaled_left_disp, device)
    scaled_right_disp_heatmap = u.to_heatmap(scaled_right_disp, device)

    left_error_heatmap = u.to_heatmap(left_error, device)
    right_error_heatmap = u.to_heatmap(right_error, device)

    scaled_left_error = (left_error - min_error) / (max_error - min_error)
    scaled_right_error = (right_error - min_error) / (max_error - min_error)
    scaled_left_error_heatmap = u.to_heatmap(scaled_left_error, device)
    scaled_right_error_heatmap = u.to_heatmap(scaled_right_error, device)

    grid = torch.stack((
        left_image, right_image, left_recon, right_recon,
        left_disp_heatmap, right_disp_heatmap,
        scaled_left_disp_heatmap, scaled_right_disp_heatmap,
        left_error_heatmap, right_error_heatmap,
        scaled_left_error_heatmap, scaled_right_error_heatmap))

    return make_grid(grid, nrow=4)


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
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

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
        disc_loss_string = f'{disc_loss_per_image:.2e}' \
            if disc_loss_per_image is not None else None

        print(f'{description}:'
              f'\n\tmodel loss: {model_loss_per_image:.2e}'
              f'\n\tdiscriminator loss: {disc_loss_string}'
              f'\n\tdisparity scale: {scale:.2f}')

    return model_loss_per_image, disc_loss_per_image
