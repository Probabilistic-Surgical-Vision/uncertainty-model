import os.path
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from . import utils as u
from .utils import Device, ImagePyramid, PyramidPair


def save_comparison(comparison: Tensor, directory: str,
                    epoch: Optional[int] = None,
                    is_final: bool = True) -> None:

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    filename = 'final.png' if is_final else f'epoch_{epoch+1}.png'
    filepath = os.path.join(directory, filename)

    save_image(comparison, filepath)


def create_comparison(image_pyramid: PyramidPair, disparities: ImagePyramid,
                      recon_pyramid: PyramidPair,
                      device: Device = 'cpu') -> Tensor:

    left_pyramid, right_pyramid = image_pyramid
    left_recon_pyramid, right_recon_pyramid = recon_pyramid

    # Get the largest scale image from the pyramids
    left, right = left_pyramid[0], right_pyramid[0]
    left_disp, right_disp = torch.split(disparities[0], [1, 1], 1)
    left_recon, right_recon = left_recon_pyramid[0], right_recon_pyramid[0]

    disp = u.combine_disparity(left_disp[0], right_disp[0], device)
    # Scale up to increase disparity contrast
    scaled_disp = (disp - disp.min()) / (disp.max() - disp.min())

    left_disp = u.to_heatmap(left_disp[0], device)
    right_disp = u.to_heatmap(right_disp[0], device)
    disp = u.to_heatmap(disp, device)
    scaled_disp = u.to_heatmap(scaled_disp, device)

    grid = torch.stack((left[0], right[0],
                       left_disp, right_disp,
                       disp, scaled_disp,
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

        left_pyramid = u.scale_pyramid(left, scales)
        right_pyramid = u.scale_pyramid(right, scales)

        disparities = model(left, scale)

        image_pyramid = (left_pyramid, right_pyramid)
        recon_pyramid = u.reconstruct_pyramid(disparities, left_pyramid,
                                              right_pyramid)

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
