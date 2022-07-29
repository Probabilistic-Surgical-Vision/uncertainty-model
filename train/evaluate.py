import os.path
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from .utils import Device, to_heatmap, \
    reconstruct_left_image, reconstruct_right_image


def save_comparison(comparison: Tensor, directory: str,
                    epoch: Optional[int] = None,
                    is_final: bool = True) -> None:

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    
    filename = 'final.png' if is_final else f'epoch_{epoch+1}.png'
    filepath = os.path.join(directory, filename)

    save_image(comparison, filepath)


def create_comparison(left: Tensor, right: Tensor, disparity: Tensor,
                      device: Device = 'cpu') -> Tensor:

    left_disp, right_disp = torch.split(disparity, [1, 1], 1)
    
    left_recon = reconstruct_left_image(left_disp, right)
    right_recon = reconstruct_right_image(right_disp, left)

    left_disp = to_heatmap(left_disp[0].detach(), device, inverse=True)
    right_disp = to_heatmap(right_disp[0].detach(), device, inverse=True)

    grid = torch.stack((
        left[0], right[0],
        left_disp, right_disp,
        left_recon[0], right_recon[0]))

    return make_grid(grid, nrow=2)


@torch.no_grad()
def evaluate_model(model: Module, loader: DataLoader,
                   loss_function: Module, disparity_scale: float = 1.0,
                   save_comparison_to: Optional[str] = None,
                   epoch: Optional[int] = None, is_final: bool = True,
                   device: Device = 'cpu') -> float:

    running_loss = 0

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    tepoch = tqdm.tqdm(loader, 'Evaluation', unit='batch')

    for i, image_pair in enumerate(tepoch):
        left = image_pair["left"].to(device)
        right = image_pair["right"].to(device)

        disparities = model(left, disparity_scale)
        loss = loss_function(left, right, disparities)

        running_loss += loss.item()

        average_loss_per_image = running_loss / ((i+1) * batch_size)
        tepoch.set_postfix(loss=average_loss_per_image)

        if save_comparison_to is not None and i == 0:
            # Use the full-size disparity image
            comparison = create_comparison(left, right, disparities[0], device)
            save_comparison(comparison, save_comparison_to, epoch, is_final)

    return average_loss_per_image
