import os.path
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt

import tqdm

from .loss import MonodepthLoss

Device = Union[torch.device, str]


def to_heatmap(x: Tensor, device: Device = 'cpu', inverse: bool = False,
               colour_map: str = 'inferno') -> Tensor:
    
    image = x.squeeze(0).numpy()
    image = 1 - image if inverse else image

    transform = plt.get_cmap(colour_map)
    heatmap = transform(image)[:,:,:3] # remove alpha channel

    return torch.from_numpy(heatmap).to(device).permute(2, 0, 1)


def save_comparison(comparison: Tensor, directory: str,
                    epoch: Optional[int] = None,
                    is_final: bool = True) -> None:

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    
    filename = 'final.png' if is_final else f'epoch_{epoch+1}.png'
    filepath = os.path.join(directory, filename)

    save_image(comparison, filepath)


def create_comparison(left: Tensor, right: Tensor,
                      loss_function: MonodepthLoss,
                      device: Device = 'cpu') -> Tensor:

    left_disp_batch, right_disp_batch = loss_function.disparities[0]
    left_recon_batch, right_recon_batch = loss_function.reconstructions[0]

    left_disp = to_heatmap(left_disp_batch[0].detach(), device, inverse=True)
    right_disp = to_heatmap(right_disp_batch[0].detach(), device, inverse=True)

    left_recon = left_recon_batch[0].detach()
    right_recon = right_recon_batch[0].detach()

    grid = torch.stack((
        left[0], right[0],
        left_disp, right_disp,
        left_recon, right_recon))

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
            comparison = create_comparison(left, right, loss_function, device)
            save_comparison(comparison, save_comparison_to, epoch, is_final)

    return average_loss_per_image
