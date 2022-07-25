import os.path
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from .loss import MonodepthLoss

Device = Union[torch.device, str]


def create_comparison_image(left: Tensor, right: Tensor,
                            loss_function: MonodepthLoss) -> Tensor:

    left_disp_batch, right_disp_batch = loss_function.disparities[0]
    left_recon_batch, right_recon_batch = loss_function.reconstructions[0]

    left_disp = left_disp_batch[0].detach()
    right_disp = right_disp_batch[0].detach()

    left_disp = torch.cat((left_disp, left_disp, left_disp), dim=0)
    right_disp = torch.cat((right_disp, right_disp, right_disp), dim=0)

    left_recon = left_recon_batch[0].detach()
    right_recon = right_recon_batch[0].detach()

    grid = torch.stack((left[0], left_disp, left_recon,
                       right[0], right_disp, right_recon), dim=0)

    return make_grid(grid, nrow=3)


@torch.no_grad()
def evaluate_model(model: Module, loader: DataLoader,
                   loss_function: Module, disparity_scale: float = 1.0,
                   save_comparison_to: Optional[str] = None,
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
            filepath = os.path.join(save_comparison_to, 'comparison.png')
            image = create_comparison_image(left, right, loss_function)
            save_image(image, filepath)

    return average_loss_per_image
