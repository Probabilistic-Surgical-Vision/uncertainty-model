import tqdm
import torch
import os.path

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from typing import Optional, Union

from .loss import MonodepthLoss

def create_comparison_image(left: Tensor, right: Tensor,
                            loss_function: MonodepthLoss) -> Tensor:

    left_disp_batch, right_disp_batch = loss_function.disparities[0]
    left_recon_batch, right_recon_batch = loss_function.reconstructions[0]

    left_disp = left_disp_batch[0].detach()
    right_disp = right_disp_batch[0].detach()

    left_recon = left_recon_batch[0].detach()
    right_recon = right_recon_batch[0].detach()

    grid = torch.cat((left, left_disp, left_recon,
                      right, right_disp, right_recon), dim=0)
    
    return make_grid(grid)

@torch.no_grad()
def evaluate_model(model: Module, loader: DataLoader,
                   loss_function: Module, disparity_scale: float = 1.0,
                   save_comparison_to: Optional[str] = None,
                   device: Union[torch.device, str] = "cpu") -> float:

    running_loss = 0

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
            else len(loader)

    tepoch = tqdm.tqdm(loader, "Evaluation", unit="batch")

    for i, (left, right) in enumerate(tepoch):
        left, right = left.to(device), right.to(device)
        disparities = model(left, disparity_scale)
        
        loss = loss_function(left, right, disparities)

        running_loss += loss.item()

        average_loss_per_image = running_loss / (i * batch_size)
        tepoch.set_postfix(loss=average_loss_per_image)

        if save_comparison_to is not None and i == 0:
            filepath = os.path.join(save_comparison_to, "comparison.png")
            image = create_comparison_image(left, right, loss_function)
            save_image(image, filepath)

    return average_loss_per_image