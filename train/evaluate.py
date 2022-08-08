import os.path
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import save_image

import tqdm

from . import utils as u
from .utils import Device


def save_comparison(image: Tensor, disparity: Tensor, recon: Tensor,
                    directory: str, epoch_number: Optional[int] = None,
                    is_final: bool = True, device: Device = 'cpu') -> None:
    
    disparity_image = u.get_comparison(image, disparity, recon,
                                       add_scaled=True, device=device)

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    filename = 'final.png' if is_final else f'epoch_{epoch_number:03}.png'
    filepath = os.path.join(directory, filename)

    print(f'Saving comparison to:\n\t{filepath}')
    save_image(disparity_image, filepath)


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
            save_comparison(image_pyramid[0][0], disparities[0][0],
                            recon_pyramid[0][0], save_evaluation_to,
                            epoch_number, is_final, device)

    if no_pbar and rank == 0:
        disc_loss_string = f'{disc_loss_per_image:.2e}' \
            if disc_loss_per_image is not None else None

        print(f'{description}:'
              f'\n\tmodel loss: {model_loss_per_image:.2e}'
              f'\n\tdiscriminator loss: {disc_loss_string}'
              f'\n\tdisparity scale: {scale:.2f}')

    return model_loss_per_image, disc_loss_per_image
