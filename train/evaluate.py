import os.path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import save_image

import tqdm

from .loss import WeightedSSIMLoss

from . import utils as u
from .utils import Device


def save_comparison(image: Tensor, disparity: Tensor, recon: Tensor,
                    directory: str, epoch_number: Optional[int] = None,
                    is_final: bool = True, device: Device = 'cpu') -> None:
    """Save a .png comparing the original image, disparity and recon images.

    Args:
        image (Tensor): The original image pair.
        disparity (Tensor): The disparity image pair.
        recon (Tensor): The reconstructed image pair.
        directory (str): The directory to save the result to.
        epoch_number (Optional[int], optional): The epoch number for creating
            a filename. Defaults to None.
        is_final (bool, optional): Forces the filename to be `final.png`.
            Defaults to True.
        device (Device, optional): The torch device to use. Defaults to 'cpu'.
    """
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
                   no_pbar: bool = False,
                   rank: int = 0) -> Tuple[float, float]:
    """Loop through the validation set and report model losses.

    Args:
        model (Module): The model to test.
        loader (DataLoader): The validation loader to iterate through.
        loss_function (Module): The loss function to report.
        scale (float, optional): A multiplier to scale the model's predictions
            by. Defaults to 1.0.
        disc (Optional[Module], optional): The discriminator to test.
            Defaults to None.
        disc_loss_function (Optional[Module], optional): The discriminator
            loss functions to report. Defaults to None.
        save_evaluation_to (Optional[str], optional): Path to the directory to
            save comparison images to. Defaults to None.
        epoch_number (Optional[int], optional): The epoch number in training.
            Defaults to None.
        is_final (bool, optional): The evaluation is taking place
            post-training. Defaults to True.
        scales (int, optional): The size of the image pyramid. Defaults to 4.
        device (Device, optional): The torch device to use. Defaults to 'cpu'.
        no_pbar (bool, optional): Disable `tqdm` progress bar. Defaults to
            False.
        rank (int, optional): The rank of the process (for multiprocessing
            only). Defaults to 0.

    Returns:
        float: The average model loss per image.
        float: The average discriminator loss per image.
    """
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

    model.eval()
    val_loss_function = WeightedSSIMLoss()

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        images = torch.cat([left, right], dim=1)
        disparity = model(left, scale)
        left_disp, right_disp = torch.split(disparity, [1, 1], dim=1)
        left_recon = u.reconstruct_left_image(left_disp, right)
        right_recon = u.reconstruct_right_image(right_disp, left)

        recon = torch.cat((left_recon, right_recon), dim=1)

        model_loss = val_loss_function.forward(images, recon)

        """
        if disc is not None:
            disc_loss = u.run_discriminator(image_pyramid, recon_pyramid,
                                            disc, disc_loss_function,
                                            batch_size)
        """

        if rank > 0:
            continue

        running_model_loss += model_loss.item()
        model_loss_per_image = running_model_loss / ((i+1) * batch_size)

        """
        if disc is not None:
            running_disc_loss += disc_loss.item()
            disc_loss_per_image = running_disc_loss / ((i+1) * batch_size)
        """

        tepoch.set_postfix(loss=model_loss_per_image,
                           disc=disc_loss_per_image,
                           scale=scale)

        if save_evaluation_to is not None and i == 0:
            save_comparison(images[0], disparity[0], recon[0],
                            save_evaluation_to, epoch_number,
                            is_final, device)

    if no_pbar and rank == 0:
        disc_loss_string = f'{disc_loss_per_image:.2e}' \
            if disc_loss_per_image is not None else None

        print(f'{description}:'
              f'\n\tmodel loss: {model_loss_per_image:.2e}'
              f'\n\tdiscriminator loss: {disc_loss_string}'
              f'\n\tdisparity scale: {scale:.2f}')

    return model_loss_per_image, disc_loss_per_image
