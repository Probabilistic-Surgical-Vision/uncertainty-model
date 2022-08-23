import os.path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from torchmetrics.functional import \
    structural_similarity_index_measure as ssim

import tqdm

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
                   save_evaluation_to: Optional[str] = None,
                   epoch_number: Optional[int] = None,
                   scale: int = 4, is_final: bool = True,
                   kernel_size: int = 11,
                   no_pbar: bool = False,
                   device: Device = 'cpu',
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
    running_left_ssim = 0
    running_right_ssim = 0

    average_left_ssim = None
    average_right_ssim = None

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch',
                       disable=(no_pbar or rank > 0))

    model.eval()

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        images = torch.cat([left, right], dim=1)
        disparity = model(left, scale)
        left_disp, right_disp = torch.split(disparity, [1, 1], dim=1)
        left_recon = u.reconstruct_left_image(left_disp, right)
        right_recon = u.reconstruct_right_image(right_disp, left)

        left_ssim = ssim(left_recon, left, kernel_size=kernel_size)
        right_ssim = ssim(right_recon, right, kernel_size=kernel_size)

        if rank > 0:
            continue

        running_left_ssim += left_ssim.item()
        average_left_ssim = running_left_ssim / ((i+1) * batch_size)

        running_right_ssim += right_ssim.item()
        average_right_ssim = running_right_ssim / ((i+1) * batch_size)

        tepoch.set_postfix(left=average_left_ssim,
                           right=average_right_ssim,
                           scale=scale)

        if save_evaluation_to is not None and i == 0:
            recon = torch.cat((left_recon, right_recon), dim=1)

            save_comparison(images[0], disparity[0], recon[0],
                            save_evaluation_to, epoch_number,
                            is_final, device)

    if no_pbar and rank == 0:
        print(f'{description}:'
              f'\n\tleft ssim: {average_left_ssim:.2f}'
              f'\n\tright ssim: {average_right_ssim:.2f}'
              f'\n\tdisparity scale: {scale:.2f}')

    return average_left_ssim, average_right_ssim
