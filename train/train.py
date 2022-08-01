import os
import os.path
from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import tqdm

from .evaluate import evaluate_model
from . import utils as u
from .utils import Device, PyramidPair


def run_discriminator(discriminator: Module, disc_loss_function: Module,
                      image_pyramid: PyramidPair,
                      recon_pyramid: PyramidPair) -> Tensor:

    real_pred = discriminator(*image_pyramid)
    real_labels = torch.ones_like(real_pred)

    real_loss = disc_loss_function(real_pred, real_labels)

    fake_pred = discriminator(*recon_pyramid)
    fake_labels = torch.zeros_like(fake_pred)

    fake_loss = disc_loss_function(fake_pred, fake_labels)

    return (real_loss + fake_loss) / 2


def save_model(model: Module, model_directory: str,
               epoch: Optional[int] = None,
               is_final: bool = False) -> None:

    if not os.path.isdir(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    filename = 'final.pt' if is_final else f'epoch_{epoch+1:03}.pt'
    filepath = os.path.join(model_directory, filename)

    torch.save(model.state_dict(), filepath)


def train_one_epoch(model: Module, loader: DataLoader, loss_function: Module,
                    model_optimiser: Optimizer, scale: float,
                    disc: Optional[Module] = None,
                    disc_optimiser: Optional[Optimizer] = None,
                    disc_loss_function: Optional[Module] = None,
                    epoch_number: Optional[int] = None,
                    scales: int = 4, perceptual_update_freq: int = 10,
                    device: Device = 'cpu', no_pbar: bool = False) -> float:
    model.train()

    running_model_loss = 0
    running_disc_loss = 0

    batch_size = loader.batch_size \
        if loader.batch_size is not None else len(loader)
    description = f'Epoch #{epoch_number}' \
        if epoch_number is not None else 'Epoch'
    disc_clone = deepcopy(disc) if disc is not None else None

    tepoch = tqdm.tqdm(loader, description, unit='batch', disable=no_pbar)

    for i, image_pair in enumerate(tepoch):
        model_optimiser.zero_grad()

        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        left_pyramid = u.scale_pyramid(left, scales)
        right_pyramid = u.scale_pyramid(right, scales)

        disparities = model(left, scale)

        image_pyramid = (left_pyramid, right_pyramid)
        recon_pyramid = u.reconstruct_pyramid(disparities, left_pyramid,
                                              right_pyramid)

        model_loss = loss_function(image_pyramid, disparities,
                                   recon_pyramid, i, disc_clone)

        model_loss.backward()
        model_optimiser.step()

        running_model_loss += model_loss.item()
        model_loss_per_image = running_model_loss / ((i+1) * batch_size)

        if disc is not None:
            disc_optimiser.zero_grad()
            disc_loss = run_discriminator(disc, disc_loss_function,
                                          image_pyramid, recon_pyramid)
            disc_loss.backward()
            disc_optimiser.step()

            running_disc_loss += disc_loss.item()
            disc_loss_per_image = running_disc_loss / ((i+1) * batch_size)
        else:
            disc_loss_per_image = None
        
        if i % perceptual_update_freq == 0:
            disc_clone.load_state_dict(disc.state_dict())

        tepoch.set_postfix(model=model_loss_per_image,
                           disc=disc_loss_per_image,
                           scale=scale)

    return model_loss_per_image, disc_loss_per_image


def train_model(model: Module, loader: DataLoader, loss_function: Module,
                epochs: int, learning_rate: float,
                discriminator: Optional[Module] = None,
                disc_loss_function: Optional[Module] = None,
                scheduler_decay_rate: float = 0.1,
                scheduler_step_size: int = 15,
                perceptual_update_freq: int = 10,
                val_loader: Optional[DataLoader] = None,
                evaluate_every: Optional[int] = None,
                save_evaluation_to: Optional[str] = None,
                save_every: Optional[int] = None,
                save_model_to: Optional[str] = None,
                device: Device = 'cpu',
                no_pbar: bool = False) -> Tuple[List[float], List[float]]:

    model_optimiser = Adam(model.parameters(), learning_rate)

    disc_optimiser = Adam(discriminator.parameters(), learning_rate) \
        if discriminator is not None else None

    scheduler = StepLR(model_optimiser, scheduler_step_size,
                       scheduler_decay_rate)

    training_losses = []
    validation_losses = []

    if save_model_to is not None:
        date = datetime.now().strftime('%Y%m%d%H%M%S')
        folder = f'model_{date}'
        model_directory = os.path.join(save_model_to, folder)
        comparison_directory = os.path.join(save_evaluation_to, folder)

    for i in range(epochs):
        scale = u.adjust_disparity_scale(epoch=i)

        loss = train_one_epoch(model, loader, loss_function, model_optimiser,
                               scale, discriminator, disc_optimiser,
                               disc_loss_function, (i+1),
                               perceptual_update_freq,
                               device=device, no_pbar=no_pbar)

        training_losses.append(loss)
        scheduler.step()

        if evaluate_every is not None and (i+1) % evaluate_every == 0:
            loss = evaluate_model(model, val_loader, loss_function, scale,
                                  comparison_directory, epoch=i,
                                  device=device, is_final=False)

            validation_losses.append(loss)

        if save_every is not None and (i+1) % save_every == 0:
            save_model(model, model_directory, epoch=i)

    print('Training completed.')

    if save_model_to is not None:
        save_model(model, model_directory, is_final=True)

    return training_losses, validation_losses
