import os
import os.path
from datetime import datetime
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import tqdm

from .evaluate import evaluate_model
from .utils import adjust_disparity_scale

Device = Union[torch.device, str]


def save_model(model: Module, model_directory: str,
               epoch: Optional[int] = None,
               is_final: bool = False) -> None:

    if not os.path.isdir(model_directory):
        os.makedirs(model_directory, exist_ok=True)
    
    filename = 'final.pt' if is_final else f'epoch_{epoch+1}.pt'
    filepath = os.path.join(model_directory, filename)

    torch.save(model.state_dict(), filepath)


def train_one_epoch(model: Module, loader: DataLoader, loss_function: Module,
                    model_optimiser: Optimizer, disparity_scale: float,
                    discriminator: Module, disc_optimiser: Optimizer,
                    disc_loss_function: Module,
                    epoch_number: Optional[int] = None,
                    device: Device = 'cpu') -> float:
    model.train()

    running_model_loss = 0
    batch_size = loader.batch_size \
        if loader.batch_size is not None else len(loader)
    description = f'Epoch #{epoch_number}' \
        if epoch_number is not None else 'Epoch'

    tepoch = tqdm.tqdm(loader, description, unit='batch')

    for i, image_pair in enumerate(tepoch):
        model_optimiser.zero_grad()

        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        disparities = model(left, disparity_scale)
        model_loss = loss_function(left, right, disparities,
                                   discriminator)

        model_loss.backward()
        model_optimiser.step()

        disc_loss = disc_loss_function(left, right, disparities)

        disc_loss.backward()
        disc_optimiser.step()

        running_model_loss += model_loss.item()

        average_loss_per_image = running_model_loss / ((i+1) * batch_size)
        tepoch.set_postfix(loss=average_loss_per_image)

    return average_loss_per_image


def train_model(model: Module, loader: DataLoader, loss_function: Module,
                discriminator: Module, epochs: int, learning_rate: float,
                scheduler_decay_rate: float = 0.1,
                scheduler_step_size: int = 15,
                val_loader: Optional[DataLoader] = None,
                evaluate_every: Optional[int] = None,
                save_comparison: Optional[str] = None,
                save_every: Optional[int] = None,
                save_path: Optional[str] = None,
                device: Device = 'cpu') -> Tuple[List[float], List[float]]:

    model_optimiser = Adam(model.parameters(), learning_rate)
    disc_optimiser = Adam(discriminator.parameters(), learning_rate)

    scheduler = StepLR(model_optimiser, scheduler_step_size, scheduler_decay_rate)

    training_losses = []
    validation_losses = []

    if save_path is not None:
        date = datetime.now().strftime('%Y%m%d%H%M%S')
        folder = f'model_{date}'
        model_directory = os.path.join(save_path, folder)
        comparison_directory = os.path.join(save_comparison, folder)

    for i in range(epochs):
        scale = adjust_disparity_scale(epoch=i)

        loss = train_one_epoch(model, loader, model_optimiser, loss_function,
                               scale, discriminator, disc_optimiser,
                               epoch_number=(i+1), device=device)

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

    if save_path is not None:
        save_model(model, model_directory, is_final=True)

    return training_losses, validation_losses
