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
from .loss import MonodepthLoss
from .utils import adjust_disparity_scale

Device = Union[torch.device, str]


def train_one_epoch(model: Module, loader: DataLoader, optimiser: Optimizer,
                    loss_function: Module, disparity_scale: float,
                    epoch_number: Optional[int] = None,
                    device: Device = 'cpu') -> float:
    model.train()

    running_loss = 0
    batch_size = loader.batch_size \
        if loader.batch_size is not None else len(loader)
    description = f'Epoch #{epoch_number}' \
        if epoch_number is not None else 'Epoch'

    tepoch = tqdm.tqdm(loader, description, unit='batch')

    for i, image_pair in enumerate(tepoch):
        optimiser.zero_grad()

        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        disparities = model(left, disparity_scale)

        loss = loss_function(left, right, disparities)

        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        average_loss_per_image = running_loss / ((i+1) * batch_size)
        tepoch.set_postfix(loss=average_loss_per_image)

    return average_loss_per_image


def train_model(model: Module, loader: DataLoader, epochs: int,
                learning_rate: float, scheduler_step_size: int = 15,
                scheduler_decay_rate: float = 0.1,
                val_loader: Optional[DataLoader] = None,
                evaluate_every: Optional[int] = None,
                save_every: Optional[int] = None,
                save_path: Optional[str] = None,
                device: Device = 'cpu') -> Tuple[List[float], List[float]]:

    optimiser = Adam(model.parameters(), learning_rate)
    scheduler = StepLR(optimiser, scheduler_step_size, scheduler_decay_rate)

    loss_function = MonodepthLoss()

    training_losses = []
    validation_losses = []

    if save_path is not None:
        date = datetime.now().strftime('%Y%m%d%H%M%S')
        directory = os.path.join(save_path, f'model_{date}')

        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

    for i in range(epochs):
        scale = adjust_disparity_scale(epoch=i)

        loss = train_one_epoch(model, loader, optimiser, loss_function,
                               scale, epoch_number=(i+1), device=device)

        training_losses.append(loss)
        scheduler.step()

        if evaluate_every is not None and i % evaluate_every == 0:
            loss = evaluate_model(model, val_loader, loss_function,
                                  scale, device=device)

            validation_losses.append(loss)

        if save_every is not None and i % save_every == 0:
            filepath = os.path.join(directory, f'epoch_{i+1}.pt')
            torch.save(model.state_dict(), filepath)

    print('Training completed.')

    if save_path is not None:
        filepath = os.path.join(directory, 'final.pt')
        torch.save(model.state_dict(), filepath)

    return training_losses, validation_losses
