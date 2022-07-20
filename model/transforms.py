import torch

from torch import Tensor
from torchvision import transforms
from numpy import random, ndarray, float64
from typing import Tuple, Union

TensorTuple = Tuple[Tensor, Tensor]
BoundsTuple = Tuple[Union[int, float], Union[int, float]]


class ResizeImage:
    def __init__(self, size: Tuple[int, int] = (256, 512)):
        self.transform = transforms.Resize(size)

    def __call__(self, images: TensorTuple) -> TensorTuple:
        left_image, right_image = images

        left_image = self.transform(left_image)
        right_image = self.transform(right_image)

        return left_image, right_image

class ToTensor:
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, images: TensorTuple) -> TensorTuple:
        left_image, right_image = images

        left_image = self.transform(left_image)
        right_image = self.transform(right_image)

        return left_image, right_image

class RandomFlip:
    def __init__(self, p: float = 0.5):
        self.probability = p
        self.transform = transforms.RandomHorizontalFlip(1)

    def __call__(self, images: TensorTuple) -> TensorTuple:
        left_image, right_image = images

        if random.random() < self.probability:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image

class RandomAugment:
    def __init__(self, p: float, gamma: BoundsTuple,
                 brightness: BoundsTuple, colour: BoundsTuple):
        
        self.probability = p

        self.gamma = gamma
        self.brightness = brightness
        self.colour = colour

    def shift_gamma(self, x: Tensor, gamma: float64) -> Tensor:
        return x ** gamma

    def shift_brightness(self, x: Tensor, brightness: float64) -> Tensor:
        return x * brightness

    def shift_colour(self, x: Tensor, colour: ndarray) -> Tensor:
        return torch.tensordot(x, torch.tensor(colour), dims=0)

    def transform(self, x: Tensor, gamma: float64, brightness:
                  float64, colour: ndarray) -> Tensor:

        x = self.shift_gamma(x, gamma)
        x = self.shift_brightness(x, brightness)
        x = self.shift_colour(x, colour)

        return torch.clamp(x, 0, 1)

    def __call__(self, images: TensorTuple) -> TensorTuple:
        left_image, right_image = images

        if random.random() < self.probability:
            gamma = random.uniform(*self.gamma)
            brightness = random.uniform(*self.brightness)
            colour = random.uniform(*self.colour, 3)

            left_image = self.transform(left_image, gamma, brightness, colour)
            right_image = self.transform(right_image, gamma, brightness, colour)

        return left_image, right_image