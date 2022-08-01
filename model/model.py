from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .decoder import DepthDecoder
from .encoder import RandomEncoder


class RandomlyConnectedModel(nn.Module):

    def __init__(self, config: dict) -> None:

        super().__init__()

        self.encoder = RandomEncoder(config['encoder'])
        self.decoder = DepthDecoder(config['decoder'])

    def forward(self, image: Tensor, scale: float) -> Tuple[Tensor, ...]:
        encodings = self.encoder(image)
        return self.decoder(image, *encodings, scale=scale)
