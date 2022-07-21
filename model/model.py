from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from .decoder import DepthDecoder
from .encoder import RandomEncoder


class RandomlyConnectedModel(nn.Module):

    def __init__(self, nodes: Optional[int] = None,
                 load_graph: Optional[str] = None) -> None:

        super().__init__()

        self.encoder = RandomEncoder(nodes, load_graph)
        self.decoder = DepthDecoder()

    def forward(self, image: Tensor, scale: float) -> Tuple[Tensor, ...]:
        encodings = self.encoder(image)
        return self.decoder(image, encodings, scale)
