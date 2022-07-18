import torch.nn as nn

from torch import Tensor
from typing import Optional

from .encoder import RandomEncoder
from .decoder import MonoDepthDecoder


class MonoRandNNmodel(nn.Module):
    def __init__(self, nodes: Optional[int] = None,
                 load_graph: Optional[str] = None):

        super().__init__()

        self.encoder = RandomEncoder(nodes, load_graph)
        self.decoder = MonoDepthDecoder()

    def forward(self, left_image: Tensor, disparity_scale: float):
        encodings = self.encoder(left_image) 
        return self.decoder(left_image, encodings, disparity_scale)