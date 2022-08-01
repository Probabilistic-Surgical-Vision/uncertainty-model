from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage


class RandomEncoder(nn.Module):
    def __init__(self, config: dict) -> None:

        super().__init__()

        nodes = config['nodes']
        seed = config['seed']
        load_graph = config['load_graph']

        self.layers = nn.ModuleList()

        for i, layer_config in enumerate(config['layers']):
            self.layers.append(EncoderStage(**layer_config, stage=(i+1),
                                            nodes=nodes, seed=seed,
                                            load_graph=load_graph))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        encodings = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x)
            else:
                out = layer(out)

            encodings.append(out)

        return tuple(encodings)
