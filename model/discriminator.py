from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage

KernelSize = Union[int, Tuple[int, int]]
ImagePyramid = List[Tensor]


class RandomDiscriminator(nn.Module):
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

        self.conv = EncoderStage(**config['conv'],
                                 stage=(len(self.layers)+1),
                                 nodes=nodes, seed=seed,
                                 load_graph=load_graph)

        self.linear = nn.Linear(config['linear-in-features'], 1)

    def features(self, lefts: ImagePyramid,
                 rights: ImagePyramid) -> ImagePyramid:

        pyramid_iterator = zip(lefts, rights, self.layers)
        features = []

        for i, (left, right, layer) in enumerate(pyramid_iterator):
            stereo_concat = torch.cat((left, right), dim=1)

            if i == 0:
                out = layer(stereo_concat)
            else:
                feature_concat = torch.cat((out, stereo_concat), dim=1)
                out = layer(feature_concat)

            features.append(out)

        # Final layer isn't part of the pyramid
        out = self.conv(out)
        features.append(out)

        return features

    def forward(self, lefts: ImagePyramid, rights: ImagePyramid) -> Tensor:
        out = self.features(lefts, rights)[-1]
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return torch.sigmoid(out)
