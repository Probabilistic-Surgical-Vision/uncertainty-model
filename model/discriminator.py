from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage

KernelSize = Union[int, Tuple[int, int]]
ImagePyramid = List[Tensor]


class RandomDiscriminator(nn.Module):
    def __init__(self, nodes: Optional[int] = None,
                 seed: Optional[int] = None,
                 load_graph: Optional[str] = None) -> None:

        super().__init__()

        self.enc1 = EncoderStage(in_channels=6, out_channels=32,
                                 kernel_size=7, nodes=nodes, seed=seed,
                                 stage=1, heads=8, load_graph=load_graph)

        self.enc2 = EncoderStage(in_channels=38, out_channels=64,
                                 kernel_size=5, nodes=nodes, seed=seed,
                                 stage=2, heads=8, load_graph=load_graph)

        self.enc3 = EncoderStage(in_channels=70, out_channels=128,
                                 kernel_size=3, nodes=nodes, seed=seed,
                                 stage=3, heads=8, load_graph=load_graph)

        self.enc4 = EncoderStage(in_channels=134, out_channels=256,
                                 kernel_size=3, nodes=nodes, seed=seed,
                                 stage=4, heads=8, load_graph=load_graph)

        self.enc5 = EncoderStage(in_channels=256, out_channels=256,
                                 kernel_size=3, nodes=nodes, seed=seed,
                                 stage=5, heads=8, load_graph=load_graph)

        self.layers = nn.ModuleList([
            self.enc1, self.enc2,
            self.enc3, self.enc4
        ])

        self.linear = nn.Linear(in_features=32768, out_features=1)

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
        out = self.enc5(out)
        features.append(out)

        return features

    def forward(self, lefts: ImagePyramid, rights: ImagePyramid) -> Tensor:
        out = self.features(lefts, rights)[-1]
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return torch.sigmoid(out)
