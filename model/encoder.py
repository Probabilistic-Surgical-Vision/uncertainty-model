from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage


class RandomEncoder(nn.Module):
    def __init__(self, nodes: Optional[int] = None,
                 seed: Optional[int] = None,
                 load_graph: Optional[str] = None) -> None:

        super().__init__()

        self.enc1 = EncoderStage(in_channels=3, out_channels=32,
                                 kernel_size=7, nodes=nodes, seed=seed,
                                 stage=1, heads=8, load_graph=load_graph)

        self.enc2 = EncoderStage(in_channels=32, out_channels=64,
                                 kernel_size=5, nodes=nodes, seed=seed,
                                 stage=2, heads=8, load_graph=load_graph)

        self.enc3 = EncoderStage(in_channels=64, out_channels=128,
                                 kernel_size=3, nodes=nodes, seed=seed,
                                 stage=3, heads=8, load_graph=load_graph)

        self.enc4 = EncoderStage(in_channels=128, out_channels=256,
                                 kernel_size=3, nodes=nodes, seed=seed,
                                 stage=4, heads=8, load_graph=load_graph)

        self.enc5 = EncoderStage(in_channels=256, out_channels=512,
                                 kernel_size=3, nodes=nodes, seed=seed,
                                 stage=5, heads=8, load_graph=load_graph)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        return x1, x2, x3, x4, x5
