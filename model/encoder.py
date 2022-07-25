import os
import os.path

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential

from . import graph as g
from .attention import EfficientAttention
from .layers.encoder import EncoderStage

KernelSize = Union[int, Tuple[int, int]]


class RandomEncoder(nn.Module):
    def __init__(self, nodes: Optional[int] = None,
                 seed: Optional[int] = None,
                 load_graph: Optional[str] = None) -> None:

        super().__init__()

        self.enc1 = self.build_encoder_stage(in_channels=3, out_channels=32,
                                             kernel_size=7, nodes=nodes,
                                             seed=seed, stage=1, heads=8,
                                             load_graph=load_graph)

        self.enc2 = self.build_encoder_stage(in_channels=32, out_channels=64,
                                             kernel_size=5, nodes=nodes,
                                             seed=seed, stage=2, heads=8,
                                             load_graph=load_graph)

        self.enc3 = self.build_encoder_stage(in_channels=64, out_channels=128,
                                             kernel_size=3, nodes=nodes,
                                             seed=seed, stage=3, heads=8,
                                             load_graph=load_graph)

        self.enc4 = self.build_encoder_stage(in_channels=128, out_channels=256,
                                             kernel_size=3, nodes=nodes,
                                             seed=seed, stage=4, heads=8,
                                             load_graph=load_graph)

        self.enc5 = self.build_encoder_stage(in_channels=256, out_channels=512,
                                             kernel_size=3, nodes=nodes,
                                             seed=seed, stage=5, heads=8,
                                             load_graph=load_graph)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def build_encoder_stage(self, in_channels: int, out_channels: int,
                            kernel_size: KernelSize, stage: int, heads: int,
                            nodes: int = 5, p: float = 0.75, k: int = 4,
                            seed: Optional[int] = None,
                            load_graph: Optional[str] = None,
                            save_graph: Optional[str] = None) -> Sequential:

        if load_graph is not None:
            filename = f"stage_{stage}.gpickle"
            filepath = os.path.join(load_graph, filename)
            graph = g.load_graph(filepath)
        else:
            graph = g.build_graph(nodes, k, p, seed=(stage*seed))

            if save_graph is not None:
                directory = f'nodes_{nodes}_seed_{seed}'
                directory_path = os.path.join(save_graph, directory)

                if not os.path.isdir(directory_path):
                    os.makedirs(directory_path, exist_ok=True)

                filename = f'stage_{stage}.gpickle'
                filepath = os.path.join(directory_path, filename)

                g.save_graph(graph, filepath)

        return nn.Sequential(
            EncoderStage(graph, in_channels, out_channels, kernel_size),
            EfficientAttention(out_channels, out_channels, out_channels, heads)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        return x1, x2, x3, x4, x5
