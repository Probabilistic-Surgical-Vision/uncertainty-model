import torch.nn as nn

from torch import Tensor
from torch.nn import Sequential
from typing import Optional, Tuple, Union

from . import graph as g

from .attention import EfficientAttention
from .layers.encoder import EncoderStage


class RandomEncoder(nn.Module):
    def __init__(self, nodes: Optional[int] = None,
                 load_graph: Optional[str] = None):

        super().__init__()

        self.enc1 = self.build_encoder_stage(in_channels=3, out_channels=32,
                                             kernel_size=7, nodes=nodes,
                                             stage=1, load_graph=load_graph)

        self.enc2 = self.build_encoder_stage(in_channels=32, out_channels=64,
                                             kernel_size=5, nodes=nodes,
                                             stage=2, load_graph=load_graph)

        self.enc3 = self.build_encoder_stage(in_channels=64, out_channels=128,
                                             kernel_size=3, nodes=nodes, stage=3,
                                             load_graph=load_graph)

        self.enc4 = self.build_encoder_stage(in_channels=128, out_channels=256,
                                             kernel_size=3, nodes=nodes,
                                             stage=4, load_graph=load_graph)

        self.enc5 = self.build_encoder_stage(in_channels=256, out_channels=512,
                                             kernel_size=3, nodes=nodes,
                                             stage=5, load_graph=load_graph)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def build_encoder_stage(self, in_channels: int, out_channels: int,
                            kernel_size: Union[int, Tuple[int, int]],
                            stage: int, heads: int, nodes: int = 5,
                            p: float = 0.75, k: int = 0.5,
                            seed: Optional[int] = None,
                            load_graph: Optional[str] = None,
                            save_graph: Optional[str] = None) -> Sequential:

        if load_graph is not None:
            graph = g.load_graph(load_graph)
        else:
            graph = g.build_graph(nodes, k, p, seed=(stage*seed))
            
            if save_graph is not None:
                g.save_graph(graph, save_graph)

        return nn.Sequential(
            EncoderStage(graph, in_channels, out_channels, kernel_size),
            EfficientAttention(out_channels, out_channels, heads, out_channels)
        )
  
    def forward(self, x: Tensor):

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        return x1, x2, x3, x4, x5