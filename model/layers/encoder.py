import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, Size
from typing import Tuple, Union
from networkx import Graph

from ..graph import Node, get_graph_info


class ConvELUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]]):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

        padding_size = (kernel_size - 1) // 2
        self.padding = tuple([padding_size] * 4)
        
    def forward(self, x: Tensor) -> Tensor:
        x_padded = F.pad(x, self.padding)
        return self.layers(x_padded)

class NodeBlock(nn.Module):
    def __init__(self, node: Node, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]]):

        super().__init__()

        self.numberof_inputs = len(node.inputs)
        
        if self.numberof_inputs > 1:
            initial_means = torch.ones(self.numberof_inputs)
            self.mean_weight = nn.Parameter(initial_means)

        if node.type == 0: # If node is an input node
            triplet_in_channels = in_channels
            triplet_stride = 2
        else:
            triplet_in_channels = out_channels
            triplet_stride = 1

        self.convolution = ConvELUBlock(triplet_in_channels, out_channels,
                                        kernel_size, stride=triplet_stride)
    
    def resize_input(self, input: Tensor, desired_size: Size) -> Tensor:
        _, _, input_h, input_w = input.size()
        _, _, desired_h, desired_w = desired_size

        dw = desired_w - input_w
        dh = desired_h - input_h

        pad_size = [
            dw // 2, dw - dw // 2,
            dh // 2, dh - dh // 2
        ]

        return F.pad(input, pad_size, mode="reflect")

    def forward(self, *inputs) -> Tensor:
        if self.numberof_inputs == 1:
            out = inputs[0]
        else:
            out = torch.sigmoid(self.mean_weight[0]) * inputs[0]

            for i, x in enumerate(inputs[1:]):
                if x.size(2) != out.size(2):
                    x = self.resize_input(x, out.size())
                
                out += self.sigmoid(self.mean_weight[i]) * x

        return self.convolution(out)

class EncoderStage(nn.Module):
    def __init__(self, graph: Graph, in_channels: int, out_channels: int,
                kernel_size: Union[int, Tuple[int, int]]):
        super().__init__()

        self.nodes, self.in_nodes, self.out_nodes = get_graph_info(graph)
        
        self.node_blocks = nn.ModuleList()
        
        for node in self.nodes:
            block = NodeBlock(node, in_channels, out_channels, kernel_size)
            self.node_blocks.append(block)
  
    def resize_output(self, output: Tensor, desired_size: Size) -> Tensor:
        _, _, output_h, output_w = output.size()
        _, _, desired_h, desired_w = desired_size

        dw = desired_w - output_w
        dh = desired_h - output_h

        pad_size = [
            dw // 2, dw - dw // 2,
            dh // 2, dh - dh // 2
        ]

        return F.pad(input, pad_size, mode="reflect")

    def forward(self, x: Tensor) -> Tensor:
        results = { id:self.node_blocks[id](x) for id in self.in_nodes }

        for id, node in enumerate(self.nodes):
            if id in self.in_nodes:
                continue

            inputs = [ results[input_id] for input_id in node.inputs ]
            results[id] = self.node_blocks[id](*inputs)
        
        for i, id in enumerate(self.out_nodes):
            if i == 0:
                out = results[id]
                continue

            if out.size(2) != results[id].size(2):
                results[id] = self.resize_output(results[id], out.size())
        
            out += results[id]

        return out / len(self.out_nodes)