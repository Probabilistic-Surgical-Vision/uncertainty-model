import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Optional, Union


class ConvLayer(nn.Module):
    """Layer to pad and convolve inputs
    This is used as disp_conv too, followed by sigmoid layer
    """
    def __init__(self, in_channels: int, out_channels: int,
                 reflection: bool = True, sigmoid: bool = False,
                 kernel_size: Union[int, Tuple[int, int]] = 3):

        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1) if reflection \
                else nn.ZeroPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.Sigmoid() if sigmoid \
                else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class ConvELUBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 batch_norm: bool = False):

        super().__init__()
        
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels) if batch_norm \
                else nn.Identity(),
            nn.ELU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class SELayer(nn.Module):
    # from: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def __init__(self, channels: int, reduction: int = 16, fc: bool = True):
        super().__init__()

        self.fc = fc
        self.channels_reduced = channels // reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excite = nn.Sequential(
            nn.Linear(channels, self.channels_reduced, bias=False) if fc \
                else nn.Conv2d(channels, self.channels_reduced,
                               kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels_reduced, channels, bias=False) if fc \
                else nn.Conv2d(self.channels_reduced, channels,
                               kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        x_squeezed = self.squeeze(x)

        if self.fc:
            b, c, _, _ = x.size()
            x_squeezed = x_squeezed.view(b, c)
            x_excited = self.excite(x_squeezed) \
                .view(b, c, 1, 1) \
                    .expand_as(x)
        else:
            x_excited = self.excite(x_squeezed)
        
        return x * x_excited

class DecoderStage(nn.Module):

    DecoderOut = Tuple[Tensor, Tensor, Optional[Tensor]]

    def __init__(self, x1_in_channels: int, x2_in_channels: int,
                 out_channels: int, upsample_channels: int,
                 skip_in_channels: int, skip_out_channels: int,
                 disp_in_channels: int = 2, disp_out_channels: int = 2,
                 batch_norm: bool = True, fc: bool = True, scale: float = 2.0,
                 concat_disp: bool = True, calculate_disp: bool = True):
        """
        input is x1, x2, prev_disp, prev_skip
        
        x1_upsampled <- upsample(x1) { upsample_channels <- x1_channels // divisor }
        skip <- se(prev_skip + x1_upsampled) { skip_channels <- prev_skip_channels + x2_channels }
        x_concat <- iconv(x1_upsampled + skip + prev_disp? ) { out_channels <- upsample_channels + skip_channels + disp_channels }
        disp <- disp_conv(x_concat) { disp_channels <- out_channels }
        """
        self.scale = scale
        
        self.calculate_disp = calculate_disp
        self.concat_disp = concat_disp
        
        self.upsample = nn.Sequential(
            ConvELUBlock(x1_in_channels, upsample_channels * int(scale ** 2),
                         batch_norm=batch_norm),
            nn.PixelShuffle(upscale_factor=self.scale)
        )
        
        self.squeeze_excite = nn.Sequential(
            ConvELUBlock(x2_in_channels + skip_in_channels, skip_out_channels,
                         kernel_size=1, batch_norm=True),
            SELayer(channels=skip_out_channels, fc=fc)
        )

        iconv_in_channels = upsample_channels + skip_out_channels
        iconv_in_channels += disp_in_channels if concat_disp else 0

        self.iconv = ConvELUBlock(iconv_in_channels, out_channels,
                                  batch_norm=batch_norm)

        self.disp_conv = ConvLayer(out_channels, disp_out_channels,
                                   sigmoid=True)

    def forward(self, x1: Tensor, x2: Tensor, prev_skip: Tensor,
                prev_disp: Optional[Tensor] = None,
                disp_scale: Optional[float] = 1.0) -> DecoderOut:

        prev_skip = F.interpolate(prev_skip, scale_factor=self.scale,
                                  align_corners=True, mode='bilinear')
        prev_disp = F.interpolate(prev_disp, scale_factor=self.scale,
                                  align_corners=True, mode='bilinear')

        x1_upsampled = self.upsample(x1)
        skip = self.squeeze_excite(torch.cat(x2, prev_skip))
        x_concat = torch.cat((x1_upsampled, skip), 1)

        if self.concat_disp:
            x_concat = torch.cat((x_concat, prev_disp), 1)

        out = self.iconv(x_concat)

        disparity = torch.float(disp_scale) * self.disp_conv(out) \
            if self.calculate_disp else None

        return out, skip, disparity