from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .layers.decoder import DecoderStage


class DepthDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.dec5 = DecoderStage(in_channels=512, feature_in_channels=256,
                                 skip_in_channels=512, upsample_channels=128,
                                 out_channels=256, skip_out_channels=512,
                                 concat_disp=False, calculate_disp=False)

        self.dec4 = DecoderStage(in_channels=256, feature_in_channels=128,
                                 skip_in_channels=512, upsample_channels=64,
                                 out_channels=256, skip_out_channels=256,
                                 concat_disp=False)

        self.dec3 = DecoderStage(in_channels=256, feature_in_channels=64,
                                 skip_in_channels=256, upsample_channels=64,
                                 out_channels=128, skip_out_channels=128)

        self.dec2 = DecoderStage(in_channels=128, feature_in_channels=32,
                                 out_channels=64, upsample_channels=16,
                                 skip_in_channels=128, skip_out_channels=64)

        self.dec1 = DecoderStage(in_channels=64, feature_in_channels=3,
                                 skip_in_channels=64, upsample_channels=8,
                                 out_channels=32, skip_out_channels=32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, left_image: Tensor, *feature_maps: Tensor,
                scale: float = 2.0) -> Tuple[Tensor, ...]:

        f1, f2, f3, f4, x4 = feature_maps

        out5, skip5, _ = self.dec5.forward(x4, f4, x4, scale=scale)

        out4, skip4, disp4 = self.dec4(out5, f3, skip5, scale=scale)
        out3, skip3, disp3 = self.dec3(out4, f2, skip4, disp4, scale)
        out2, skip2, disp2 = self.dec2(out3, f1, skip3, disp3, scale)

        _, _, disp1 = self.dec1(out2, left_image, skip2, disp2, scale)

        return disp1, disp2, disp3, disp4
