import torch.nn as nn
from torch import Tensor

from .layers.decoder import DecoderStage


class MonoDepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

        self.dec5 = DecoderStage(x1_in_channels=512, x2_in_channels=256,
                                 out_channels=256, upsample_channels=128,
                                 skip_in_channels=512, skip_out_channels=512,
                                 concat_disp=False, calculate_disp=False)
        
        self.dec4 = DecoderStage(x1_in_channels=256, x2_in_channels=128,
                                 out_channels=256, upsample_channels=64,
                                 skip_in_channels=512, skip_out_channels=256,
                                 concat_disp=False)

        self.dec3 = DecoderStage(x1_in_channels=256, x2_in_channels=64,
                                 out_channels=128, upsample_channels=64,
                                 skip_in_channels=256, skip_out_channels=128) 

        self.dec2 = DecoderStage(x1_in_channels=128, x2_in_channels=32,
                                 out_channels=64, upsample_channels=16,
                                 skip_in_channels=128, skip_out_channels=64)
        
        self.dec1 = DecoderStage(x1_in_channels=64, x2_in_channels=3,
                                 out_channels=32, upsample_channels=8,
                                 skip_in_channels=64, skip_out_channels=32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, left_image: Tensor, *feature_maps, disp_scale: float = 2.0):
        x4, f4, f3, f2, f1 = feature_maps

        out5, skip5, _ = self.dec5(x4, f4, x4, disp_scale)

        out4, skip4, disp4 = self.dec4(out5, f3, skip5, disp_scale)
        out3, skip3, disp3 = self.dec3(out4, f2, skip4, disp_scale)
        out2, skip2, disp2 = self.dec2(out3, f1, skip3, disp_scale)
        
        _, _, disp1 = self.dec1(out2, left_image, skip2, disp_scale)

        return disp1, disp2, disp3, disp4