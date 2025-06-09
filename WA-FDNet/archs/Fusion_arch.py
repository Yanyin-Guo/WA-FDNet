import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from ultralytics.nn.modules.block import Conv, C3k2, A2C2f, C3k
from ultralytics.nn.modules import Concat, Detect
from basicsr.utils.registry import ARCH_REGISTRY
from scripts.nn import *

@ARCH_REGISTRY.register()    
class Shared_Encoder_Fusion(nn.Module):
    def __init__(self, in_channels=16, vis=False):
        super().__init__()
        self.vis = vis
        block = ResBlock
        block1 = ResBlock_con

        self.inc_vi = self._make_layer(block, 3, in_channels * 2) #32, 256
        self.down_encoder1_vi = self._make_layer(block1, in_channels * 2, in_channels * 2, 1)  # 32,128
        self.down_encoder2_vi = self._make_layer(block1, in_channels * 2, in_channels * 4, 1)  # 64,64
        self.backbone_1_1_1 = torch.nn.Sequential(
            C3k2(64, 128, 1, False, 0.25),
            Conv(128, 128, 3, 2), 
        )

        self.inc_ir = self._make_layer(block, 1, in_channels * 2) #32, 256
        self.down_encoder1_ir = self._make_layer(block1, in_channels * 2, in_channels * 2, 1)  # 32,128
        self.down_encoder2_ir = self._make_layer(block1, in_channels * 2, in_channels * 4, 1)  # 64,64
        self.backbone_1_2_1 = torch.nn.Sequential(
            C3k2(64, 128, 1, False, 0.25),
            Conv(128, 128, 3, 2), 
        )

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_vi = x['vi']
        x_ir = x['ir']
        x1_vi = self.inc_vi(x_vi)  # 32 256
        x2_vi = self.down_encoder1_vi(x1_vi)  # 32 128
        x3_vi = self.down_encoder2_vi(x2_vi)  # 64 64
        x4_vi_f = self.backbone_1_1_1(x3_vi)  # 128 32

        x1_ir = self.inc_ir(x_ir)  # 32 256
        x2_ir = self.down_encoder1_ir(x1_ir)  # 32 128
        x3_ir = self.down_encoder2_ir(x2_ir)  # 64 64
        x4_ir_f = self.backbone_1_2_1(x3_ir)  # 128 32

        return x1_vi, x1_ir, x2_vi, x2_ir, x3_vi, x3_ir, x4_vi_f, x4_ir_f

@ARCH_REGISTRY.register()
class Fusion_Decoder_Fusion(nn.Module):
    def __init__(self, in_channels=16, out_channel=1, vis=False):
        super().__init__()
        self.vis = vis
        self.in_channels = in_channels
        self.out_channel = out_channel

        self.enhance_1 = enFusion_block(in_channels*2)
        self.enhance_2 = enFusion_block(in_channels*2)
        self.enhance_3 = enFusion_block(in_channels*4)
        self.enhance_4 = enFusion_block(in_channels*8)
        self.neck = ResBlock(in_channels*8, in_channels*4)
        
        self.up_decoder3 = UpBlock(in_channels * 8, in_channels * 2, nb_conv=2)
        self.up_decoder2 = UpBlock_SCA(in_channels * 4, in_channels * 2, nb_conv=2)
        self.up_decoder1 = UpBlock_SCA(in_channels * 4, in_channels, nb_conv=2)
        # self.outc = Res_block_SA(in_channels, out_channel)
        self.outc = nn.Conv2d(in_channels, out_channel, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x1_vi, x1_ir, x2_vi, x2_ir, x3_vi, x3_ir, x4_vi, x4_ir):
        x1 = self.enhance_1(x1_vi, x1_ir)
        x2 = self.enhance_2(x2_vi, x2_ir)
        x3 = self.enhance_3(x3_vi, x3_ir)
        d4 = self.enhance_4(x4_vi, x4_ir)

        d3 = self.up_decoder3(self.neck(d4), x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))

        return out