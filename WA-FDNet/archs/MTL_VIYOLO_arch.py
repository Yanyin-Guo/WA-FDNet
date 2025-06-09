import torch
from torch import nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from ultralytics.nn.modules.block import Conv, C3k2, A2C2f, C3k
from ultralytics.nn.modules import Concat, Detect
from basicsr.utils.registry import ARCH_REGISTRY
from scripts.nn import *

@ARCH_REGISTRY.register()    
class MTL_Shared_Encoder(nn.Module):
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
        self.backbone_1_1_2 = C3k2(128, 256, 1, False, 0.25) #4
        self.backbone_2_1 = torch.nn.Sequential(
            Conv(256, 256, 3, 2), #5
            A2C2f(256, 256, 2, True, 4) #6
        )
        self.backbone_3_1 = torch.nn.Sequential(
            Conv(256, 512, 3, 2), #7
            A2C2f(512, 512, 2, True, 1) #8
        )

        self.inc_ir = self._make_layer(block, 1, in_channels * 2) #32, 256
        self.down_encoder1_ir = self._make_layer(block1, in_channels * 2, in_channels * 2, 1)  # 32,128
        self.down_encoder2_ir = self._make_layer(block1, in_channels * 2, in_channels * 4, 1)  # 64,64
        self.backbone_1_2_1 = torch.nn.Sequential(
            C3k2(64, 128, 1, False, 0.25),
            Conv(128, 128, 3, 2), 
        )
        self.backbone_1_2_2 = C3k2(128, 256, 1, False, 0.25) #4
        self.backbone_2_2 = torch.nn.Sequential(
            Conv(256, 256, 3, 2), #5
            A2C2f(256, 256, 2, True, 4) #6
        )
        self.backbone_3_2 = torch.nn.Sequential(
            Conv(256, 512, 3, 2), #7
            A2C2f(512, 512, 2, True, 1) #8
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
        x4_vi = self.backbone_1_1_2(x4_vi_f)  # 256 32
        x5_vi = self.backbone_2_1(x4_vi)  # 256 16
        x6_vi = self.backbone_3_1(x5_vi)  # 512 8

        x1_ir = self.inc_ir(x_ir)  # 32 256
        x2_ir = self.down_encoder1_ir(x1_ir)  # 32 128
        x3_ir = self.down_encoder2_ir(x2_ir)  # 64 64
        x4_ir_f = self.backbone_1_2_1(x3_ir)  # 128 32
        x4_ir = self.backbone_1_2_2(x4_ir_f)  # 256 32
        x5_ir = self.backbone_2_2(x4_ir)  # 256 16
        x6_ir = self.backbone_3_2(x5_ir)  # 512 8

        return x1_vi, x1_ir, x2_vi, x2_ir, x3_vi, x3_ir, x4_vi_f, x4_ir_f, x4_vi, x4_ir, x5_vi, x5_ir, x6_vi, x6_ir

@ARCH_REGISTRY.register()
class MTL_Fusion_Decoder(nn.Module):
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

    def forward(self, x1_vi, x1_ir, x2_vi, x2_ir, x3_vi, x3_ir, x4_vi, x4_ir, x5_vi, x5_ir, x6_vi, x6_ir):
        x1 = self.enhance_1(x1_vi, x1_ir)
        x2 = self.enhance_2(x2_vi, x2_ir)
        x3 = self.enhance_3(x3_vi, x3_ir)
        d4 = self.enhance_4(x4_vi, x4_ir)

        d3 = self.up_decoder3(self.neck(d4), x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))

        return out

@ARCH_REGISTRY.register() 
class MTL_Detection_Decoder(nn.Module):
    def __init__(self, in_channels=16, im_v=256, im_h=256, nc=6, vis=False):
        super().__init__()
        self.vis = vis
        self.in_channels = in_channels
        self.im_v = im_v
        self.im_h =im_h
        self.nc = nc
        self.det_fusion_1 = CrossFusionBlock(self.in_channels*16,int(self.im_v/32), int(self.im_h/32))
        self.det_fusion_2 = CrossFusionBlock(self.in_channels*16,int(self.im_v/64), int(self.im_h/64))
        self.det_fusion_3 = CrossFusionBlock(self.in_channels*32,int(self.im_v/128), int(self.im_h/128))

        self.us1 = nn.Upsample(None, 2, 'nearest') #9
        self.head_1 = A2C2f(768, 256, 1, False, -1) #11
        self.us2 = nn.Upsample(None, 2, 'nearest') #12
        self.head_2 = A2C2f(512, 128, 1, False, -1) #14
        self.headout_1 = Conv(128, 128, 3, 2) #15
        self.head_3 = A2C2f(384, 256, 1, False, -1) #17
        self.headout_2 = Conv(256, 256, 3, 2) #18
        self.head_4= C3k2(768, 512, 1, True) #20
        det_head = Detect(self.nc, [128, 256, 512]) #21
        if isinstance(det_head, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            det_head.inplace = True
            det_head.stride = torch.tensor([8., 16., 32.])  # forward
            self.stride = det_head.stride
            det_head.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  
        self.Decoder =  det_head
        self.con = Concat(1)

    def forward(self, x4_vi, x4_ir, x5_vi, x5_ir, x6_vi, x6_ir):
        x4 = self.det_fusion_1(x4_vi, x4_ir)
        x5 = self.det_fusion_2(x5_vi, x5_ir)
        x6 = self.det_fusion_3(x6_vi, x6_ir)

        x7 = self.us1(x6)
        x8 = self.con([x7, x5])
        x9 = self.head_1(x8)
        x10 = self.us2(x9)
        x11 = self.con([x10, x4])
        x12 = self.head_2(x11)
        x13 = self.headout_1(x12)
        x14 = self.con([x13, x9])
        x15 = self.head_3(x14)
        x16 = self.headout_2(x15)
        x17 = self.con([x16, x6])
        x18 = self.head_4(x17)
        x19 = self.Decoder([x12, x15, x18])
        
        return x19
