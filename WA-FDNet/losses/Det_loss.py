import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from ultralytics.utils.loss import *
from losses.MTL_VI_loss import MTL_fuse_v8DetectionLoss

@LOSS_REGISTRY.register()
class Detloss_VI(nn.Module):
    def __init__(self, device='cuda', nc=6):
        super(Detloss_VI, self).__init__()
        self.device = torch.device('cuda' if device == 'cuda' else 'cpu')
        self.det_v8DetectionLoss = MTL_fuse_v8DetectionLoss(self.device, nc=nc)

    def forward(self, preds, batch):
        self.loss_det, self.loss_items = self.det_v8DetectionLoss(preds, batch)

        return self.loss_det