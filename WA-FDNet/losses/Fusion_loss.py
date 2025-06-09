import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from ultralytics.utils.loss import *
from scripts.util import RGB2YCrCb, YCrCb2RGB
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

@LOSS_REGISTRY.register()
class Fusionloss_VI(nn.Module):
    def __init__(self, device='cuda'):
        super(Fusionloss_VI, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, alpha, image_vis, image_ir, generate_img):
        image_vis = RGB2YCrCb(image_vis)[:,:1,:,]
        image_y = image_vis
        B, C, H, W = image_vis.shape
        image_ir = image_ir.expand(B, C, H, W)
        x_in_max = torch.max(image_y, image_ir)
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)

        loss_in =  F.l1_loss(generate_img, x_in_max)
        loss_grad  = F.l1_loss(generate_img_grad, x_grad_joint)
        loss_fusion = alpha*(loss_in) + (1-alpha)*loss_grad

        return loss_fusion
