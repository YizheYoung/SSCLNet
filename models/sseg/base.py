#--------------------------723_hrnet40_contra-------------------------------------------
from models.backbone.hrnet import HRNet
from models.backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d

import torch
from torch import nn
import torch.nn.functional as F


def get_backbone(backbone, pretrained):
    if backbone == "resnet18":
        backbone = resnet18(pretrained)
    elif backbone == "resnet34":
        backbone = resnet34(pretrained)
    elif backbone == "resnet50":
        backbone = resnet50(pretrained)
    elif backbone == "resnet101":
        backbone = resnet101(pretrained)
    elif backbone == "resnet152":
        backbone = resnet152(pretrained)

    elif backbone == "resnext50":
        backbone = resnext50_32x4d(pretrained)
    elif backbone == "resnext101":
        backbone = resnext101_32x8d(pretrained)

    elif "hrnet" in backbone:
        backbone = HRNet(backbone, pretrained)

    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone


class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)

    def base_forward(self, x1, x2):

        return 

    def forward(self, x1, x2, tta=False):
        if not tta:
            return self.base_forward(x1, x2)
        else:
            out1_scd, out2_scd = self.base_forward(x1, x2)
            out1_scd = F.softmax(out1_scd, dim=1)
            out2_scd = F.softmax(out2_scd, dim=1)

            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out1_scd, cur_out2_scd = self.base_forward(x1, x2)
            out1_scd += F.softmax(cur_out1_scd, dim=1).flip(2)
            out2_scd += F.softmax(cur_out2_scd, dim=1).flip(2)

            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out1_scd, cur_out2_scd = self.base_forward(x1, x2)
            out1_scd += F.softmax(cur_out1_scd, dim=1).flip(3)
            out2_scd += F.softmax(cur_out2_scd, dim=1).flip(3)

            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out1_scd, cur_out2_scd = self.base_forward(x1, x2)
            out1_scd += F.softmax(cur_out1_scd, dim=1).flip(3).transpose(2, 3)
            out2_scd += F.softmax(cur_out2_scd, dim=1).flip(3).transpose(2, 3)

            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out1_scd, cur_out2_scd = self.base_forward(x1, x2)
            out1_scd += F.softmax(cur_out1_scd, dim=1).transpose(2, 3).flip(3)
            out2_scd += F.softmax(cur_out2_scd, dim=1).transpose(2, 3).flip(3)

            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out1_scd, cur_out2_scd = self.base_forward(x1, x2)
            out1_scd += F.softmax(cur_out1_scd, dim=1).flip(3).flip(2)
            out2_scd += F.softmax(cur_out2_scd, dim=1).flip(3).flip(2)

            out1_scd /= 6.0
            out2_scd /= 6.0

            return out1_scd, out2_scd
        
