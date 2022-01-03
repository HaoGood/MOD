# GIoU and Linear IoU are added by following
# https://github.com/yqyao/FCOS_PLUS/blob/master/maskrcnn_benchmark/layers/iou_loss.py.
import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, ious, gious, weight=None, funcs="train"):
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            if funcs == "stats":
                return (losses * weight)
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            if funcs == "stats":
                return losses
            return losses.sum()
