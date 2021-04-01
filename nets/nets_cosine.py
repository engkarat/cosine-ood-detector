from nets.densenet import DenseNet
from nets.wide_resnet import WideResNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetCosine(DenseNet):
    def __init__(
            self, depth, num_classes, growth_rate=12, reduction=0.5,
            bottleneck=True, drop_rate=0.0, input_n_channel=3,
        ):
        super(DenseNetCosine, self).__init__(
            depth, num_classes, growth_rate, reduction, bottleneck,
            drop_rate, input_n_channel,
        )
        self.fc = nn.Linear(self.in_planes, num_classes, bias=False)
        self.fc_w = nn.Parameter(self.fc.weight)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(self.in_planes, 1)

    def forward(
        self, x, y=None, mixup=None, alpha=None, all_pred=False,
        candidate_layers=[0, 1, 2, 3],
    ):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        scale = torch.exp(self.bn_scale(self.fc_scale(out)))
        x_norm = F.normalize(out)
        w_norm = F.normalize(self.fc_w)
        w_norm_transposed = torch.transpose(w_norm, 0, 1)

        cos_sim = torch.mm(x_norm, w_norm_transposed) # cos_theta
        scaled_cosine = cos_sim * scale
        softmax = F.softmax(scaled_cosine, 1)

        if all_pred:
            return scaled_cosine, softmax, scale, cos_sim
        else:
            return scaled_cosine, scale


class WideResNetCosine(WideResNet):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, input_n_channel=3):
        super(WideResNetCosine, self).__init__(
            depth, num_classes, widen_factor, drop_rate, False, input_n_channel=input_n_channel
        )
        self.fc = nn.Linear(64*widen_factor, num_classes, bias=False)
        self.fc_w = nn.Parameter(self.fc.weight)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(64*widen_factor, 1)

    def forward(
        self, x, y=None, mixup=None, alpha=None, all_pred=False,
        candidate_layers=[0, 1, 2, 3],
    ):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        scale = torch.exp(self.bn_scale(self.fc_scale(out)))
        x_norm = F.normalize(out)
        w_norm = F.normalize(self.fc_w)
        w_norm_transposed = torch.transpose(w_norm, 0, 1)

        cos_sim = torch.mm(x_norm, w_norm_transposed) # cos_theta
        scaled_cosine = cos_sim * scale
        softmax = F.softmax(scaled_cosine, 1)

        if all_pred:
            return scaled_cosine, softmax, scale, cos_sim
        else:
            return scaled_cosine, scale
