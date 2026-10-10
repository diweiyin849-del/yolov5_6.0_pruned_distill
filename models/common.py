# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
import torch.nn.functional as F

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync


def autopad(k, p=None, d=1):  # 增加 d
    if p is None:
        # 计算正确填充：考虑膨胀率
        p = (k + (k-1)*(d-1) - 1) // 2
    return p


class Conv(nn.Module):
    # Standard convolution with dilation support
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):  # 新增 d=1
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 传入 dilation=d
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)



class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3SPP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, k=(5, 9, 13), g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = SPP(c_, c_, k)
        self.cv3 = Conv(2 * c_, c2, 1)
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y1 = self.m(y1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class C2FBottleneck(nn.Module):
    """Standard bottleneck for C2f module."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2FBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))



class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)




class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        # 深度可分离卷积
        self.cv1 = Conv(c1, c2, k, s, None, g)
        self.cv2 = Conv(c2, c2, 3, 1, None, c2)
        self.cv3 = Conv(c2, c2, 1, 1, None, 1)
        # 空洞卷积
        self.cv4 = Conv(c2, c2, 3, 1, None, 1, 2)
        self.cv5 = Conv(2*c2, c2, 3, 1, None, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        y3 = self.cv3(y2)
        z1 = self.cv4(y1)

        return self.cv5(torch.cat([y3, z1], 1))



class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet    s=1 时：只能用于 c1 == c2（特征提取，不改变通道数）。 s=2 时：可以用于 c1 != c2（下采样 + 通道翻倍）。
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # 自动适配g分组数目
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),     # 自动适配g分组数目
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class C3Ghost(nn.Module):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2,k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# ---------------------------- MobileNetV2 -------------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, c1, c2, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(c1 * expand_ratio))
        self.use_res_connect = self.stride == 1 and c1 == c2

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(c1, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ---------------------------- MobileNetV3 -------------------------------

class ConvBNHswish(nn.Module):
    def __init__(self, c1, c2, stride):
        super(ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SELayer(nn.Module):
    def __init__(self, c1,c2, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作 (FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c2),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # 学习到的每一个channel的权重
        return x * y

class MblNetV3_Blk(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MblNetV3_Blk, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        # 输入通道数=扩张通道数，则不进行通道扩张
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim,hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # 否则，先进行通道扩张
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim,hidden_dim) if use_se else nn.Sequential(),
                nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y

# ---------------------------- ShuffleNet -------------------------------

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,channels_per_group, height, width) # 形状: (B, 2, 4, H, W)

    x = torch.transpose(x, 1, 2).contiguous() # 交换后形状: (B, 4, 2, H, W)

    # flatten
    x = x.view(batchsize, -1, height, width) # 形状变回: (B, 8, H, W) 转置且洗牌了

    return x


class ConvBNReLUMaxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(ConvBNReLUMaxpool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class ShuffleNet_Blk(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleNet_Blk, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # 按照维度1进行split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out



class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class SENet(nn.Module):
    def __init__(self, c1, c2,ratio=16):
        super(SENet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Conv2d(c1, c1 // ratio, kernel_size=1, bias=False)
        self.l2 = nn.Conv2d(c1 // ratio, c2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        return x * y                 # 广播乘法


class BAM(nn.Module):
    """
    BAM: Bottleneck Attention Module (BMVC 2018)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    适配说明：
      - 你的 parse_model 会自动注入 channels=C 作为第一个构造参数，
        因此 __init__ 必须以 channels 开头。

    BAM = Channel Gate + Spatial Gate，然后融合成一个注意力图 M：
      M = sigmoid( Mc + Ms )
      out = x * (1 + M)   # 论文常用 residual gating（可选）
    """

    def __init__(self,c1: int,c2: int,reduction: int = 16,dilations=(1, 2, 4),):
        super().__init__()
        assert c1 > 0
        self.channels = c1
        # -------------------------
        # 1) Channel Gate (Mc)
        # -------------------------
        mid = max(1, c1 // reduction)

        # GAP: (B,C,H,W)->(B,C,1,1)
        # MLP: C -> mid -> C
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                   # (B,C,1,1)
            nn.Conv2d(c1, mid, 1, bias=False),   # (B,mid,1,1)
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c2, 1, bias=False),   # (B,C,1,1)
        )

        # -------------------------
        # 2) Spatial Gate (Ms)
        # -------------------------
        # 先用 1x1 降维，再做多尺度空洞 3x3，最后 1x1 输出 1 通道的空间注意力
        sg_mid = max(1, c1 // reduction)

        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(c1, sg_mid, 1, bias=False),  # (B,sg_mid,H,W)
            nn.BatchNorm2d(sg_mid),
            nn.ReLU(inplace=True),
        )

        # 多空洞卷积堆叠（保持 H,W 不变）
        convs = []
        for d in dilations:
            convs += [
                nn.Conv2d(sg_mid, sg_mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(sg_mid),
                nn.ReLU(inplace=True),
            ]
        self.spatial_convs = nn.Sequential(*convs)

        # 输出 1 通道空间门控： (B,sg_mid,H,W)->(B,1,H,W)
        self.spatial_out = nn.Conv2d(sg_mid, 1, 1, bias=False)

        # -------------------------
        # 3) Fusion
        # -------------------------
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # Channel gate:
        # Mc: (B, C, 1, 1)
        Mc = self.channel_gate(x)

        # Spatial gate:
        # Ms: (B, 1, H, W)
        s = self.spatial_reduce(x)          # (B, sg_mid, H, W)
        s = self.spatial_convs(s)           # (B, sg_mid, H, W)
        Ms = self.spatial_out(s)            # (B, 1, H, W)

        # 融合成注意力图：
        # Mc 广播到 (B,C,H,W)，Ms 广播到 (B,C,H,W)
        # M: (B, C, H, W)
        M = self.sigmoid(Mc + Ms)

        # residual gating（论文常用）：out = x * (1 + M)
        return x * (1.0 + M)


class A2(nn.Module):
    """
    A2-Nets: Double Attention Block (NeurIPS 2018)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    适配说明：
      - 你的 parse_model 会把 channels=C 自动注入到构造函数第一个参数，
        所以这里 __init__(channels, ...) 必须以 channels 开头。

    核心思想：Gather + Distribute（双注意力）
      1) Gather：从全局空间 (H*W) 里“聚合”出一组全局特征描述子
      2) Distribute：再把这些全局描述子按位置需要“分配”回每个位置
    """

    def __init__(self, c1: int,c2: int, reduction: int = 4, L: int = 32):
        """
        Args:
            channels: 输入通道数 C（由 parse_model 注入）
            reduction: 压缩比，inter = max(C//reduction, L)
            L: inter 的最小值（防止通道太小）
        """
        super().__init__()
        assert c1 > 0
        self.channels = c1
        inter = max(c1 // reduction, L)  # 中间维度 d
        self.inter = inter

        # 3 个 1x1 conv：分别产生 A（gather 权重）、B（被聚合的特征）、V（distribute 权重）
        self.convA = nn.Conv2d(c1, inter, kernel_size=1, bias=False)  # -> (B, inter, H, W)
        self.convB = nn.Conv2d(c1, inter, kernel_size=1, bias=False)  # -> (B, inter, H, W)
        self.convV = nn.Conv2d(c1, inter, kernel_size=1, bias=False)  # -> (B, inter, H, W)

        # 输出投影回 C
        self.proj = nn.Sequential(
            nn.Conv2d(inter, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # -------- Step 1) 生成三路特征 --------
        # A_map: (B, inter, H, W) -> (B, inter, N)
        A_map = self.convA(x).view(B, self.inter, N)
        # B_feat: (B, inter, H, W) -> (B, inter, N)
        B_feat = self.convB(x).view(B, self.inter, N)
        # V_map: (B, inter, H, W) -> (B, inter, N)
        V_map = self.convV(x).view(B, self.inter, N)

        # -------- Step 2) Gather：在空间维 N 上 softmax，做“全局聚合” --------
        # A_attn: (B, inter, N)，对每个“通道分组/基向量”在空间上归一化
        A_attn = F.softmax(A_map, dim=2)

        # G: (B, inter, inter)
        # (B, inter, N) @ (B, N, inter) -> (B, inter, inter)
        G = torch.bmm(B_feat, A_attn.permute(0, 2, 1))

        # -------- Step 3) Distribute：在 inter 维上 softmax，做“按位置分配” --------
        # V_attn: (B, inter, N)，对每个位置在 inter 维归一化（分配权重）
        V_attn = F.softmax(V_map, dim=1)

        # Y: (B, inter, N)
        # (B, inter, inter) @ (B, inter, N) -> (B, inter, N)
        Y = torch.bmm(G, V_attn)

        # (B, inter, N) -> (B, inter, H, W)
        Y = Y.view(B, self.inter, H, W)

        # -------- Step 4) 投影回 C + 残差 --------
        out = self.proj(Y)  # (B, C, H, W)
        return x + out

class SK(nn.Module):
    """
    SK (Selective Kernel) / SKConv
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    YOLOv5 兼容点：
    - YOLOv5 初始化会用 batch=1 做一次 forward 来推 stride
    - GAP 后是 (B, C, 1, 1)，此时 BatchNorm2d 在 train 模式下会报错
    - 所以 bottleneck 的 BN 必须去掉或改成 GroupNorm/LayerNorm
    """

    def __init__(self,c1: int,c2: int,kernels=(3, 5),reduction: int = 16,L: int = 32,groups: int = 1,act: str = "silu",):
        super().__init__()
        assert c1 > 0
        self.channels = c1
        self.kernels = list(kernels)
        self.M = len(self.kernels)

        # 激活函数
        if act.lower() == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)

        # ---- 1) Split: 多分支卷积 ----
        self.branches = nn.ModuleList()
        for k in self.kernels:
            p = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size=k, padding=p, groups=groups, bias=False),
                    nn.BatchNorm2d(c1),
                    self.act,
                )
            )

        # ---- 2) Fuse: GAP ----
        self.gap = nn.AdaptiveAvgPool2d(1)

        # bottleneck 维度 d
        d = max(c1 // reduction, L)

        # ✅ 关键修改：这里不要 BatchNorm2d(d)，否则 (1,d,1,1) 会报错
        self.fc = nn.Sequential(
            nn.Conv2d(c1, d, kernel_size=1, bias=False),
            self.act,
        )

        # 分支 logits
        self.fc_branches = nn.ModuleList([nn.Conv2d(d, c2, kernel_size=1, bias=True) for _ in range(self.M)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split
        feats = [br(x) for br in self.branches]          # list of (B,C,H,W)
        feats_stack = torch.stack(feats, dim=1)          # (B,M,C,H,W)

        # Fuse
        U = feats_stack.sum(dim=1)                       # (B,C,H,W)
        s = self.gap(U)                                  # (B,C,1,1)
        z = self.fc(s)                                   # (B,d,1,1)

        # Select
        logits = torch.stack([fc(z) for fc in self.fc_branches], dim=1)  # (B,M,C,1,1)
        attn = F.softmax(logits, dim=1)                                  # (B,M,C,1,1)

        # Weighted sum
        out = (feats_stack * attn).sum(dim=1)            # (B,C,H,W)
        return out



class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module (ICML 2021)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    说明：
      - 为了适配 Ultralytics v11 的 parse_model（会自动注入 channels=C），
        这里 __init__ 的第一个参数必须接收 channels（即使 SimAM 本身不依赖 C）。
      - 注意力权重为逐元素 (B,C,H,W)，属于“像素/位置级”重标定。
    """

    def __init__(self, c1: int,c2: int, e_lambda: float = 1e-4):
        """
        Args:
            channels: 输入通道数 C（由 parse_model 自动注入；SimAM 内部不使用，仅占位适配）
            e_lambda: 稳定项（避免除零），论文/常用实现一般取 1e-4
        """
        super().__init__()
        self.channels = c1  # 可选：存一下方便 debug（不影响计算）
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # ---- Step A: 通道均值 ----
        # mu: (B, C, 1, 1)
        mu = x.mean(dim=(2, 3), keepdim=True)
        # ---- Step B: 偏差平方 ----
        # d: (B, C, H, W)
        d = (x - mu).pow(2)
        # ---- Step C: 通道内均方（方差项）----
        # v: (B, C, 1, 1)
        v = d.mean(dim=(2, 3), keepdim=True)
        # ---- Step D: 能量评分（常用简化实现）----
        # score: (B, C, H, W)
        score = d / (4.0 * (v + self.e_lambda)) + 0.5
        # ---- Step E: 注意力权重（逐元素）----
        # attn: (B, C, H, W)
        attn = self.sigmoid(score)
        # ---- Step F: 重标定输出 ----
        # out: (B, C, H, W)
        return x * attn



class ZPool(nn.Module):
    """
    Z-Pool: 通道维的 AvgPool + MaxPool 拼接
    输入:  (B, C, H, W)
    输出:  (B, 2, H, W)  其中 dim=1 的 2 来自 [avg, max]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # avg: (B,1,H,W), max: (B,1,H,W)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        # cat: (B,2,H,W)
        return torch.cat([avg, mx], dim=1)


class BasicConv(nn.Module):
    """
    Conv + BN (+ 可选 ReLU)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionGate(nn.Module):
    """
    Triplet Attention 的核心 Gate：
      x -> ZPool -> 7x7 Conv -> Sigmoid -> 与 x 相乘

    输入:  (B, C, H, W)
    输出:  (B, C, H, W)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.zpool = ZPool()
        p = kernel_size // 2
        # (B,2,H,W) -> (B,1,H,W)
        self.conv = BasicConv(2, 1, k=kernel_size, p=p, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z: (B,2,H,W)
        z = self.zpool(x)
        # attn: (B,1,H,W)
        attn = self.sigmoid(self.conv(z))
        # 广播到通道维： (B,C,H,W) * (B,1,H,W)
        return x * attn


class TripletAttention(nn.Module):
    """
    Triplet Attention (2021)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    思想：
      不只做 (C,H,W) 上的常规空间注意力，
      而是通过“维度置换”在三个视角上做注意力：
        1) (C, H) 视角：对 W 方向建模（通过 permute 实现）
        2) (C, W) 视角：对 H 方向建模
        3) (H, W) 视角：常规空间注意力
      最后把三路结果平均融合。

    适配说明：
      - 你的 parse_model 会注入 channels=C 到 __init__ 的第一个参数，
        但 TripletAttention 本身不依赖 C，所以这里 channels 仅作占位适配。
    """

    def __init__(self, c1: int, c2: int,kernel_size: int = 7, no_spatial: bool = False):
        """
        Args:
            channels: 输入通道数 C（parse_model 注入，占位即可）
            kernel_size: Gate 中 7x7 卷积核大小
            no_spatial: 是否去掉常规 HW 分支（默认 False：三分支都用）
        """
        super().__init__()
        self.channels = c1
        self.no_spatial = no_spatial

        # 三个 Gate（其中 HW 可选）
        self.gate_cw = AttentionGate(kernel_size=kernel_size)  # 处理 C-W 视角（建模 H）
        self.gate_ch = AttentionGate(kernel_size=kernel_size)  # 处理 C-H 视角（建模 W）
        self.gate_hw = AttentionGate(kernel_size=kernel_size)  # 常规 H-W 视角

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        # ---- 分支1：C-W 视角（通过交换 H 和 C）----
        # x1: (B, H, C, W)
        x1 = x.permute(0, 2, 1, 3).contiguous()
        # gate 后仍是 (B, H, C, W)
        x1 = self.gate_cw(x1)
        # 变回 (B, C, H, W)
        x1 = x1.permute(0, 2, 1, 3).contiguous()

        # ---- 分支2：C-H 视角（通过交换 W 和 C）----
        # x2: (B, W, H, C)
        x2 = x.permute(0, 3, 2, 1).contiguous()
        x2 = self.gate_ch(x2)
        # 变回 (B, C, H, W)
        x2 = x2.permute(0, 3, 2, 1).contiguous()

        # ---- 分支3：常规 HW 视角 ----
        if self.no_spatial:
            out = (x1 + x2) / 2.0
        else:
            x3 = self.gate_hw(x)  # (B,C,H,W)
            out = (x1 + x2 + x3) / 3.0

        return out



class SLAM(nn.Module):
    """
    SLAM: Spatial Location Attention Module (2025)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    三个分支生成注意力图：
      1) 垂直方向: M_H 形状 (B, C, 1, W)   —— pool over H
      2) 水平方向: M_W 形状 (B, C, H, 1)   —— pool over W
      3) 空间方向: M_S 形状 (B, 1, H, W)   —— pool over C

    最终输出:
      out = x * M_H * M_W * M_S    (利用广播机制 broadcast)
    """
    def __init__(self, c1: int, c2: int,reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: 输入通道数 C（⚠️必须放第一个，匹配 parse_model 的注入）
            reduction: 1x1瓶颈压缩比（W0/W1）
            kernel_size: 空间分支卷积核大小，论文常用 7
        """
        super().__init__()
        assert c1 > 0
        hidden = max(1, c1 // reduction)
        self.h_conv1 = nn.Conv2d(c1, hidden, kernel_size=1, bias=False)  # W0: C -> hidden
        self.h_act = nn.ReLU(inplace=True)
        self.h_conv2 = nn.Conv2d(hidden, c2, kernel_size=1, bias=False)  # W1: hidden -> C
        self.w_conv1 = nn.Conv2d(c1, hidden, kernel_size=1, bias=False)  # W0: C -> hidden
        self.w_act = nn.ReLU(inplace=True)
        self.w_conv2 = nn.Conv2d(hidden, c2, kernel_size=1, bias=False)  # W1: hidden -> C
        # --------- S（空间）分支：7x7 卷积（论文写作 f_{7x7}^{DW}）---------
        pad = kernel_size // 2
        self.s_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        # ============================================================
        # 1) 水平方向注意力 M_W : (B, C, 1, W)
        #    = sigmoid( W1(W0( F_H_avg + F_H_max )) )
        #    这里“pool over H”，保留 W
        # ============================================================
        fh_avg = x.mean(dim=2, keepdim=True)                 # (B, C, 1, W)
        fh_max = torch.max(x, dim=2, keepdim=True)[0]        # (B, C, 1, W)
        fh = fh_avg + fh_max                                 # ⊕  (B, C, 1, W)
        mh = self.w_conv2(self.w_act(self.w_conv1(fh)))   # (B, C, 1, W)
        mh = self.sigmoid(mh)                                 # σ  (B, C, 1, W)
        # ============================================================
        # 2) 垂直方向注意力 M_H : (B, C, H, 1)
        #    = sigmoid( W1(W0( F_W_avg + F_W_max )) )
        #    这里“pool over W”，保留 H
        # ============================================================
        fw_avg = x.mean(dim=3, keepdim=True)                 # (B, C, H, 1)
        fw_max = torch.max(x, dim=3, keepdim=True)[0]        # (B, C, H, 1)
        fw = fw_avg + fw_max                                 # ⊕  (B, C, H, 1)
        mw = self.h_conv2(self.h_act(self.h_conv1(fw)))   # (B, C, H, 1)
        mw = self.sigmoid(mw)                                 # σ  (B, C, H, 1)
        # ============================================================
        # 3) 通道注意力 M_S : (B, 1, H, W)
        #    = sigmoid( f_{7x7}( F_S_avg + F_S_max ) )
        #    这里“pool over C”，得到 1×H×W
        # ============================================================
        fs_avg = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)
        fs_max = torch.max(x, dim=1, keepdim=True)[0]        # (B, 1, H, W)
        fs = fs_avg + fs_max                                 # ⊕  (B, 1, H, W)
        ms = self.s_conv(fs)                                  # (B, 1, H, W)
        ms = self.sigmoid(ms)                                 # σ  (B, 1, H, W)
        # ============================================================
        # 4) 逐元素相乘（⊗），依赖广播：
        #    mh 会在 H 维广播，mw 会在 W 维广播，ms 会在 C 维广播
        # ============================================================
        out = x * mh * mw * ms                                # (B, C, H, W)
        return out


def _make_divisible_groups(channels: int, groups: int) -> int:
    """让 GN 的 groups 一定能整除 channels（工程里更稳）"""
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g

class ELA(nn.Module):
    """
    ELA: Efficient Local Attention (arXiv 2024)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    流程（与 CA 类似，但更轻更直接）：
      1) strip pooling: 得到 (B,C,H,1) 和 (B,C,1,W)
      2) 1D conv(沿 H/W) + GN: 生成两个方向的注意力权重
      3) sigmoid 后逐元素缩放：out = x * a_h * a_w
    """

    def __init__(self, c1: int, c2: int,k: int = 7, gn_groups: int = 8, use_residual: bool = True):
        super().__init__()
        assert k % 2 == 1, "k 建议用奇数"

        self.C = c1
        self.k = k
        self.use_residual = use_residual

        g = _make_divisible_groups(c1, gn_groups)

        # 对输入做 GN（论文强调 GN 相比 BN 更稳）
        self.gn = nn.GroupNorm(num_groups=g, num_channels=c1)

        # 两个方向各一条 1D conv：保持通道数不变（不降维）
        # Conv1d 输入是 (B, C, L)
        self.conv_h = nn.Conv1d(c1, c2, kernel_size=k, padding=k // 2, groups=1, bias=True)
        self.conv_w = nn.Conv1d(c1, c2, kernel_size=k, padding=k // 2, groups=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        x_gn = self.gn(x)  # (B, C, H, W)

        # 1) strip pooling（沿某一维做平均）
        x_h = x_gn.mean(dim=3, keepdim=True)  # (B, C, H, 1)  在 W 上池化
        x_w = x_gn.mean(dim=2, keepdim=True)  # (B, C, 1, W)  在 H 上池化

        # 2) 变成 Conv1d 需要的 (B, C, L)
        x_h_1d = x_h.squeeze(3)              # (B, C, H)
        x_w_1d = x_w.squeeze(2)              # (B, C, W)

        # 3) 1D conv + sigmoid 得到方向权重
        a_h = self.sigmoid(self.conv_h(x_h_1d))  # (B, C, H)
        a_w = self.sigmoid(self.conv_w(x_w_1d))  # (B, C, W)

        # 4) reshape 回来用于广播相乘
        a_h = a_h.unsqueeze(3)  # (B, C, H, 1)
        a_w = a_w.unsqueeze(2)  # (B, C, 1, W)

        out = x * a_h * a_w     # (B, C, H, W)  广播到对应维度
        if self.use_residual:
            out = out + x
        return out




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.max_pool=nn.AdaptiveMaxPool2d((1,1))
        self.fc1=nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid=nn.Sigmoid()  # sigmoid之后就是注意力矩阵

    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Defining the spatial module atenttion
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size,padding=3,bias=False)
        # kernel size = 7 padding is 3: (n-7+1) +2p = n
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True) # keepdim=True 保持dim=1是为了便于后续与其他形状为 (b, 1, h, w) 的特征图进行拼接或广播运算。
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=True,g=1,e=0.5):
        super(CBAM,self).__init__()
        self.channel_attention= ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)*x
        out = self.spatial_attention(out)*out
        return out


class CBAMBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # 加入CBAM模块
        self.channel_attention = ChannelAttention(c2, 16)
        self.spatial_attention = SpatialAttention(7)

    def forward(self, x):
        x2 = self.cv2(self.cv1(x))  # x和x2的channel数相同
        # 在bottleneck模块中shortcut之前加入CBAM模块
        out = self.channel_attention(x2) * x2
        out = self.spatial_attention(out) * out
        return x + out if self.add else out
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CBAMC3(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut) for _ in range(n)))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CANet(nn.Module):
    def __init__(self, in_channels, oup, groups=32):
        super(CANet, self).__init__()
        # CA是一种带坐标位置信息的通道注意力：把空间信息沿H/W两个方向分解
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) #None 表示不改变该维度的大小（即保持原始高度 H）,1 表示将该维度池化到长度为 1（即w方向压缩为 1）
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) #None 表示不改变该维度的大小（即保持原始宽度 W）,1 表示将该维度池化到长度为 1（即h方向压缩为 1）
        mip = max(8, in_channels // groups)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.Hardswish(inplace=True)  # h_swish()

    def forward(self, x):
        identity = x
        b,c,h,w = x.size()
        x_h = self.pool_h(x)  # b,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2) #b,c,w,1
        y = torch.cat([x_h, x_w], dim=2)  # B*C*(W+H)*1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2) #x_h 形状：(B, mip, H, 1)x_w 形状：(B, mip, W, 1)
        x_w = x_w.permute(0, 1, 3, 2) # B,mip,1,w
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)# x_h.expand(-1, -1, h, w)：将 (B, oup, H, 1) 在宽度维上复制 w 次，变成 (B, oup, H, W)。此时同一行内的所有列共享相同的行权重。
        x_w = x_w.expand(-1, -1, h, w)# x_w.expand(-1, -1, h, w)：将 (B, oup, 1, W) 在高度维上复制 h 次，变成 (B, oup, H, W)。此时同一列内的所有行共享相同的列权重。
        y = identity * x_w * x_h
        return y


class CABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # 加入CA模块
        self.ca_attention = CANet(c1, c2)

    def forward(self, x):
        x2 = self.cv2(self.cv1(x))  # x和x2的channel数相同
        # 在bottleneck模块中shortcut之前加入CA模块
        out = self.ca_attention(x2)
        return x + out if self.add else out
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CAC3(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut) for _ in range(n)))


class ECANet(nn.Module):
    def __init__(self, c1, c2,k_size=5):
        super(ECANet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # 轻量化通道注意力机制
        y = self.conv(y.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)  #注意力权重矩阵
        y = x * y.expand_as(x)
        return y


class GAMNet(nn.Module):
    # 像素级的通道注意力+ 空间注意力
    def __init__(self,c1,c2, rate=4):
        super(GAMNet, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(c1,c1//rate),
            nn.ReLU(inplace=True),
            nn.Linear(c1//rate,c1)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1,c1//rate, kernel_size=7, padding=3),
            nn.BatchNorm2d(c1//rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//rate,c1, kernel_size=7, padding=3),
            nn.BatchNorm2d(c1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  #Linear 层会被应用于 H*W 次（每个空间位置一次）
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()  #最终的注意力矩阵才需要sigmoid;
        out = x * x_spatial_att  #in_channels和out_channels必须相同
        return out




class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        #   TensorRT:               *.engine
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '8.0.0', verbose=True)  # version requirement
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # torch to numpy
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.engine or self.onnx:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup


# ------------------------------剪枝添加代码------------------------------------
class DetectPrunedMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        #   TensorRT:               *.engine
        from models.experimental import attempt_download, attempt_load_pruned  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            model = attempt_load_pruned(weights if isinstance(weights, list) else w, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '8.0.0', verbose=True)  # version requirement
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # torch to numpy
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.engine or self.onnx:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup

# ------------------------------剪枝添加代码------------------------------------

class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

"""
# ---------------------------- BiFPN ---------------------------------
class Concat_bifpn(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat_bifpn, self).__init__()
        # 设置可学习参数 学习不同分支的权重
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分，
        # 即model.parameters()会包含这个parameter， 从而在参数优化的时候可以自动一起优化
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):  # mutil-layer 1-3 layers #ADD or Concat
        # print("bifpn:",x.shape)
        if len(x) == 2:  # 两个分支add操作
            w = self.w1_weight
            # 将权重进行归一化
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            # Fast normalized fusion
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:  # 三个分支add操作
            w = self.w2_weight
            # 将权重进行归一化
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            # Fast normalized fusion
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x

    # ---------------------------- ASFF ---------------------------------
"""

# ---------------------------- BiFPN ---------------------------------
class Concat_bifpn(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat_bifpn, self).__init__()
        # 设置可学习参数 学习不同分支的权重
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分，
        # 即model.parameters()会包含这个parameter， 从而在参数优化的时候可以自动一起优化
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):  # mutil-layer 1-3 layers #ADD or Concat
        # print("bifpn:",x.shape)
        if len(x) == 2:  # 两个分支cat操作
            w = self.w1_weight
            # 将权重进行归一化
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            # Fast normalized fusion
            x = self.cv1(self.act(torch.cat((weight[0] * x[0] ,weight[1] * x[1]),dim=1)))
        elif len(x) == 3:  # 三个分支cat操作
            w = self.w2_weight
            # 将权重进行归一化
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            # Fast normalized fusion
            x = self.cv1(self.act(torch.cat((weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]), dim=1)))
        return x



class ASFFV5(nn.Module):
    def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5 which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

# ---------------------------- Slim-Neck ---------------------------------
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # ========== 正确的通道混洗（Channel Shuffle）==========
        b, c, h, w = x2.shape
        # 1. 重塑为 (b, 2, c//2, h, w) —— 把通道分成两半
        y = x2.reshape(b, 2, c // 2, h, w)
        # 2. 交换维度 1 和 2 —— 实现交错排列
        y = y.permute(0, 2, 1, 3, 4)
        # 3. 展平回 (b, c, h, w)
        return y.reshape(b, -1, h, w)


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1,))
        self.shortcut = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1,1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


