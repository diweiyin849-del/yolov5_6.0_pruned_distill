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
from models.common import Conv

class BottleneckPruned(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3Pruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3Pruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in+cv2out, cv3out, 1)
        self.m = nn.Sequential(*[BottleneckPruned(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3SPPPruned(nn.Module):
    """C3 with SPP module (剪枝版本)"""
    def __init__(self, cv1in, cv1out, cv2out, cv3out,bottlecv1out,bottlecv2out, n=1, shortcut=True, g=1, k=(5, 9, 13)):
        super(C3SPPPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        # SPP 模块：输入输出都是 cv1out
        self.m = SPPPruned(cv1out, bottlecv1out, bottlecv2out, k)
        # cv3 拼接后降维
        self.cv3 = Conv(bottlecv2out + cv2out, cv3out, 1,1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y1 = self.m(y1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPPruned(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, cv1in, cv1out,cv2out, k=(5, 9, 13)):
        super().__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out*4 , cv2out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPFPruned(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super(SPPFPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


"""
class GhostBottleneckPruned(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(GhostBottleneckPruned, self).__init__()
        self.cv1 = GhostConvPruned(cv1in, cv1out, 1, 1)
        self.cv2 = GhostConvPruned(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3GhostPruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3GhostPruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = GhostConvPruned(cv1in, cv1out, 1, 1)
        self.cv2 = GhostConvPruned(cv1in, cv2out, 1, 1)
        self.cv3 = GhostConvPruned(cv3in+cv2out, cv3out, 1)
        self.m = nn.Sequential(*[GhostBottleneckPruned(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
"""

class GhostConvPruned(nn.Module):
    # 剪枝版本，cv1 普通卷积，cv2 深度卷积（groups = cv1out）
    def __init__(self, cv1in, cv1out, cv3out,cv4out,cv5out,k=1, s=1, g=1):
        super().__init__()
        # cv1 使用传入的 g（通常为 1）
        self.cv1 = Conv(cv1in, cv1out, k, s, None, g)
        # cv2 是深度卷积，groups 固定为 cv1out
        self.cv2 = Conv(cv1out, cv1out, 3, 1, None, g=cv1out)
        self.cv3 = Conv(cv1out,cv3out, 1, 1, None, 1)
        # 空洞卷积
        self.cv4 = Conv(cv1out, cv4out, 3, 1, None, 1, 2)
        self.cv5 = Conv(cv3out+cv4out, cv5out, 3, 1, None, 1)


    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        y3 = self.cv3(y2)
        z1 = self.cv4(y1)

        return self.cv5(torch.cat([y3, z1], 1))
# ---------------------------- BiFPN ---------------------------------
class Concat_bifpnPruned(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, cv1in, cv1out):
        super(Concat_bifpnPruned, self).__init__()
        # 设置可学习参数 学习不同分支的权重
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分，
        # 即model.parameters()会包含这个parameter， 从而在参数优化的时候可以自动一起优化
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = Conv(cv1in, cv1out, 1, 1, 0)
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




class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module (ICML 2021)
    输入/输出: (B, C, H, W) -> (B, C, H, W)

    说明：
      - 为了适配 Ultralytics v11 的 parse_model（会自动注入 channels=C），
        这里 __init__ 的第一个参数必须接收 channels（即使 SimAM 本身不依赖 C）。
      - 注意力权重为逐元素 (B,C,H,W)，属于“像素/位置级”重标定。
    """

    def __init__(self, c1=None, c2=None, e_lambda: float = 1e-4):
        """
        Args:
            channels: 输入通道数 C（由 parse_model 自动注入；SimAM 内部不使用，仅占位适配）
            e_lambda: 稳定项（避免除零），论文/常用实现一般取 1e-4
        """
        super().__init__()
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
        return x * y             # 广播乘法


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


