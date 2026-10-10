# -*- coding: utf-8 -*-
# @Time : 2021/5/24 下午4:36 
# @Author : midaskong 
# @File : prune_utils.py 
# @Description:

import torch
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def gather_bn_weights(module_list):
    # prune_idx = list(range(len(module_list)))
    # # module_list 是一个字典，键是层名，值是 BN 层对象,.keys()：取出所有的键（名字），比如 "model.5.bn"。.values()：取出所有的值（层对象），比如 BatchNorm2d(64)。
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]
    bn_weights = torch.zeros(sum(size_list)) # torch.zeros(512) 生成一个长度 512 的一维全零张量。
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights

def gather_conv_weights(module_list):
    # prune_idx = list(range(len(module_list)))
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]

    conv_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        conv_weights[index:(index + size)] = idx.weight.data.abs().sum(dim=1).sum(dim=1).sum(dim=1).clone()
        index += size
    return conv_weights


def obtain_bn_mask(bn_module, thre):
    thre = thre.cuda()
    # bn_module.weight.data：取出当前 BN 层的 gamma（缩放因子），它是一个一维张量，长度等于该层的通道数（比如 64）。
    # .abs()：取绝对值。剪枝只看“强度”（数值大小），不看正负号。-0.01 和 0.01 一样不重要。
    # .ge(thre)：ge 是 >=（大于等于）。逐元素比较：如果 gamma 的绝对值 ≥ thre，标记为 True；否则标记为 False。得到的是一个布尔张量（[True, False, True, ...]）。
    # .float()：把 True 变成 1.0，False 变成 0.0。
    mask = bn_module.weight.data.abs().ge(thre).float()
    ones = (mask == 1.).sum().sum()
    while ones <= 1:
        thre -= 0.005 # 把阈值调低 0.005（相当于“放水”，让更多原本小于阈值的通道被判定为 ≥ 新阈值，从而被保留）。
        mask = bn_module.weight.data.abs().ge(thre).float()
        ones = (mask == 1.).sum().sum()
    return mask
    # thre = thre.cuda()
    # mask = bn_module.weight.data.abs().ge(thre).float()
    #
    # return mask


def obtain_conv_mask(conv_module, thre):
    thre = thre.cuda()
    mask = conv_module.weight.data.abs().sum(dim=1).sum(dim=1).sum(dim=1).ge(thre).float()
    return mask

def uodate_pruned_yolov5_cfg(model, maskbndict):
    # save pruned yolov5 model in yaml format:
    # model:
    #   model to be pruned
    # maskbndict:
    #   key     : module name
    #   value   : bn layer mask index
    return