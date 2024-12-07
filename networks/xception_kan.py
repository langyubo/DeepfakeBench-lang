
import functools
import os
import argparse
import logging

import math
import torch
import torch.autograd
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F


from .attention import SpatialAttention, ChannelAttention
from .efficient_kan import KANLinear
from .attention import simam_module

import torch.utils.model_zoo as model_zoo
from torch.nn import init
from typing import Union
from metrics.registry import BACKBONE


logger = logging.getLogger(__name__)

class SeparableConv2d(nn.Module):
    """
    初始化一个可分离卷积2D层。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int or tuple, optional): 卷积核的大小。可以是一个整数，表示所有维度上使用同一个大小，或者是一个元组，每个维度上使用不同的大小。默认为1。
    - stride (int or tuple, optional): 卷积步长。与kernel_size的解释相同。默认为1。
    - padding (int or tuple, optional): 卷积前的填充。与kernel_size的解释相同。默认为0。
    - dilation (int or tuple, optional): 卷积核元素之间的间距。与kernel_size的解释相同。默认为1。
    - bias (bool, optional): 是否在卷积层中使用偏置。默认为False。

    返回值:
    - 无
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 初始化深度卷积层，用于按通道进行卷积。
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        # 初始化点卷积层，用于按通道权重缩放。
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        """
        初始化Block类的实例。

        参数:
        - in_filters: 输入通道数。
        - out_filters: 输出通道数。
        - reps: 稀疏卷积层的数量。
        - strides: 卷积步长，默认为1。
        - start_with_relu: 是否以ReLU激活函数开始，默认为True。
        - grow_first: 是否先增加过滤器数量，默认为True。

        不返回值，但构建了网络结构。
        """
        super(Block, self).__init__()

        # 根据输入输出通道数和步长决定是否添加跳跃连接
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        # 根据grow_first标志决定是否先增加过滤器数量
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        # 添加reps-1个稀疏卷积层
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        # 如果grow_first为False，则在此增加过滤器数量
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        # 根据start_with_relu标志决定是否去掉第一个ReLU激活函数
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        # 如果步长不为1，则添加最大池化层
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


def add_gaussian_noise(ins, mean=0, stddev=0.2):
    """
    向输入数据添加高斯噪声。
    该函数为输入的张量添加基于高斯分布的噪声，可以用于数据增强或模拟真实世界中的噪声。
    param ins 输入的张量，要求具有可加性。
    param mean 噪声的均值,默认为0。
    param stddev 噪声的标准差,默认为0.2。
    return 返回添加了高斯噪声的输入张量。
    """
    # 生成与输入数据大小相同的高斯噪声张量
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


@BACKBONE.register_module(module_name="xception_kan")
class Xception_kan(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_kan_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception_kan, self).__init__()
        self.num_classes = xception_kan_config["num_classes"]
        self.mode = xception_kan_config["mode"]
        inc = xception_kan_config["inc"]
        dropout = xception_kan_config["dropout"]
        include_top = xception_kan_config["kan"]
        attention_type = xception_kan_config["attention_type"]
        # attention_param = xception_kan_config["attention_param"]
        if attention_type == "simam":
            self.attention1 = functools.partial(simam_module, e_lambda=1e-4)
        else:
            self.attention1 = None

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)

        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)
    
        # middle flow
        self.block4 = nn.Sequential(
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728)
        )
        self.block5 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))
        self.block6 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))
        self.block7 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))

        self.block8 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))
        self.block9 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))
        self.block10 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))
        self.block11 = nn.Sequential(Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True),
            self.attention1(728))

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # self.last_linear = nn.Linear(2048, self.num_classes)
        # self.last_linear = KANLinear(2048, self.num_classes)

        if include_top:
            self.last_linear = KANLinear(2048, self.num_classes)
        else:
            self.last_linear = nn.Linear(2048, self.num_classes)

        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)
        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)

        return x

    def classifier(self, features):
        """
        执行分类器的功能。
        
        参数:
        - features: 输入的特征数据，预期是一个四维张量，格式为(batch_size, channels, height, width)。
        
        返回:
        - out: 经过分类器处理后的输出，是一个二维张量，格式为(batch_size, num_classes)，其中num_classes是类别数量。
        """
        # 使用ReLU激活函数对输入特征进行非线性变换
        x = self.relu(features)

        # 对特征进行自适应平均池化，将特征图尺寸缩放到1x1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # 将特征图展平为一维向量，准备进行全连接层的计算
        x = x.view(x.size(0), -1)
        # 通过最后一个线性层（全连接层）进行分类预测
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x
