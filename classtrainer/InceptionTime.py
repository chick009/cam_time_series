
from torch import topk
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
from typing import cast, Union, List
import time

def conv1d_same_padding(in_channels, out_channels, kernel_size):
    stride = 1
    dilation = 1

    # Calculate the padding size
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

    # Create the convolutional layer with the calculated padding
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    return conv

class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck_channel = 2):
        super(InceptionBlock, self).__init__()
        
        kernel_size = [10, 20, 40]

        self.block1 = nn.Sequential(
            conv1d_same_padding(in_channel, bottleneck_channel, 1),
            conv1d_same_padding(bottleneck_channel, out_channel, kernel_size[0])
        )

        self.block2 = nn.Sequential(
            conv1d_same_padding(in_channel, bottleneck_channel, 1),
            conv1d_same_padding(bottleneck_channel, out_channel, kernel_size[1])
        )

        self.block3 = nn.Sequential(
            conv1d_same_padding(in_channel, bottleneck_channel, 1),
            conv1d_same_padding(bottleneck_channel, out_channel, kernel_size[2])
        )

        self.block4 = nn.Sequential(
            nn.MaxPool1d(kernel_size = 3, padding = 1),
            conv1d_same_padding(in_channel, out_channel, 1)
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)
        x = torch.cat([block1, block2, block3, block4], 1)
        return x

class InceptionLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InceptionLayer, self).__init__()
        
        self.layers = nn.Sequential(
            InceptionBlock(in_channel, out_channel),
            InceptionBlock(in_channel, out_channel),
            InceptionBlock(in_channel, out_channel)
        )

    def forward(self, x):
        return x + self.layers(x)

class InceptionTime(nn.Module):
    def __init__(self, args):
        super(InceptionTime, self).__init__()
        
        in_channel = args['in_channel']
        out_channel = args['out_channel']
        self.layers = nn.Sequential(
            InceptionLayer(in_channel, out_channel), 
            InceptionLayer(in_channel, out_channel),
            InceptionLayer(in_channel, out_channel)
        )

        self.GAP = nn.AdaptiveAvgPool1d(128)
        self.linear = nn.Linear(128, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.layers(x)
        x = self.GAP(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x
        