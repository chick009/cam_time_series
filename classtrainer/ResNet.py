
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

class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNetBlock, self).__init__()
        
        # Initialize all variables
        channels = [in_channel, out_channel, out_channel, out_channel]
        kernel_size = [8, 5, 3]

        # Initialize the blocks
        blocks = [conv1d_same_padding(channels[i], channels[i + 1], kernel_size[i]) for i in range(len(kernel_size))]
        self.blocks = nn.Sequential(*blocks)
        # Initialize functional blocks
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.blocks(x)
        x = self.bn(x)
        x = self.relu(x)

        return x



class ResNetBaseline(nn.Module):
    def __init__(self, args):
        super(ResNetBaseline, self).__init__()
        in_channel = args['in_channel']
        num_class = args['num_class']

        self.block = nn.Sequential(*[
            ResNetBlock(100, 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 128)
        ])

        self.GAP = nn.AdaptiveAvgPool1d(128)
        self.linear = nn.Linear(128, num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.block(x)
        x = self.GAP(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x