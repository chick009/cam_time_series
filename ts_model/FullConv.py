
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

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Conv_Block, self).__init__()

        self.conv = conv1d_same_padding(in_channel, out_channel, kernel_size)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class FCNModel(nn.Module):
    
    def __init__(self, args):
        super(FCNModel, self).__init__()
        

        '''
        kernels = [8, 5, 3]
        in_list = [self.in_channels, 128, 256]
        out_list = [128, 256, 128]

        self.block1 = nn.Sequential(
            conv1d_same_padding(in_channels= in_list[0], out_channels= out_list[0], kernel_size = kernels[0]),
            nn.BatchNorm1d(out_list[0])
            nn.ReLU())
        
        self.block2 = nn.Sequential(
            conv1d_same_padding(in_channels= in_list[1], out_channels=out_list[1], kernel_size = kernels[1]),
            nn.BatchNorm1d(out_list[1])
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            conv1d_same_padding(in_channels= in_list[2], out_channels=out_list[2], kernel_size= kernels[2]),
            nn.BatchNorm1d(out_list[2])
            nn.ReLU()
        )
        '''
        
        kernel_size_s = [8, 5, 3]
        channels = [args['in_channel'], 128, 256, 128]
        num_class = args['num_class']

        self.conv_layers = nn.Sequential(*[
			Conv_Block(in_channel=channels[i], out_channel=channels[i + 1], kernel_size= kernel_size_s[i]) for i in range(len(kernel_size_s))
		])

        self.GAP = nn.AdaptiveAvgPool1d(128)
        self.linear = nn.Linear(128, num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Assuming the inputs are in shape B x S x D

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        '''
        x = self.conv_layers(x)
        x = self.GAP(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x

        
        

        