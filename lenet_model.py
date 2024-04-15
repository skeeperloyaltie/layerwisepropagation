import torch
import torch.nn as nn
import torch.nn.functional as F
from lrp import *

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = LRPConv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = LRPConv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.last_conv_layer = self.conv2

        self.fc1 = LRPLinear(16 * 4 * 4, 120)
        self.fc2 = LRPLinear(120, 84)
        self.fc3 = LRPLinear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Optionally, store the output for visualization
        self.last_conv_output = x.detach()

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

