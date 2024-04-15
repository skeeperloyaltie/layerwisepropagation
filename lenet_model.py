import torch
import torch.nn as nn
import torch.nn.functional as F
from lrp import *

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = LRPConv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = LRPConv2d(6, 16, kernel_size=5, stride=1, padding=0) 
        self.fc1 = LRPLinear(16 * 4 * 4, 120)
        self.fc2 = LRPLinear(120, 84)
        self.fc3 = LRPLinear(84, 10)

        # Store the last convolutional layer separately if needed for specific processing
        self.last_conv_layer = self.conv2

        # Optional: Create a list of all layers for easier access during LRP
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            print("After conv1:", x.shape)  # Debugging output
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            print("After conv2 (last_conv_layer):", x.shape)  # Critical to check the last conv layer output
            self.last_conv_output = x.detach()
            # Continue through FC layers
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def get_layers(self):
        """ Return the layers for systematic access during LRP. """
        return [self.fc3, self.fc2, self.fc1, self.conv2, self.conv1]

