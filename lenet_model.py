import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer: 1 input channel (grayscale image), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16*4*4 comes from the dimension reduction after conv and pooling layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for MNIST (0-9 digits)

    def forward(self, x):
        # Apply first convolution, followed by ReLU non-linearity and 2x2 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # Apply second convolution, followed by ReLU non-linearity and 2x2 max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)
        # Apply three fully connected layers with ReLU non-linearity for the first two
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No non-linearity for the output layer
        return x
