import torch
import torch.nn as nn

"""
Output format: (batch, channels, dim1, dim2, dim3)
Layer   Filter shape        Output shape
input   (32, 1, 32, 32, 32)
conv1   (1, 32, 2, 2, 2)    (32, 32, 16, 16, 16)
conv2   (32, 64, 5, 2, 1)   (32, 64, 12, 12, 12)
pool2   (2, 2, 2)           (32, 64, 6, 6, 6)
fc      (1728)              (32, 576)
fc      (576)               (32, 192)
out                         (192, num_classes)
"""

class CNN(nn.Module):

    """
    Our own convolutional neural net

    Author: Lukas Janik-Jones
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.conv_1 = nn.Conv3d(1, 32, kernel_size=2, stride=2, padding=2)
        self.conv_2 = nn.Conv3d(32, 64, kernel_size=5, stride=2)
        self.pool = nn.MaxPool3d(2)

        self.full_1 = nn.Linear(1728, 576)
        self.full_2 = nn.Linear(576, 192)


        self.out = nn.Linear(192, self.num_classes)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        # conv(1, 1) -> 32 (32,32,32)
        # pool (2)   -> 32 (16,16,16)
        # conv(5, 2) -> 64 (6,6,6)
        # pool (2)   -> 64 (3,3,3)
        # full       -> 1728 to 576
        # full       -> 576 to 192
        # full       -> 192 to num_classes

        # First conv/pool layers
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool(x)

        # Second conv layer
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.drop(x)

        # Flatten input
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.full_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.full_2(x)
        x = self.relu(x)
        x = self.drop(x)
        
        x = self.out(x)

        return x