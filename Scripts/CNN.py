import torch
import torch.nn as nn

"""
Layer Filter shape Output shape
input (32, 1, 32, 32, 32)
conv1 (32, 1, 3, 3, 3) (32, 32, 32, 32, 32)
pool1 (2, 2, 2) (32, 32, 16, 16, 16)
conv2 (64, 32, 3, 3, 3) (32, 64, 16, 16, 16)
pool2 (2, 2, 2) (32, 64, 8, 8, 8)
conv3 (128, 64, 3, 3, 3) (32, 128, 8, 8, 8)
pool3 (2, 2, 2) (32, 128, 4, 4, 4)
conv4 (256, 128, 3, 3, 3) (32, 256, 4, 4, 4)
pool4 (2, 2, 2) (32, 256, 2, 2, 2)
fc5 (1024) (32, 1024)
fc6 (1024) (32, 1024)
softmax7 (32, 18)
"""

class GCNN(nn.Module):

    """
    3D model classification using convolutional neural network
    https://cs229.stanford.edu/proj2015/146_report.pdf

    PyTorch implementation of VoxNet architecture.

    Author: Lukas Janik-Jones
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.conv_1 = nn.Conv3d(1, 32, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv3d(32, 64, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv3d(64, 128, kernel_size=1, stride=1)
        self.conv_4 = nn.Conv3d(128, 128, kernel_size=1, stride=1)
        self.pool = nn.MaxPool3d(2)

        self.full = nn.Linear(1024, 1024)
        self.soft = nn.Softmax(1)
        self.out = nn.Linear(1024, self.num_classes)

        self.relu = nn.ReLU()
        self.drop_1 = nn.Dropout(p=0.2)
        self.drop_2 = nn.Dropout(p=0.4)

    def forward(self, x):
        # First conv/pool layers
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.drop_1(x)
        x = self.pool(x)

        # Second conv/pool layers
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.drop_1(x)
        x = self.pool(x)

        # Third conv/pool layers
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.drop_1(x)
        x = self.pool(x)

        # Fourth conv/pool layers
        x = self.conv_4(x)
        x = self.relu(x)
        x = self.drop_1(x)
        x = self.pool(x)

        # Flatten input
        x = x.view(x.size(0), -1)

        # First fully connected layer
        x = self.full(x)
        x = self.relu(x)
        x = self.drop_1(x)
    
        # Second fully connected layer
        x = self.full(x)
        x = self.relu(x)
        x = self.drop_1(x)
        
        x = self.soft(x)
        
        x = self.out(x)

        return x