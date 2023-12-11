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

class CNN(nn.Module):

    """
    3D model classification using convolutional neural network
    https://cs229.stanford.edu/proj2015/146_report.pdf

    PyTorch implementation of 3DCNN architecture by JunYoung Gwak.

    Authors: Kyle Lukaszek and Lukas Janik-Jones
    """

    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # First conv/pool layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Second conv/pool layers
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Third conv/pool layers
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Fourth conv/pool layers
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Fully connected layer 1
        self.fc5 = nn.Linear(256 * 2 * 2 * 2, 1024)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)

        # Fully connected layer 2
        self.fc6 = nn.Linear(1024, 1024)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.5)

        # Output layer
        self.softmax7 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 2 * 2)  # Flatten the tensor
        x = self.dropout5(self.relu5(self.fc5(x)))
        x = self.dropout6(self.relu6(self.fc6(x)))
        x = self.softmax7(x)
        return x