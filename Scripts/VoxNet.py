import torch
import torch.nn as nn

class VoxNet(nn.Module):

    """
    VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition
    https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf

    PyTorch implementation of VoxNet architecture.

    Author: Lukas Janik-Jones
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.conv_1 = nn.Conv3d(32, 32, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv3d(32, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool3d(2)
        # I believe the input size should be number of out_channels of last conv3d layer
        # multiplied by the size of the data after pooling: (6,6,6)
        self.full = nn.Linear((6*6*6), 128)
        self.out = nn.Linear(128, self.num_classes)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = self.relu(x)
        
        x = self.pool(x)
        x = self.drop(x)

        # Flatten input
        x = x.view(x.size(0), -1)

        # print(x.size())

        x = self.full(x)
        
        # x = self.full(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.out(x)

        return x