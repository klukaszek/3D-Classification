import torch
import torch.nn as nn

class VoxNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.conv_1 = nn.Conv3d(1, 32, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv3d(32, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool3d(2)
        self.full = nn.Linear(____, 128)
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
        
        x = self.full(x)
        x = self.relu(x)
        x = self.out(x)

        return x