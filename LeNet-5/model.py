from typing import Any

import torch
import torch.nn as nn

"""
Layer S2 is a sub-sampling layer with 6 feature maps of
size 14x14. Each unit in each feature map is connected to a
2x2 neighborhood in the corresponding feature map in C1.
The four inputs to a unit in S2 are added, then multiplied
by a trainable coeffient, and added to a trainable bias.
The result is passed through a sigmoidal function. The
2x2 receptive fields are non-overlapping, therefore feature
maps in S2 have half the number of rows and column as
feature maps in C1. Layer S2 has 12 trainable parameters
and 5,880 connections.
"""
class layer_s2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Layer C1 is a convolutional layer with 6 feature maps.
        Each unit in each feature map is connected to a 5x5 neighborhood 
        in the input. The size of the feature maps is 28x28
        which prevents connection from the input from falling off
        the boundary. 
        C1 contains 5 x 5 (kernel size) = 25 + 1 (bias) = 26 * 6 (6 feature maps) = 156 trainable parameters, and
        5x5 = 25 (inputs) + 1 (bias) = 26 x 6 x 28 x 28 (6 maps, each is 28x28) = 122,304 connections.
        """
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)


        self.layer_s2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(32, 32)
        )
        self.layer_4 = nn.Sequential(
            nn.Linear(32, 32)
        )
        self.layer_5 = nn.Sequential(
            nn.Linear(32, 32)
        )
        self.layer_6 = nn.Sequential(
            nn.Linear(32, 32)
        )
        self.layer_7 = nn.Sequential(
            nn.Linear(32, 32)
        )
