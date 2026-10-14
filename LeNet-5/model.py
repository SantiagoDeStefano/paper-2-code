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
(6 coefficient and 6 bias for each features map = 6)
and 2x2 (2x2 kernel) = 4 + 1 (bias) = 5 * (14 * 14) (output after kernel)
= 980 * 6 (features map) = 5,880 connections.
"""
class layer_s2(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff = nn.Parameter(torch.ones(6))
        self.bias = nn.Parameter(torch.zeros(6))
    def forward(self, x):
        x = torch.nn.AvgPool2d(kernel_size=2, divisor_override=1)(x)
        x = x * self.coeff.view(1, 6, 1, 1)
        x = x + self.bias.view(1, 6, 1, 1)
        return torch.sigmoid(x)


"""
Layer C3 is a convolutional layer with 16 feature maps.
Each unit in each feature map is connected to several 5x5
neighborhoods at identical locations in a subset of S2's
feature maps. Table I shows the set of S2 feature maps
combined by each C3 feature map. Why not connect every S2 feature 
map to every C3 feature map? The reason is twofold. First, 
a non-complete connection scheme keeps the number of connections
within reasonable bounds. More importantly, it forces a break
of symmetry in the network. Different feature maps are forced
to extract different (hopefully complementary) features because
they get different sets of inputs. The rationale behind the 
connection scheme in table I is the following. The first six 
C3 feature maps take inputs from every contiguous subsets 
of three feature maps in S2. The next six take input from 
every contiguous subset of four. The next three take input 
from some discontinuous subsets of four. Finally the last 
one takes input from all S2 feature maps. Layer C3 has 

5 * 5 (kernel) = 25 * 3 (3 inputs at a time from layer S2)
= 75 + 1 (bias) = 76 * 6 (six 3 inputs of layer S2) = 456

5 * 5 (kernel) = 25 * 4 (4 inputs at a time from layer S2)
= 100 + 1 (bias) = 101 * 9 (nine 4 inputs of layer S2) = 909

5 * 5 (kernel) = 25 * 6 (6 inputs at a time from layer S2)
= 150 + 1 (bias) = 151 * 1 (one 6 inputs of layer S2) = 151

456 + 909 + 151 = 1,516 trainable parameters and 


C3 output size: 
H_out = floor( (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 )
      = floor( (14   + 2 * 0       - 1        * (5           - 1) - 1) / 1      + 1 )
      = floor(10)
      = 10
10 * 10 = 100 (units) * 1516 (params) = 151,600 connections.
"""
class layer_c3(nn.Module):
    """
    Table 1. Each column indicates which feature map 
    in S2 are combined by the units in a particular 
    feature map of C3 
    """
    TABLE = [
        [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 4, 5], [0, 1, 5],
        [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 5],
        [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
        [0, 1, 2, 3, 4, 5],
    ]
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        for subset in self.TABLE:
            num_inputs = len(subset)
            conv = nn.Conv2d(in_channels=num_inputs, out_channels=1, kernel_size=5)
            self.convs.append(conv)

    def forward(self, x):
        outs = []
        for conv, subset in zip(self.convs, self.TABLE):
            picked = x[:, subset]
            out = conv(picked)
            outs.append(out)

        x = torch.cat(outs, dim=1)
        return torch.sigmoid(x)

"""
Layer S4 is a sub-sampling layer with 16 feature maps of
size 5x5. Each unit in each feature map is connected to a
2x2 neighborhood in the corresponding feature map in C3,
in a similar way as C1 and S2. Layer S4 has
16 (features map) * ( 1 (coeff) + 1 (bias) ) = 32 trainable
parameters and 16 (features map) * 5 * 5 = 400
* ( 2 * 2 (kernel) + 1 (bias) ) = 2,000 connections.
"""
class layer_s4(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff = nn.Parameter(torch.ones(16))
        self.bias = nn.Parameter(torch.zeros(16))
    def forward(self, x):
        x = torch.nn.AvgPool2d(kernel_size=2, divisor_override=1)(x)
        x = x * self.coeff.view(1, 16, 1, 1)
        x = x + self.bias.view(1, 16, 1, 1)
        return torch.sigmoid(x)

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
        self.layer_c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), 
            nn.Sigmoid()
        )

        self.layer_s2 = nn.Sequential(
            layer_s2()
        )
        self.layer_c3 = nn.Sequential(
            layer_c3()
        )
        self.layer_s4 = nn.Sequential(
            layer_s4()
        )

        """
        Layer C5 is a convolutional layer with 120 feature maps.
        Each unit is connected to a 5x5 neighborhood on all 16
        of S4's feature maps. Here, because the size of S4 is also
        5x5, the size of C5's feature maps is 1x1: this amounts
        to a full connection between S4 and C5. C5 is labeled
        as a convolutional layer, instead of a fully-connected layer,
        because if LeNet-5 input were made bigger with everything
        else kept constant, the feature map dimension would be
        larger than 1x1. This process of dynamically increasing the
        size of a convolutional network is described in the section
        Section VII. Layer C5 has 16 (features map of S4) * 5 * 5 (kernel size)
        = 400 + 1 (bias) = 401 * 120 = 48,120 trainable connections.
        """
        self.layer_c5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Sigmoid()
        )
        
        """
        Layer F6, contains 84 units (the reason for this number
        comes from the design of the output layer, explained below) 
        and is fully connected to C5. It has 84 * (120 + 1) = 10,164 trainable parameters.
        """
        self.layer_f6 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid()
        )

        self.out = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.layer_c1(x)
        x = self.layer_s2(x)
        x = self.layer_c3(x)
        x = self.layer_s4(x)
        x = self.layer_c5(x)
        x = x.view(x.size(0), -1)
        x = self.layer_f6(x)
        x = self.out(x)
        return x

if __name__ == "__main__":
    model = LeNet5()
    out = model(torch.randn(1, 1, 32, 32))
    print(out.shape)
    print(sum(p.numel() for p in model.parameters()))