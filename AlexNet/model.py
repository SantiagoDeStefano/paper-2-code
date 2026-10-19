import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        """
        The net contains eight layers with weights; 
        the first five are convolutional and the remaining
        three are fully-connected. The first convolutional 
        layer filters the 224x224x3 input image with 96 kernels
        of size 11x11x3 with a stride of 4 pixels.
        Params: (11 * 11 * 3 + 1) * 96 = 34,944;
        The input image is 227x227, 224x224 in the paper is wrong
        (227 - 11)/4 + 1 = 55 x 55 x 96
        Connections: 55 * 55 * 34,944 = 105,705,600
        """
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # (55 - 3)/2 + 1 = 27 x 27 x 96
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        """
        The second convolutional layer takes as input the
        (response-normalized and pooled) output of the first 
        convolutional layer and filters it with 256 kernels of size 5x5x48.
        48 for one of the two GPUs so for one GPU training it is 48*2 = 96;
        Params: (5 * 5 * 96 + 1) * 256 = 614,656;
        (27 + 2*2 - 1*(5 - 1) - 1)/1 + 1 = 27 x 27 x 256
        Connections: 27 * 27 * 614,656 = 448,084,224
        """
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # (27 - 3)/2 + 1 = 13 x 13 x (192 * 2) = 13x13x384
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        """
        The third convolutional layer has 384 kernels of 
        size 3 x 3 x 256 connected to the (normalized, pooled)
        outputs of the second convolutional layer.
        Params: (3 * 3 * 256 + 1) * 384 = 885,120;
        (13 + 2*1 - 1*(3 - 1) - 1)/1 + 1 = 13 x 13 x 384
        Connections: 13 * 13 * 885,120 = 149,585,280
        """
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # (13 + 1*2 - 3)/1  + 1 = 13 x 13 x 384
        )

        """
        The fourth convolutional layer has 384 kernels of size 3x3x384
        Params: (3 * 3 * 384 + 1) * 384 = 1,327,488;
        (13 + 2*1 - 1*(3-1) -1)/1 + 1 = 13 x 13 x 384
        Connections: 13 * 13 * 1,327,488 = 224,345,472
        """
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU()
        )

        """
        The fifth convolutional layer has 256 kernels of size 3x3x384. 
        The fully-connected layers have 4096 neurons each.
        Params: (3 * 3 * 384 + 1) * 256 = 884,992
        (13 + 2*1 - 1*(3-1) -1)/1 + 1 = 13 x 13 x 256
        Connections: 13 * 13 * 884,992 = 149,563,648
        """
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (13 - 3)/2 + 1 = 6x6x256
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
