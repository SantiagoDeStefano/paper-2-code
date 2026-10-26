"""
Resized with its shorter side randomly sampled in [256, 480];

224x224 crop is randomly sampled from an image or its horizontal flip
with the per-pixel mean being subtracted;

Standard color augmentation;

Batch normalization right after right after each 
convolution and before activation;

SGD with a mini-batch size of 256:
- lr: 0.1
- lr * 0.1 step
- trained for 60 * e4
- weight decay: 1e-4
- momentum: 0.9
"""
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Block 1
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.conv3_x = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=128),
        )

        self.conv4_x = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )   

        self.shortcut4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256),
        )

        self.conv5_x = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )

        self.shortcut5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=512),
        )

        self.classification = nn.Sequential(
            # Output of AdaptiveAvgPool2d:
            # (Batch, 1, 1, 512) so we should add torch.flatten()
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(1*1*512, 1000)
        )
    def forward(self, x):
        x = self.conv1(x)

        # [0]  MaxPool
        # [1]  Conv   ┐
        # [2]  BN     │
        # [3]  ReLU   │ block 1's F
        # [4]  Conv   │
        # [5]  BN     ┘
        # [6]  ReLU     block 1's final ReLU
        # [7]  Conv   ┐
        # [8]  BN     │
        # [9]  ReLU   │ block 2's F
        # [10] Conv   │
        # [11] BN     ┘
        # [12] ReLU     block 2's final ReLU
        x = self.conv2_x[0](x)
        x = self.conv2_x[6](self.conv2_x[1:6](x) + x)
        x = self.conv2_x[12](self.conv2_x[7:12](x) + x)

        # conv3_x / conv4_x / conv5_x layout:
        # [0]  Conv (stride 2, C_in -> C)   ┐
        # [1]  BN                           │
        # [2]  ReLU                         │ block 1's F
        # [3]  Conv (C -> C)                │
        # [4]  BN                           ┘
        # [5]  ReLU                           block 1's final ReLU (after + shortcutN(x))
        # [6]  Conv (C -> C)                ┐
        # [7]  BN                           │
        # [8]  ReLU                         │ block 2's F
        # [9]  Conv (C -> C)                │
        # [10] BN                           ┘
        # [11] ReLU                           block 2's final ReLU (after + x)
        x = self.conv3_x[5](self.conv3_x[0:5](x) + self.shortcut3(x))
        x = self.conv3_x[11](self.conv3_x[6:11](x) + x)

        x = self.conv4_x[5](self.conv4_x[0:5](x) + self.shortcut4(x))
        x = self.conv4_x[11](self.conv4_x[6:11](x) + x)

        x = self.conv5_x[5](self.conv5_x[0:5](x) + self.shortcut5(x))
        x = self.conv5_x[11](self.conv5_x[6:11](x) + x)

        x = self.classification(x)
        return x