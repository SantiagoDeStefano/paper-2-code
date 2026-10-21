import torch
import torch.nn as nn

class VGG_D(nn.Module):
    def __init__(self):
        super().__init__()

        """
        Table 1: ConvNet configurations, column E, second row
        Params:
        - 3 to 64: (3 * 3 * 3 + 1) * 64 = 1,792
        - 64 to 64: (3 * 3 * 64 + 1) * 64 = 36,928
        Connections:
        - 3 to 64: 224 * 224 * 1,792 = 89,915,392
        - 64 to 64: 224 * 224 * 36,928 = 1,852,899,328
        """
        self.layer_1 = nn.Sequential(
            # (224 + 2*1 - 1*(3-1) - 1)/1 + 1 = 224 x 224 x 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # (224 + 2*1 - 1*(3-1) - 1)/1 + 1 = 224 x 224 x 64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # (224 + 2*0 - 1*(2-1) - 1)/2 + 1 = 112 x 112 x 64
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        Table 1: ConvNet configurations, column E, third row
        Params:
        - 64 to 128: (3 * 3 * 64 + 1) * 128 = 73,856
        - 128 to 128: (3 * 3 * 128 + 1) * 128 = 147,584
        Connections:
        - 64 to 128: 112 * 112 * 73,856 = 926,449,664
        - 128 to 128: 112 * 112 * 147,584 = 1,851,293,696
        """
        self.layer_2 = nn.Sequential(
            # (112 + 2*1 - 1*(3-1) - 1)/1 + 1 = 112 x 112 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # (112 + 2*1 - 1*(3-1) - 1)/1 + 1 = 112 x 112 x 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # (112 + 2*0 - 1*(2-1) - 1)/2 + 1 = 56 x 56 x 128
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        Table 1: ConvNet configurations, column E, fourth row
        Params:
        - 128 to 256: (3 * 3 * 128 + 1) * 256 = 295,168
        - 256 to 256: (3 * 3 * 256 + 1) * 256 = 590,080
        - 256 to 256: (3 * 3 * 256 + 1) * 256 = 590,080
        Connections:
        - 128 to 256: 56 * 56 * 295,168 = 925,646,848
        - 256 to 256: 56 * 56 * 590,080 = 1,850,490,880
        - 256 to 256: 56 * 56 * 590,080 = 1,850,490,880
        """
        self.layer_3 = nn.Sequential(
            # (56 + 2*1 - 1*(3-1) - 1)/1 + 1 = 56 x 56 x 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (56 + 2*1 - 1*(3-1) - 1)/1 + 1 = 56 x 56 x 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (56 + 2*1 - 1*(3-1) - 1)/1 + 1 = 56 x 56 x 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (56 + 2*0 - 1*(2-1) - 1)/2 + 1 = 28 x 28 x 256
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        Table 1: ConvNet configurations, column E, fifth row
        Params:
        - 256 to 512: (3 * 3 * 256 + 1) * 512 = 1,180,160
        - 512 to 512: (3 * 3 * 512 + 1) * 512 = 2,359,808
        - 512 to 512: (3 * 3 * 512 + 1) * 512 = 2,359,808
        Connections:
        - 256 to 512: 28 * 28 * 1,180,160 = 925,245,440
        - 512 to 512: 28 * 28 * 2,359,808 = 1,850,089,472
        - 512 to 512: 28 * 28 * 2,359,808 = 1,850,089,472
        """
        self.layer_4 = nn.Sequential(
            # (28 + 2*1 - 1*(3-1) - 1)/1 + 1 = 28 x 28 x 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # (28 + 2*1 - 1*(3-1) - 1)/1 + 1 = 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # (28 + 2*1 - 1*(3-1) - 1)/1 + 1 = 28 x 28 x 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # (28 + 2*0 - 1*(2-1) - 1)/2 + 1 = 14 x 14 x 512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """
        Table 1: ConvNet configurations, column E, sixth row
        Params:
        - 512 to 512: (3 * 3 * 512 + 1) * 512 = 2,359,808
        - 512 to 512: (3 * 3 * 512 + 1) * 512 = 2,359,808
        - 512 to 512: (3 * 3 * 512 + 1) * 512 = 2,359,808
        Connections:
        - 512 to 512: 14 * 14 * 2,359,808 = 462,522,368
        - 512 to 512: 14 * 14 * 2,359,808 = 462,522,368
        - 512 to 512: 14 * 14 * 2,359,808 = 462,522,368
        """
        self.layer_5 = nn.Sequential(
            # (14 + 2*1 - 1*(3-1) - 1)/1 + 1 = 14 x 14 x 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # (14 + 2*1 - 1*(3-1) - 1)/1 + 1 = 14 x 14 x 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # (14 + 2*1 - 1*(3-1) - 1)/1 + 1 = 14 x 14 x 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # (14 + 2*0 - 1*(2-1) - 1)/2 + 1 = 7 x 7 x 512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # It should be 1000 but for imagenette2 it's only 10
            nn.Linear(4096, 10)
            # We use nn.CrossEntropyLoss; in which expects raw logits
            # Apply softmax only during inference
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        # As nn.Conv2d returns (Batch, C_out, H_out, W_out)
        # But FC (Linear) expects (Batch, ...)
        # So we need torch.flatten(x, 1) to make it became (Batch, C_in*H_out*W_out)
        x = torch.flatten(x, 1)
        x = self.layer_classifier(x)
        return x

# Lame ahh model because the full model make my laptop became a smoke bomb
class VGG_D_LameAhh(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(224*224*3, 10)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return x