import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return input + x * self.res_scale

class EDSRLITE(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(8)]
        )

        self.aggregation = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1)
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64*scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    
    def forward(self,x):
        x = self.feature_extraction(x)
        residual = x
        x = self.res_blocks(x)
        x = self.aggregation(x)
        x += residual
        x = self.upsampling(x)
        return x