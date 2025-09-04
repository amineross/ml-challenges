import torch
import torch.nn as nn

class EDSRLITE(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        res_block = []

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.PReLU()
        )

        for _ in range(16):
            res_block.append(nn.Conv2d(64, 64, 3, padding=1))
            res_block.append(nn.PReLU())
            res_block.append(nn.Conv2d(64, 64, 3, padding=1))
        self.mapping = nn.Sequential(*res_block)

        self.aggregation = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU()
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64*scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    
    def forward(self,x):
        x = self.feature_extraction(x)
        x = self.mapping(x)
        x = self.aggregation(x)
        x = self.upsampling(x)
        return x