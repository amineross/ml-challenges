import torch
import torch.nn as nn
import torch.nn.functional as F

class ESPCN(nn.Module):
    def __init__(self, inputSize, outputSize, scale):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 3*(scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.couches(x)