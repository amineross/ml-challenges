import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ESPCN(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
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
    
    def pnsr(self, y, y_hat, n, m, d: float = 255.0):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.detach().cpu().numpy()
        sse = ((y - y_hat) ** 2).sum()
        if sse == 0.0:
            return float('inf')
        return 10.0 * np.log10((n * m * 3 * (d ** 2)) / sse)