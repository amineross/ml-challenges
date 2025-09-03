import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FSRCNN(nn.Module):
    def __init__(self, d=56, s=12, scale=4):
        super().__init__()
        mapping_layers = []

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, d, 5, padding=2),
            nn.PReLU()
        )

        self.reducing = nn.Sequential(nn.Conv2d(d, s, 1, padding=0),
            nn.PReLU()
        )

        for _ in range(4):
            mapping_layers.append(nn.Conv2d(s, s, 3, padding=1))
            mapping_layers.append(nn.PReLU())
        self.mapping = nn.Sequential(*mapping_layers)
        

        self.expanding = nn.Sequential(nn.Conv2d(s, d, 1, padding=0),
            nn.PReLU()
        )

        # Fix the upsampling layer to match exact output dimensions
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(d, 3, 9, stride=scale, padding=4, output_padding=3),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.reducing(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.upsampling(x)
        return x