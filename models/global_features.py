import torch as tr
from torch import nn


class GlobalFeatures(nn.Module):
    def __init__(self, in_dim, dim):
        super(GlobalFeatures, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(dim),
            nn.AdaptiveAvgPool2d(1)
            )
        self.mixer = nn.Sequential(
            nn.Conv2d(in_dim + dim, in_dim, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(in_dim)
            )

    def forward(self, x):
        gf = self.input(x)
        gf = nn.functional.interpolate(gf, size=(x.shape[2], x.shape[3]))
        x = tr.cat([x, gf], dim=1)
        return self.mixer(x)