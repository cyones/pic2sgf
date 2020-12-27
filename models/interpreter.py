import torch as tr
from torch import nn
from .iblock import iblock
from .global_features import GlobalFeatures


def Pooling(in_dim, out_dim):
    return nn.Sequential(nn.BatchNorm2d(in_dim), nn.GELU(),
                         nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.MaxPool2d(2))


class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3),
            nn.GELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 12, kernel_size=3, padding=1),
            iblock(12), iblock(12), iblock(12), iblock(12),
            Pooling(12, 24), 
            iblock(24), iblock(24), iblock(24), iblock(24),
            Pooling(24, 48), 
            iblock(48), iblock(48), iblock(48), iblock(48),
            Pooling(48, 96),
            iblock(96), iblock(96), iblock(96), iblock(96),
            nn.GELU(), nn.BatchNorm2d(96),
            )
        self.displacement = nn.Sequential(
            nn.Conv2d(96, 24, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(24),
            nn.Conv2d(24, 2, kernel_size=1)
        )
        self.position = nn.Sequential(
            nn.Conv2d(96, 24, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(24),
            nn.Conv2d(24, 3, kernel_size=1)
        )
        self.wrong = nn.Sequential(
            nn.Conv2d(96, 24, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(24),
            nn.Conv2d(24, 1, kernel_size=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        position = self.position(x)
        displacement = self.displacement(x)
        wrong = self.wrong(x).view(-1, 1)
        return position, displacement, wrong
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
