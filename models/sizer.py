import torch as tr
from torch import nn
from .iblock import iblock


class Sizer(nn.Module):
    def __init__(self):
        super(Sizer, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=3),
                                         iblock(8), nn.GELU(), nn.BatchNorm2d(8),
                                         nn.Conv2d(8, 16, kernel_size=2, stride=2),
                                         iblock(16), nn.GELU(), nn.BatchNorm2d(16),
                                         nn.Conv2d(16, 32, kernel_size=2, stride=2),
                                         iblock(32), nn.GELU(), nn.BatchNorm2d(32),
                                         nn.Conv2d(32, 64, kernel_size=2, stride=2),
                                         iblock(64), nn.GELU(), nn.BatchNorm2d(64),
                                         nn.AdaptiveAvgPool2d(1))
        
        self.linear = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.BatchNorm1d(16),
                                    nn.Dropout(0.2), nn.Linear(16, 4), nn.Sigmoid())
                                         

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.linear(x.squeeze(3).squeeze(2))
        return x
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
