import torch as tr
from torch import nn
from .iblock import iblock


class Sizer(nn.Module):
    def __init__(self):
        super(Sizer, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(3, 10, kernel_size=2, stride=2), # 96
                                         iblock(10), nn.GELU(), nn.BatchNorm2d(10),
                                         nn.Conv2d(10, 20, kernel_size=2, stride=2), # 48
                                         iblock(20), nn.GELU(), nn.BatchNorm2d(20),
                                         nn.Conv2d(20, 40, kernel_size=2, stride=2), # 24
                                         iblock(40), nn.GELU(), nn.BatchNorm2d(40),
                                         nn.Conv2d(40, 80, kernel_size=2, stride=2), # 12
                                         iblock(80), nn.GELU(), nn.BatchNorm2d(80),
                                         nn.Conv2d(80, 80, kernel_size=2, stride=2), # 6
                                         iblock(80), nn.GELU(), nn.BatchNorm2d(80),
                                         nn.AdaptiveAvgPool2d(1))
        
        self.linear = nn.Sequential(nn.Linear(80, 20), nn.GELU(), nn.BatchNorm1d(20),
                                    nn.Dropout(0.25), nn.Linear(20, 4), nn.Sigmoid())
                                         

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.linear(x.squeeze(3).squeeze(2))
        return x
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
