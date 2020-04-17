import torch as tr
from torch import nn
from torchvision import transforms


class iblock(nn.Module):
    def __init__(self, dims):
        super(iblock, self).__init__()
        self.conv_path = nn.Sequential(nn.BatchNorm2d(dims), nn.ReLU(),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1),
                                       nn.ReLU(), nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1))
        
    def forward(self, x):
        return x + self.conv_path(x)

class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(3, 8, kernel_size=2, stride=2), 
                                         iblock(8), nn.ReLU(), nn.BatchNorm2d(8),
                                         nn.Conv2d(8, 16, kernel_size=2, stride=2),
                                         iblock(16), nn.ReLU(), nn.BatchNorm2d(16),
                                         nn.Conv2d(16, 32, kernel_size=2, stride=2),
                                         iblock(32), nn.ReLU(), nn.BatchNorm2d(32),
                                         nn.Conv2d(32, 64, kernel_size=2, stride=2),
                                         iblock(64), nn.ReLU(), nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 3, kernel_size=1))

    def forward(self, x):
        x = self.conv_blocks(x)
        return x
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
