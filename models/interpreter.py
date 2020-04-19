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

def Pooling(in_dim, out_dim):
    return nn.Sequential(nn.BatchNorm2d(in_dim), nn.ReLU(),
                         nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2))

class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2),
            iblock(8), iblock(8), Pooling(8, 16),
            iblock(16), iblock(16), Pooling(16, 32),
            iblock(32), iblock(32), Pooling(32, 64),
            iblock(64), iblock(64),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.5), 
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout2d(0.5), 
            nn.Conv2d(16, 3, kernel_size=1),
            )

    def forward(self, x):
        x = self.conv_blocks(x)
        return x
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
