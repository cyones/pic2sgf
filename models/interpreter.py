import torch as tr
from torch import nn
from torchvision import transforms


class iblock(nn.Module):
    def __init__(self, dims):
        super(iblock, self).__init__()
        self.conv_path = nn.Sequential(nn.BatchNorm2d(dims), nn.ELU(),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1),
                                       nn.ELU(), nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1))
        
    def forward(self, x):
        return x + self.conv_path(x)

def Pooling(in_dim, out_dim):
    return nn.Sequential(nn.BatchNorm2d(in_dim), nn.ELU(),
                         nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2))

class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=2, stride=2),
            iblock(10), Pooling(10, 20),
            iblock(20), Pooling(20, 40),
            iblock(40), Pooling(40, 80),
            iblock(80),
            nn.ELU(), nn.BatchNorm2d(80), nn.Dropout2d(0.2),
            nn.Conv2d(80, 20, kernel_size=1),
            nn.ELU(), nn.BatchNorm2d(20)
            )
        self.position = nn.Conv2d(20, 3, kernel_size=1)
        self.wrong = nn.Sequential(
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(), nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        position = self.position(x)
        wrong = self.wrong(x).squeeze(1).squeeze(1)
        return position, wrong
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
