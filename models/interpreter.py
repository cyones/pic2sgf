import torch as tr
from torch import nn
from torchvision import transforms


class conv_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_block, self).__init__()
        self.convs = nn.Sequential(nn.ELU(),
                                   nn.BatchNorm2d(in_dim),
                                   nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1))
    def forward(self, x):
        return self.convs(x)


class resnet_block(nn.Module):
    def __init__(self, dims):
        super(resnet_block, self).__init__()
        self.conv_path = nn.Sequential(nn.ELU(),
                                       nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1),
                                       nn.ELU(),
                                       nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1))
        
    def forward(self, x):
        return x + self.conv_path(x)

    

class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1),
                                         resnet_block(8), nn.AvgPool2d(2),
                                         resnet_block(8), nn.AvgPool2d(2),
                                         resnet_block(8), nn.AvgPool2d(2),
                                         resnet_block(8), nn.AvgPool2d(2),
                                         resnet_block(8))
        
        self.board_interpreter = nn.Sequential(nn.Conv2d(8, 1, kernel_size=1), nn.Tanh())

    def forward(self, x):
        x = self.conv_blocks(x)
        board = self.board_interpreter(x).squeeze()
        return board
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)