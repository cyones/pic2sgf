import torch as tr
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_block, self).__init__()
        self.convs = nn.Sequential(nn.ELU(),
                                   nn.BatchNorm2d(in_dim),
                                   nn.Conv2d(in_dim, out_dim, kernel_size=[3,3], padding=[1,1]))
    def forward(self, x):
        return self.convs(x)


class resnet_block(nn.Module):
    def __init__(self, dims):
        super(resnet_block, self).__init__()
        self.conv_path = nn.Sequential(nn.ELU(),
                                       nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=[3,3], padding=[1,1]),
                                       nn.ELU(),
                                       nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=[3,3], padding=[1,1]))
        
    def forward(self, x):
        return x + self.conv_path(x)

    

class Sizer(nn.Module):
    def __init__(self):
        super(Sizer, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), 
                                         resnet_block(8), resnet_block(8), nn.MaxPool2d(2),
                                         conv_block(8, 16),
                                         resnet_block(16), resnet_block(16), nn.MaxPool2d(2),
                                         conv_block(16, 32),
                                         resnet_block(32), resnet_block(32), nn.MaxPool2d(2),
                                         conv_block(32, 64),
                                         resnet_block(64), resnet_block(64), nn.MaxPool2d(2),
                                         resnet_block(64), resnet_block(64), nn.MaxPool2d(2),
                                         resnet_block(64), resnet_block(64), nn.ELU(), nn.BatchNorm2d(64))
        
        self.out_layer = nn.Linear(3072, 2)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(-1, 3072)
        x = self.out_layer(x)
        return x
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
