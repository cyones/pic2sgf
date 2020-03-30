import torch as tr
from torch import nn
from torchvision import transforms


class conv_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_block, self).__init__()
        self.convs = nn.Sequential(
            nn.ELU(),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=[3,3], padding=[1,1])
        )
    def forward(self, x):
        return self.convs(x)


class resnet_block(nn.Module):
    def __init__(self, dims):
        super(resnet_block, self).__init__()
        self.conv_path = nn.Sequential(conv_block(dims, dims), conv_block(dims, dims))
        
    def forward(self, x):
        return x + self.conv_path(x)
    

class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.pre_cnn = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1),
                                     resnet_block(8), resnet_block(8), nn.MaxPool2d(2),
                                     conv_block(8, 16), resnet_block(16), resnet_block(16), nn.MaxPool2d(2)
                                     )

        self.in_cnn = nn.ModuleList([
                                     nn.Sequential(conv_block(16, 32), resnet_block(32), resnet_block(32)),
                                     nn.Sequential(conv_block(32, 64), resnet_block(64), resnet_block(64))
                                     ])

        self.bottom = nn.Sequential(resnet_block(64), resnet_block(64))

        self.out_cnn = nn.ModuleList([
                                      nn.Sequential(resnet_block(64), resnet_block(64), conv_block(64, 32)),
                                      nn.Sequential(resnet_block(32), resnet_block(32), conv_block(32, 16)),
        ])
            
        self.last_cnn = nn.Sequential(conv_block(16, 3), nn.Sigmoid())

    def forward(self, x):
        x = self.pre_cnn(x)

        mid = []
        for i in range(len(self.in_cnn)):
            x = self.in_cnn[i](x)
            mid.append(x)
            x = self.downscale(x)
        
        x = self.bottom(x)

        for i in range(len(self.out_cnn)):
            x = self.out_cnn[i]( self.upscale(x) + mid.pop() )
        x = self.last_cnn(x)
        return x

    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)