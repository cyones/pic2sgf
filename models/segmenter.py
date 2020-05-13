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

class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.downscale = nn.ModuleList([Pooling(8, 16), Pooling(16, 32), Pooling(32, 64)])
        self.upscale = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        
        self.pre_cnn = nn.Sequential(nn.Conv2d(3, 8, kernel_size=2, stride=2))

        self.in_cnn = nn.ModuleList([nn.Sequential(iblock( 8), iblock( 8)),
                                     nn.Sequential(iblock(16), iblock(16)),
                                     nn.Sequential(iblock(32), iblock(32))
                                     ])

        self.bottom = nn.Sequential(iblock(64), iblock(64),
                                    nn.BatchNorm2d(64), nn.ELU(),
                                    nn.Conv2d(64, 32, kernel_size=1))

        self.out_cnn = nn.ModuleList([nn.Sequential(iblock(32), iblock(32),
                                                    nn.ELU(), nn.BatchNorm2d(32),
                                                    nn.Conv2d(32, 16, kernel_size=1)),
                                      nn.Sequential(iblock(16), iblock(16),
                                                    nn.ELU(), nn.BatchNorm2d(16),
                                                    nn.Conv2d(16, 8, kernel_size=1)),
                                      nn.Sequential(iblock(8), iblock(8),
                                                    nn.ELU(), nn.BatchNorm2d(8))
        ])
            
        self.last_cnn = nn.Sequential(nn.Conv2d(8, 3, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.pre_cnn(x)
        mid = []
        for i in range(len(self.in_cnn)):
            x = self.in_cnn[i](x)
            mid.append(x)
            x = self.downscale[i](x)
        
        x = self.bottom(x)

        for i in range(len(self.out_cnn)):
            x = self.out_cnn[i]( self.upscale[i](x) + mid.pop() )
        x = self.last_cnn(x)
        return x

    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
