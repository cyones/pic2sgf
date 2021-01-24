import torch as tr
from torch import nn

class iblock(nn.Module):
    def __init__(self, dims):
        super(iblock, self).__init__()
        self.conv_path = nn.Sequential(nn.BatchNorm2d(dims), nn.GELU(),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1),
                                       nn.GELU(), nn.BatchNorm2d(dims),
                                       nn.Conv2d(dims, dims, kernel_size=3, padding=1))
        
    def forward(self, x):
        return x + self.conv_path(x)


def Pooling(in_dim, out_dim):
    return nn.Sequential(nn.BatchNorm2d(in_dim), nn.GELU(),
                         nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2))


class Segmenter(nn.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.downscale = nn.ModuleList([Pooling(10, 20), Pooling(20, 40), Pooling(40, 80), Pooling(80, 80)])
        self.upscale = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)])
        
        self.pre_cnn = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=2, stride=2),
            nn.BatchNorm2d(12), nn.GELU(),
            nn.Conv2d(12, 10, kernel_size=1),
            )

        self.in_cnn = nn.ModuleList([nn.Sequential(iblock(10), iblock(10), iblock(10)),
                                     nn.Sequential(iblock(20), iblock(20), iblock(20)),
                                     nn.Sequential(iblock(40), iblock(40), iblock(40)),
                                     nn.Sequential(iblock(80), iblock(80), iblock(80))
                                     ])

        self.bottom = nn.Sequential(iblock(80), iblock(80),
                                    nn.BatchNorm2d(80), nn.GELU(),
                                    nn.Conv2d(80, 80, kernel_size=1))

        self.out_cnn = nn.ModuleList([nn.Sequential(iblock(80), iblock(80),
                                                    nn.GELU(), nn.BatchNorm2d(80),
                                                    nn.Conv2d(80, 40, kernel_size=1)),
                                      nn.Sequential(iblock(40), iblock(40),
                                                    nn.GELU(), nn.BatchNorm2d(40),
                                                    nn.Conv2d(40, 20, kernel_size=1)),
                                      nn.Sequential(iblock(20), iblock(20),
                                                    nn.GELU(), nn.BatchNorm2d(20),
                                                    nn.Conv2d(20, 10, kernel_size=1)),
                                      nn.Sequential(iblock(10), iblock(10),
                                                    nn.GELU(), nn.BatchNorm2d(10))
        ])
            
        self.last_cnn = nn.Sequential(nn.Conv2d(10, 3, kernel_size=1), nn.Sigmoid())

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
