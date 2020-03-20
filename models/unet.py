import torch as tr
from torch import nn
from torchvision import transforms

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
    

class UNET(nn.Module):
    def __init__(self, prelayers, unet_levels):
        super(UNET, self).__init__()
        self.downscale = nn.AvgPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.pre_cnn = nn.ModuleList()
        self.in_cnn = nn.ModuleList()
        self.out_cnn = nn.ModuleList()
        
        self.pre_cnn.append(nn.Conv2d(3, 8, kernel_size=[3,3], padding=[1,1]))
        for i in range(prelayers):
            self.pre_cnn.append(nn.Sequential(resnet_block(8), nn.MaxPool2d(2)))

        for i in range(unet_levels):
            self.in_cnn.append(resnet_block(8))

        self.bottom_cnn = resnet_block(8)

        for i in range(unet_levels):
            self.out_cnn.append(nn.Sequential(resnet_block(16),
                                              nn.Conv2d(16, 8, kernel_size=1)))
            
        self.last_cnn = nn.Sequential(nn.ELU(),
                                      nn.BatchNorm2d(8),
                                      nn.Conv2d(8, 1, kernel_size=[3,3], padding=[1,1]),
                                      nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.pre_cnn)):
            x = self.pre_cnn[i](x)
        mid = []
        for i in range(len(self.in_cnn)):
            x = self.in_cnn[i](x)
            mid.append(x)
            x = self.downscale(x)
        x = self.bottom_cnn(x)
        for i in range(len(self.out_cnn)):
            x = self.out_cnn[i]( tr.cat([self.upscale(x), mid.pop()], dim=1) )
        x = self.last_cnn(x)
        return x
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))

    def save(self, fname):
        tr.save(self.state_dict(), fname)
