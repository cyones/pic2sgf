import torch as tr
import numpy as np
from torch import nn
from torchvision import transforms

from sklearn.cluster import KMeans

from scipy.sparse import coo_matrix



class VertexDetector():
    def __init__(self):
        self.unet_model = UNET(prelayers=2, unet_levels=4).load("models/vertex_segmenter.pmt")
        self.to_tensor = transforms.ToTensor()
        self.kmeans = KMeans(n_clusters=4,
                             n_init=5,
                             precompute_distances=True,
                             algorithm='elkan')

    def __call__(self, image):
        x = self.to_tensor(image).unsqueeze(0)
        segmentation = self.unet_model(x).numpy()

        segmentation[segmentation < 0.01] = 0.0
        segmentation = coo_matrix( segmentation )
        x = np.stack([segmentation.col, segmentation.row]).transpose()
        w = segmentation.data

        km_results = self.kmeans.fit(x, sample_weight=w)
        vertexs = km_results.cluster_centers_ * 4
        vertexs = self.order_vertexs(vertexs)
        return vertexs, segmentation.toarray()

    def order_vertexs(self, vertexs):
        last_prod = 0
        for i in range(len(vertexs)):
            prev = vertexs[(i-1)%4] - vertexs[i]
            post = vertexs[(i+1)%4] - vertexs[i]
            cross_prod = np.cross(post, prev)
            if cross_prod * last_prod < 0:
                vertexs[i], vertexs[(i+1)%4] = vertexs[(i+1)%4].copy(), vertexs[i].copy()
                i += 1
            else:
                last_prod = cross_prod
        return vertexs


class UNET(nn.Module):
    def __init__(self, prelayers, unet_levels):
        super(UNET, self).__init__()
        self.downscale = nn.AvgPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.pre_cnn = nn.ModuleList()
        self.in_cnn = nn.ModuleList()
        self.out_cnn = nn.ModuleList()
        
        for i in range(prelayers):
            self.pre_cnn.append(nn.Sequential(conv_block(3 if i == 0 else 8, 8), nn.MaxPool2d(2)))

        for i in range(unet_levels):
            self.in_cnn.append(conv_block(8, 8))
        self.bottom_cnn = conv_block(8, 8)

        for i in range(unet_levels):
            self.out_cnn.append(conv_block(16, 8))
        self.last_cnn = nn.Sequential(nn.Conv2d(8, 1, kernel_size=[3,3], padding=[1,1]),
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
        return self

    def save(self, fname):
        tr.save(self.state_dict(), fname)


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=[3,3], padding=[1,1]),
                         nn.ELU(),
                         nn.BatchNorm2d(out_dim))
    