from os import path
import numpy as np
import torch
from torchvision.transforms import functional as ft
from sklearn.cluster import KMeans

from scipy.sparse import coo_matrix

from pic2sgf.models import Segmenter

params_path = path.join(path.dirname(__file__), '../models/parameters/segmenter.pmt')

class CornerDetector():
    def __init__(self, gpu=False):
        self.unet = Segmenter()
        self.unet.load(params_path)
        self.unet.eval()
        if gpu: self.unet = self.unet.cuda()
        self.gpu = gpu
        self.kmeans = KMeans(n_clusters=4,
                             n_init=5,
                             precompute_distances=True,
                             algorithm='elkan')

    def __call__(self, image):
        tensor = ft.to_tensor(image).unsqueeze(0)
        print(tensor.dtype)
        if self.gpu: tensor = tensor.cuda()
        from time import time; start = time()
        segmentation = self.unet(tensor)
        print(f"unet: {time() - start} s")
        segmentation = segmentation.detach().cpu().numpy().squeeze()[2]
        
        segmentation[segmentation < 0.1] = 0.0
        segmentation = coo_matrix( segmentation )
        x = np.stack([segmentation.col, segmentation.row]).transpose()
        w = segmentation.data

        km_model = self.kmeans.fit(x, sample_weight=w)
        vertexs = 4 * km_model.cluster_centers_
        vertexs = self.order_vertexs(vertexs, image.size)
        return vertexs

    def order_vertexs(self, v, img_size):
        w, h = img_size
        vc = v.copy()
        idxs = np.ones(4).astype(int)
        idxs[0] = np.linalg.norm(vc, ord=2, axis=1).argmin()
        vc[idxs[0]] = np.array([float('inf'), float('inf')])

        idxs[1] = np.linalg.norm(vc - np.array([w,0]), ord=2, axis=1).argmin()
        vc[idxs[1]] = np.array([float('inf'), float('inf')])

        idxs[2] = np.linalg.norm(vc - np.array([w,h]), ord=2, axis=1).argmin()
        vc[idxs[2]] = np.array([float('inf'), float('inf')])

        idxs[3] = np.linalg.norm(vc - np.array([0,h]), ord=2, axis=1).argmin()
        vc[idxs[3]] = np.array([float('inf'), float('inf')])
        v = v[idxs]

        last_prod = 0
        for i in range(len(v)):
            prev = v[(i-1)%4] - v[i]
            post = v[(i+1)%4] - v[i]
            cross_prod = np.cross(post, prev)
            if cross_prod * last_prod < 0:
                v[i], v[(i+1)%4] = v[(i+1)%4].copy(), v[i].copy()
                i += 1
            else:
                last_prod = cross_prod
        return v
