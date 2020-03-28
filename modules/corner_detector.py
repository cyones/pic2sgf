from os import path
import numpy as np
from torchvision.transforms import functional as ft
from sklearn.cluster import KMeans

from scipy.sparse import coo_matrix

from pic2sgf.models import Segmenter

params_path = path.join(path.dirname(__file__), '../models/parameters/segmenter.pmt')

class CornerDetector():
    def __init__(self):
        self.unet = Segmenter()
        self.unet.load(params_path)
        self.unet.eval()

        self.kmeans = KMeans(n_clusters=4,
                             n_init=5,
                             precompute_distances=True,
                             algorithm='elkan')

    def __call__(self, image):
        x = ft.to_tensor(image).unsqueeze(0)
        segmentation = self.unet(x)
        segmentation = segmentation.detach().numpy().squeeze()[2]
        
        segmentation[segmentation < 0.1] = 0.0
        segmentation = coo_matrix( segmentation )
        x = np.stack([segmentation.col, segmentation.row]).transpose()
        w = segmentation.data

        km_model = self.kmeans.fit(x, sample_weight=w)
        vertexs = km_model.cluster_centers_ * 4
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

        return v[idxs]