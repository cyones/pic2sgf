from os import path
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans

from scipy.sparse import coo_matrix

from pic2sgf.models.unet import UNET

params_path = path.join(path.dirname(__file__), '../models/parameters/unet.pmt')

class VertexDetector():
    def __init__(self):
        self.unet = UNET(prelayers=2, unet_levels=4)
        self.unet.load(params_path)
        self.unet.eval()

        self.kmeans = KMeans(n_clusters=4,
                             n_init=5,
                             precompute_distances=True,
                             algorithm='elkan')
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        x = self.to_tensor(image).unsqueeze(0)
        segmentation = self.unet(x).detach().squeeze().numpy()

        segmentation[segmentation < 0.01] = 0.0
        segmentation = coo_matrix( segmentation )
        x = np.stack([segmentation.col, segmentation.row]).transpose()
        w = segmentation.data

        km_model = self.kmeans.fit(x, sample_weight=w)
        vertexs = km_model.cluster_centers_ * 4
        vertexs = self.order_vertexs(vertexs, image.size)
        return vertexs

    def order_vertexs(self, vertexs, img_size):
        w, h = img_size
        idxs = np.array([np.linalg.norm(vertexs, ord=2, axis=1).argmin(),
                         np.linalg.norm(vertexs - np.array([w,0]), ord=2, axis=1).argmin(),
                         np.linalg.norm(vertexs - np.array([w,h]), ord=2, axis=1).argmin(),
                         np.linalg.norm(vertexs - np.array([0,h]), ord=2, axis=1).argmin()])

        vertexs = vertexs[idxs]
        return vertexs
