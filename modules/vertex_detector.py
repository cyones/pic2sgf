from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans

from scipy.sparse import coo_matrix

from models.unet import UNET

class VertexDetector():
    def __init__(self):
        self.unet = UNET(prelayers=2, unet_levels=4)
        self.unet.load("models/parameters/unet.pmt")
        self.unet.eval()

        self.to_tensor = transforms.ToTensor()
        self.kmeans = KMeans(n_clusters=4,
                             n_init=5,
                             precompute_distances=True,
                             algorithm='elkan')

    def __call__(self, image):
        x = self.to_tensor(image).unsqueeze(0)
        segmentation = self.unet(x).detach().squeeze().numpy()

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

