from os import path
import numpy as np
import torch
from torchvision.transforms import functional as ft

from scipy import ndimage

from pic2sgf.models import Segmenter

params_path = path.join(path.dirname(__file__), '../models/parameters/segmenter.pmt')

class CornerDetector():
    def __init__(self, gpu=False):
        self.unet = Segmenter()
        self.unet.load(params_path)
        self.unet.eval()
        if gpu: 
            self.unet = self.unet.cuda()
        self.gpu = gpu

    def __call__(self, image):
        tensor = ft.to_tensor(image).unsqueeze(0)
        if self.gpu:
            tensor = tensor.cuda()
        segmentation = self.unet(tensor)
        segmentation = segmentation.detach().cpu().numpy().squeeze()[2]

        segmentation[segmentation < 0.1] = 0.0
        ccomponent, ncomponent = ndimage.label(segmentation)
        if ncomponent < 4: raise Exception(f"Missing {4 - ncomponent} corners.")
        vertexs = -np.ones((4, 2))
        for i in range(4):
            vertexs[i] = np.argwhere(segmentation == segmentation.max())
            cc_label = ccomponent[int(vertexs[i,0]), int(vertexs[i,1])]
            segmentation[ccomponent == cc_label] = 0.0
        vertexs = 4 * vertexs[:,[1,0]]
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
