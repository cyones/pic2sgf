from os import path
import numpy as np
from scipy import stats
import torch
from torchvision.transforms import functional as ft
from scipy import ndimage

from .exceptions import NoBoardError, BoardTooFarError, MissingCornerError
from ..models import Segmenter

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
        segmentation = segmentation.detach().cpu().numpy().squeeze()
        segmentation[segmentation < 0.1] = 0.0

        ccomponent, ncomponent = ndimage.label(segmentation[0])
        if ncomponent < 1:
            raise NoBoardError

        greather_component, component_size = stats.mode(ccomponent[ccomponent>0], axis=None)
        if component_size < 3072: 
            raise BoardTooFarError()

        segmentation = segmentation[2]
        segmentation[ccomponent != greather_component] = 0.0
        
        ccomponent, ncomponent = ndimage.label(segmentation)
        confidence = np.zeros((4))
        if ncomponent < 4:
            raise MissingCornerError(ncomponent)

        vertexs = -np.ones((4, 2))
        for i in range(4):
            max_prob = segmentation.max()
            confidence[i] = max_prob
            maxs = np.argwhere(segmentation == max_prob)
            if maxs.ndim == 2: 
                maxs = maxs[0]
            vertexs[i] = maxs
            cc_label = ccomponent[int(vertexs[i,0]), int(vertexs[i,1])]
            segmentation[ccomponent == cc_label] = 0.0
        vertexs = 2 * vertexs[:,[1,0]]
        idxs = self.order_vertexs(vertexs, image.size)
        return vertexs[idxs], confidence[idxs] 

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

        last_prod = 0
        for i in range(len(v)):
            prev = v[idxs[(i-1)%4]] - v[idxs[i]]
            post = v[idxs[(i+1)%4]] - v[idxs[i]]
            cross_prod = np.cross(post, prev)
            if cross_prod * last_prod < 0:
                idxs[i], idxs[(i+1)%4] = idxs[(i+1)%4].copy(), idxs[i].copy()
                i += 1
            else:
                last_prod = cross_prod
        return idxs
