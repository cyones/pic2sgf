import numpy as np
from PIL import Image


class BoardExtractor():
    def __init__(self, board_size, pers_correction_iters = 20):
        self.width = board_size * (2**4)
        self.height = board_size * (2**4)
        self.border = 1 / board_size
        self.pers_correction_iters = pers_correction_iters

    def calc_depth(self, v):
        z = np.array([1.0, 1.0, 1.0, 1.0])
        fv = np.concatenate([v, z.reshape(4,1)], axis=1)
        side = np.linalg.norm(v - np.roll(v, 1, axis=0), axis=1).mean()
        for i in range(self.pers_correction_iters):
            pre = np.linalg.norm(fv - np.roll(fv,  1, axis=0), axis=1)
            pos = np.linalg.norm(fv - np.roll(fv, -1, axis=0), axis=1)
            z *= 2 * side / (pre+pos)
            fv = np.concatenate([v * z[:,np.newaxis], z.reshape(4,1)], axis=1)
        return fv
    
    def __call__(self, img, vertexs):
        center = np.array([img.size[0]/2, img.size[1]/2])
        vertexs = vertexs[:] - center
        vertexs = self.calc_depth(vertexs)

        board = Image.new('RGB', (self.width, self.height))
        board_pix = board.load()
        img_pix = img.load()
        for j in range(self.height):
            for i in range(self.width):
                u = ((1.0 + self.border) * i / self.width) - self.border / 2
                v = ((1.0 + self.border) * j / self.height) - self.border / 2
                x = v * (u * vertexs[0] + (1-u) * vertexs[1]) + (1-v) * (u * vertexs[3] + (1-u) * vertexs[2])
                x = x / x[2]
                x = x[0:2] + center
                x[0], x[1] = np.clip(x[0], 0, img.size[0]-1), np.clip(x[1], 0, img.size[1]-1)
                board_pix[i,j] = img_pix[int(x[0]), int(x[1])]
        return board