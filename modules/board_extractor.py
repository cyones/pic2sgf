import numpy as np
from PIL import Image


class BoardExtractor():
    def __init__(self, board_size):
        self.width = board_size * (2**4)
        self.height = board_size * (2**4)
        self.border = 1 / board_size

    def __call__(self, img, vertexs):
        board = Image.new('RGB', (self.width, self.height))
        board_pix = board.load()
        img_pix = img.load()
        for i in range(self.width):
            for j in range(self.height):
                u = ((1.0 + self.border) * i / self.width) - self.border / 2
                v = ((1.0 + self.border) * j / self.height) - self.border / 2
                x = v * (u * vertexs[0] + (1-u) * vertexs[1]) + (1-v) * (u * vertexs[3] + (1-u) * vertexs[2])
                x[0], x[1] = np.clip(x[0], 0, img.size[0]-1), np.clip(x[1], 0, img.size[1]-1)
                board_pix[i,j] = img_pix[int(x[0]), int(x[1])]
        return board