import numpy as np
from PIL import Image
from .exceptions import TooClosedAngleError

class BoardExtractor():
    def __init__(self, board_size):
        self.size = board_size * 16
        B = np.array([[8, 8, 1],
                      [self.size-8, 8, 1],
                      [self.size-8, self.size-8, 1],
                      [8, self.size-8, 1]]).T
        self.T2 = np.linalg.inv(B[:, 0:3] * np.linalg.solve(B[:, 0:3], B[:, 3]))

    def __call__(self, img, vertexs):
        A = np.concatenate([vertexs.T, np.array([[1.0, 1.0, 1.0, 1.0]])], axis=0)
        T1 = A[:, 0:3] * np.linalg.solve(A[:, 0:3], A[:, 3])
        T = np.matmul(T1, self.T2)
        T /= T[2,2]
        board = img.transform((self.size, self.size),
                              method=Image.PERSPECTIVE,
                              data = T.reshape(-1),
                              resample=Image.BILINEAR)
        tilt = T[2,0:2]
        if np.abs(tilt.max()) > 0.01: raise TooClosedAngleError()
        return board.transpose(Image.ROTATE_180), T[2,0:2]
