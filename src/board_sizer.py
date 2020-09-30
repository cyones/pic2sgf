from os import path
import numpy as np
from torchvision.transforms import functional as ft

from . import BoardExtractor
from .exceptions import CornerDetectionError
from ..models import Sizer

params_path = path.join(path.dirname(__file__), '../models/parameters/sizer.pmt')


class BoardSizer():
    def __init__(self):
        self.cnn = Sizer()
        self.cnn.load(params_path)
        self.cnn.eval()
        self.classes = [9, 13, 19]
        self.board_extractor = BoardExtractor(192)

    def __call__(self, image, corners):
        image, _ = self.board_extractor(image, corners)
        x = ft.to_tensor(image).mean(0, keepdim=True).unsqueeze(0)

        probabilities = self.cnn(x).squeeze()
        probabilities[3] *= 1e-3

        class_idx = probabilities.detach().numpy().argmax()
        if class_idx == 3: raise CornerDetectionError()

        return self.classes[class_idx], probabilities[:2]
