from os import path
import numpy as np
from pic2sgf.models import Sizer
from torchvision.transforms import functional as ft

from pic2sgf.src import BoardExtractor

params_path = path.join(path.dirname(__file__), '../models/parameters/sizer.pmt')


class BoardSizer():
    def __init__(self):
        self.cnn = Sizer()
        self.cnn.load(params_path)
        self.cnn.eval()
        self.classes = [9, 13, 19]
        self.board_extractor = BoardExtractor(8)

    def __call__(self, image, corners):
        image, _ = self.board_extractor(image, corners)
        x = ft.to_tensor(image).mean(0, keepdim=True).unsqueeze(0)
        probabilities = self.cnn(x).detach().numpy()
        board_size = probabilities.argmax()
        return self.classes[board_size], probabilities
