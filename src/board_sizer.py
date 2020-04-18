from os import path
import numpy as np
from pic2sgf.models import Sizer
from torchvision.transforms import functional as ft

params_path = path.join(path.dirname(__file__), '../models/parameters/sizer.pmt')


class BoardSizer():
    def __init__(self):
        self.cnn = Sizer()
        self.cnn.load(params_path)
        self.cnn.eval()
        self.classes = [9, 13, 19]

    def __call__(self, image, corners):
        bbox = (corners[:,0].min(), corners[:,1].min(), corners[:,0].max(), corners[:,1].max()) 
        image = image.crop(bbox).resize((256, 192))
        x = ft.to_tensor(image).unsqueeze(0)
        probabilities = self.cnn(x).detach().numpy()
        board_size = probabilities.argmax()
        return self.classes[board_size], probabilities
