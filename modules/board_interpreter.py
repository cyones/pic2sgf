from os import path
import numpy as np
from pic2sgf.models.cnn import CNN
from torchvision import transforms

params_path = path.join(path.dirname(__file__), '../models/parameters/cnn.pmt')



class BoardInterpreter():
    def __init__(self):
        self.cnn = CNN()
        self.cnn.load(params_path)
        self.cnn.eval()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        x = self.to_tensor(image).unsqueeze(0)
        board = self.cnn(x)
        board = board.detach().numpy()
        board = board.round().astype(int)
        return board
