from os import path
import numpy as np
from pic2sgf.models import Interpreter
from torchvision.transforms import functional as ft

params_path = path.join(path.dirname(__file__), '../models/parameters/interpreter.pmt')


class BoardInterpreter():
    def __init__(self):
        self.cnn = Interpreter()
        self.cnn.load(params_path)
        self.cnn.eval()

    def __call__(self, image):
        x = ft.to_tensor(image).unsqueeze(0)
        board = self.cnn(x)
        board = board.detach().permute(1,0).numpy()
        board = board.round().astype(int)
        return board
