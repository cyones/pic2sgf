from src import CornerDetector, BoardExtractor, BoardInterpreter, BoardSizer
from PIL import Image
from PIL.ImageOps import expand
from time import time
import torch as tr
import numpy as np
IMG_SIZE = (512, 384)


class Pic2Array():
    def __init__(self, num_threads):
        tr.set_num_threads(num_threads)
        self.corner_detector = CornerDetector()
        self.board_sizer = BoardSizer()
        self.board_extractor = { 9: BoardExtractor( 9 * 24),
                                13: BoardExtractor(13 * 24),
                                19: BoardExtractor(19 * 24)}
        self.board_interpreter = BoardInterpreter()

    def resize(self, image, size):
        new_image= Image.new(image.mode, size)
        image.thumbnail(size, Image.LANCZOS)
        x_offset= (new_image.size[0] - image.size[0]) // 2
        y_offset= (new_image.size[1] - image.size[1]) // 2
        new_image.paste(image, (x_offset, y_offset))
        return new_image

    def __call__(self, image):
        start = time()
        if image.size[0] > image.size[1]:
            image = self.resize(image, (512, 384))
        elif image.size[1] > image.size[0]:
            image = self.resize(image, (384, 512))
        else:
            image = image.resize((384,384), resample=Image.LANCZOS)

        corner_prediction, corner_confidence = self.corner_detector(image)
        size_prediction, size_confidence = self.board_sizer(image, corner_prediction)
        board_image, board_tilt = self.board_extractor[size_prediction](image, corner_prediction)
        board_image = board_image.transpose(Image.ROTATE_180)
        configuration_prediction, configuration_confidence = self.board_interpreter(board_image)
        
        report = {
            'corner_prediction' : corner_prediction,
            'corner_confidence' : corner_confidence,
            'size_prediction' : size_prediction,
            'size_confidence' : size_confidence,
            'board_tilt' : board_tilt,
            'board_image' : board_image,
            'configuration_confidence' : configuration_confidence,
            'elapsed_time' : time() - start
        }
        return configuration_prediction, report

        