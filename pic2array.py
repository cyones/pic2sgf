from .src import CornerDetector, BoardExtractor, BoardInterpreter, BoardSizer
from PIL import Image
from time import time
import torch as tr
import numpy as np
IMG_SIZE = (512, 384)


class Pic2Array():
    def __init__(self, num_threads):
        tr.set_num_threads(num_threads)
        self.corner_detector = CornerDetector()
        self.board_sizer = BoardSizer()
        self.board_extractor = { 9: BoardExtractor(9),
                                13: BoardExtractor(13),
                                19: BoardExtractor(19)}
        self.board_interpreter = BoardInterpreter()


    def __call__(self, image):
        start = time()
        landscape = False
        if image.size[0] > image.size[1]:
            image = image.resize((512, 384), resample=Image.LANCZOS)
        elif image.size[0] < image.size[1]:
            landscape = True
            image = image.resize((384, 512), resample=Image.LANCZOS).transpose(Image.ROTATE_90)
        else:
            image = image.resize((384, 384), resample=Image.LANCZOS)


        corner_prediction, corner_confidence = self.corner_detector(image)
        size_prediction, size_confidence = self.board_sizer(image, corner_prediction)
        board_image, board_tilt = self.board_extractor[size_prediction](image, corner_prediction)
        board_image = board_image.transpose(Image.ROTATE_180)
        configuration_prediction, configuration_confidence = self.board_interpreter(board_image)
        
        if landscape:
            configuration_prediction = np.rot90(configuration_prediction, k=3)
            board_image = board_image.transpose(Image.ROTATE_270)
        
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

        