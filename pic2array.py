from .src import CornerDetector, BoardExtractor, BoardInterpreter, BoardSizer
from PIL import Image
import torch
IMG_SIZE = (512, 384)


class Pic2Array():
    def __init__(self, num_threads, return_report=False):
        torch.set_num_threads(num_threads)
        self.return_report = return_report
        self.corner_detector = CornerDetector()
        self.board_sizer = BoardSizer()
        self.board_extractor = { 9: BoardExtractor(9),
                                13: BoardExtractor(13),
                                19: BoardExtractor(19)}
        self.board_interpreter = BoardInterpreter()


    def __call__(self, image):
        corner_prediction, corner_confidence = self.corner_detector(image)
        size_prediction, size_confidence = self.board_sizer(image, corner_prediction)
        board_image, board_tilt = self.board_extractor[size_prediction](image, corner_prediction)
        board_image = board_image.transpose(Image.ROTATE_180)
        configuration_prediction, configuration_confidence = self.board_interpreter(board_image)

        if self.return_report:
            report = {
                'corner_prediction' : corner_prediction,
                'corner_confidence' : corner_confidence,
                'size_prediction' : size_prediction,
                'size_confidence' : size_confidence,
                'board_tilt' : board_tilt,
                'board_image' : board_image,
                'configuration_prediction' : configuration_prediction,
                'configuration_confidence' : configuration_confidence
            }
        else:
            report = None

        return configuration_prediction, report

        