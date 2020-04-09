from .src import CornerDetector, BoardExtractor, BoardInterpreter, BoardSizer
IMG_SIZE = (512, 384)

class Pic2Array():
    def __init__(self):
        self.corner_detector = CornerDetector()
        self.board_sizer = BoardSizer()
        self.board_extractor = {9 : BoardExtractor(9),
                                19: BoardExtractor(19)}
        self.board_interpreter = BoardInterpreter()


    def __call__(self, image):
        corners = self.corner_detector(image)
        board_size = self.board_sizer(image, corners)
        board_image = self.board_extractor[board_size](image, corners)
        board_configuration = self.board_interpreter(board_image)
        return board_configuration, board_image

        