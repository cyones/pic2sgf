from .modules import VertexDetector, BoardExtractor, BoardInterpreter

class Pic2Array():
    def __init__(self, board_size):
        self.vertex_detector = VertexDetector()
        board_image_size = board_size * 2**4
        self.board_extractor = BoardExtractor(board_image_size, board_image_size)
        self.board_interpreter = BoardInterpreter()


    def __call__(self, image):
        vertexs, _ = self.vertex_detector(image)
        board_image = self.board_extractor(image, vertexs)
        board_configuration, is_ok = self.board_interpreter(board_image)
        return board_configuration, is_ok, board_image

        