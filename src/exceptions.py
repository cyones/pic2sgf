
class NoBoardError(Exception):
    def __str__(self):
        return "No board found in picture."

class BoardTooFarError(Exception):
    def __str__(self):
        return "The board is too far."

class MissingCornerError(Exception):
    def __str__(self, ncorners):
        return f"There are {4-ncorners} missing corners on the picture."

class TooClosedAngleError(Exception):
    def __str__(self):
        return "The board was photographed from a too close angle."

class CornerDetectionError(Exception):
    def __str__(self):
        return "The process to detect the corners failed."