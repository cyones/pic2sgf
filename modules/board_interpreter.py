from models.cnn import CNN
from torchvision import transforms

class BoardInterpreter():
    def __init__(self):
        self.cnn = CNN()
        self.cnn.load("models/parameters/cnn.pmt")
        self.cnn.eval()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        x = self.to_tensor(image).unsqueeze(0)
        board, is_ok = self.cnn(x)
        board = board.detach().numpy().round().astype(int)
        is_ok = (is_ok > 0.5).item()
        return board, is_ok
        

