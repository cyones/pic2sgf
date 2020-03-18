import torch as tr
import torchvision
from torch import nn

class BoardInterpreter(nn.Module):
    def __init__(self):
        super(BoardInterpreter, self).__init__()
        self.conv_blocks = nn.Sequential(conv_block(3, 8),
                                         nn.AvgPool2d(2),
                                         conv_block(8, 16),
                                         nn.AvgPool2d(2),
                                         conv_block(16, 32),
                                         nn.AvgPool2d(2),
                                         conv_block(32, 64),
                                         nn.AvgPool2d(2),
                                         conv_block(64, 64))
        
        self.board_interpreter = nn.Sequential(nn.Conv2d(64, 1, kernel_size=[1,1]),
                                               nn.Tanh())
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.board_checker = nn.Sequential(nn.Linear(64, 32),
                                           nn.ELU(),
                                           nn.BatchNorm1d(32),
                                           nn.Linear(32, 8),
                                           nn.ELU(),
                                           nn.BatchNorm1d(8),
                                           nn.Linear(8, 1),
                                           nn.Sigmoid())

    def forward(self, x):
        x = self.conv_blocks(x)
        board = self.board_interpreter(x).squeeze()
        x = self.global_pooling(x)
        is_correct = self.board_checker(x.view(-1, 64)).squeeze()
        return board, is_correct
      
    def load(self, fname):
        self.load_state_dict(tr.load(fname, map_location=lambda storage, loc: storage))
        return self

    def save(self, fname):
        tr.save(self.state_dict(), fname)



def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=[3,3], padding=[1,1]),
                         nn.ELU(),
                         nn.BatchNorm2d(out_dim))