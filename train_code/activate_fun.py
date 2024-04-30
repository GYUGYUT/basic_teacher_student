import torch.nn as nn

class softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(x)

class log_softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        return self.log_softmax(x)

class sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid(dim=1)
    def forward(self, x):
        return self.sig(x) 
