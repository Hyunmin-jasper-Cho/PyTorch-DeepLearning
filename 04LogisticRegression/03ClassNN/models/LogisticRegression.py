import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.linear = nn.Linear(_in, _out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))