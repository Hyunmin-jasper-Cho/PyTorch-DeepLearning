import torch.nn as nn


class MultivariableLinearModel(nn.Module):

    def __init__(self, _in, _out):
        super().__init__()
        self.multi = nn.Linear(_in, _out)

    def forward(self, x):
        return self.multi(x)
