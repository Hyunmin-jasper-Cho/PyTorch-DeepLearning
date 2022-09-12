import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        # nn.Module 을 상속 받는다.
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
