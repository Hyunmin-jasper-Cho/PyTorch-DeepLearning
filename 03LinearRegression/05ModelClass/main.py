import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LinearRegressionModel import LinearRegressionModel
from models.MultivariableLinearModel import MultivariableLinearModel

torch.manual_seed(1)


def _get_data():
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])

    return x_train, y_train


if __name__ == '__main__':

    X, y = _get_data()

    linear_model = LinearRegressionModel()
    multi_model = MultivariableLinearModel(3, 1)

    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-5)
    epochs = 300000

    for epoch in range(1 + epochs):
        pred = linear_model(X)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 30000 == 0:
            print(f'Epoch: {epoch:4d}/{epochs:4d}, loss: {loss.item():.4f}')

