import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_data():
    x_train = torch.FloatTensor([[73, 80, 75],
                                 [93, 88, 93],
                                 [89, 91, 90],
                                 [96, 98, 100],
                                 [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

    return x_train, y_train


if __name__ == '__main__':
    X, y = _get_data()
    # X, y = X.to('mps'), y.to('mps')

    ML = nn.Linear(3, 1)  #.to('mps')
    optimizer = torch.optim.SGD(ML.parameters(), lr=2e-5)
    epochs = 1000000

    for epoch in range(1 + epochs):

        pred = ML(X)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100000 == 0:
            print(f'Epoch {epoch:4d}/{epochs:4d}, loss: {loss.item():.6f}')
