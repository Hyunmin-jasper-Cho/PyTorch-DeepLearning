""" using dot product """

import torch
import torch.optim as optim


def _get_data():
    x_train = torch.FloatTensor([[73, 80, 75],
                                 [93, 88, 93],
                                 [89, 91, 80],
                                 [96, 98, 100],
                                 [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

    return x_train, y_train


if __name__ == '__main__':
    X, y = _get_data()
    W = torch.zeros((3, 1), requires_grad=True)
    b = torch.zeros((1, 5), requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-5)
    epochs = 100000

    for epoch in range(1 + epochs):

        result = torch.matmul(X, W) + b
        loss = torch.mean((result - y) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5000 == 0:
            print('Epoch {:4d}/{} \nhypothesis: {} \nCost: {:.6f}\n\n'.format(
                epoch, epochs, result.squeeze().detach(), loss.item()
            ))
