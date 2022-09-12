import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def get_data():
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]

    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)

    return DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=2, shuffle=True)


if __name__ == '__main__':
    loader = get_data()

    W = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optim = torch.optim.SGD([W, b], lr=1)
    epochs = 100000
    for epoch in range(epochs + 1):
        loss = None
        for batch_idx, data in enumerate(loader):
            X, y = data

            forward = sigmoid(torch.matmul(X, W) + b)
            # loss = -((y * torch.log(forward)) + ((1 - y) * torch.log(1 - forward)))
            loss = F.binary_cross_entropy(forward, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        assert loss is not None
        if epoch % 10000 == 0:
            print(f'Epoch: {epoch:4d}/{epochs:4d}, loss: {loss:.4f}')

    x_new = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
    pred = sigmoid(torch.matmul(x_new, W) + b)
    print(f'Prediction result\n{pred}')

    pred = pred >= torch.FloatTensor([0.5])
    print(f'Logistic Regress \n{pred}')
