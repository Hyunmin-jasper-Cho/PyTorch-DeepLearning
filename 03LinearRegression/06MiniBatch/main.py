import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def get_data():
    x_train = torch.FloatTensor([[73, 80, 75],
                                 [93, 88, 93],
                                 [89, 91, 90],
                                 [96, 98, 100],
                                 [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

    return x_train, y_train


if __name__ == '__main__':
    X, y = get_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    model = nn.Linear(3, 1)
    optim = torch.optim.SGD(model.parameters(), lr=2e-6)

    epochs = 50000
    for epoch in range(1 + epochs):
        loss = None
        for batch_idx, data in enumerate(loader):
            tr_X, tr_y = data

            forward = model(tr_X)
            loss = F.mse_loss(forward, tr_y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        assert loss is not None
        if epoch % 5000 == 0:
            print(f'Epoch: {epoch:4d}/{epochs:4d}, loss: {loss.item():.4f}')

    new_var = torch.FloatTensor([
        [73, 80, 75]
    ])
    pred_y = model(new_var)
    print(f'Input: 73, 80, 75, output: {pred_y}')
