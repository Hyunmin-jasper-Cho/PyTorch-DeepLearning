import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.CustomDataset import CustomDataset


if __name__ == '__main__':
    dataset = CustomDataset()
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    model = nn.Linear(3, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    epochs = 100000

    for epoch in range(1 + epochs):
        for batch_idx, data in enumerate(loader):

            X, y = data
            forward = model(X)
            loss = F.mse_loss(forward, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if epoch % 10000 == 0:
                print(f'Epoch: {epoch:4d}/{epochs:4d}, batch: {batch_idx} loss: {loss.item():.4f}')

    new_var = torch.FloatTensor([[73, 80, 75]])
    print(f'Test for [73, 80, 75]: {model(new_var)}')
