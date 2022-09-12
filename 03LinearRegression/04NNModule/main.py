import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(1)

X = torch.FloatTensor(
    [[1],
     [2],
     [3]]
)

y = torch.FloatTensor(
    [[2],
     [4],
     [6]]
)

if __name__ == '__main__':

    model = nn.Linear(1, 1)  # in-out

    # model.parameters 에는 해당 Model 에 사용 되는 model parameter 값들이 존재하는데,
    # optimizer 초기화 시 해당 parameter 사용할 것임을 명시 및 lr 정의
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 100000

    for epoch in range(1 + epochs):
        result = model(X)
        loss = F.mse_loss(result, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10000 == 0:
            print(f'Epoch {epoch:4d}/{epochs:4d} loss: {loss.item():.6f}')

    print(f'Updated model parameter: {list(model.parameters())}')
    print(f'model 4 = {model(torch.FloatTensor([[4]]))}')
