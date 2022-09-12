''' 다중 선형 회귀
-> 여러개의 independent variable 로부터 하나의
dependent variable y 를 결정하는 것. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def get_data():
    x1 = torch.FloatTensor([[73], [93], [89], [96], [73]])
    x2 = torch.FloatTensor([[80], [88], [91], [98], [66]])
    x3 = torch.FloatTensor([[75], [93], [90], [100], [70]])

    w1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(1, requires_grad=True)
    w3 = torch.zeros(1, requires_grad=True)
    b_ = torch.zeros(1, requires_grad=True)

    y_ = torch.FloatTensor([[152], [185], [180], [196], [142]])

    return [x1, x2, x3], [w1, w2, w3], y_, b_


if __name__ == '__main__':
    x_lst, w_lst, y, b = get_data()

    optimizer = optim.SGD([w_lst[0], w_lst[1], w_lst[2], b], lr=1e-5)
    epochs = 200000
    for epoch in range(epochs + 1):

        # 일단 result 계산 하고,
        result = x_lst[0] * w_lst[0] + x_lst[1] * w_lst[1] * x_lst[2] * w_lst[2] + b

        # result 기반으로 loss 계산
        loss = torch.mean((result - y)**2)

        # 최적화 프로세스
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10000 == 0:
            print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
                epoch, epochs, w_lst[0].item(), w_lst[1].item(), w_lst[2].item(), b.item(), loss.item()
            ))
