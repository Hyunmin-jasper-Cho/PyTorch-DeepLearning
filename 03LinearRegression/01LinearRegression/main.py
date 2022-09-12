import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set random seed


def init():
    torch.manual_seed(1)
    x_train = torch.FloatTensor(
        [[1], [2], [3]]
    )
    y_train = torch.FloatTensor(
        [[2], [4], [6]]
    )

    return x_train, y_train


if __name__ == '__main__':
    x_train, y_train = init()

    # print out tensor's size -> 3x1
    print(f'The size of x and y\n{x_train.size()}\n{y_train.size()}')

    # 가중치 W 를 0으로 초기화 한 뒤, 학습을 통해 값이 변경 된다는
    # requires_grad 값을 True 로 설정 해 준다.
    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 경사 하강법 구현-> 학습 대상인 W, b 입력
    optimizer = optim.SGD([W, b], lr=0.01)

    epochs = 2000

    for epoch in range(epochs + 1):

        # 현재 직선의 방정식: Y = 0x + 0
        # 때문에 학습에 적합한 가설을 세운다.
        # H(x) 값을 계산 하고,
        hypothesis = x_train * W + b

        # Loss function MSE 설정
        # H(x) 를 기반 으로 cost 계산한 뒤,
        cost = torch.mean((hypothesis - y_train) ** 2)

        # gradient 0 으로 초기화
        optimizer.zero_grad()
        # loss function 미분 하여 gradient 계산
        cost.backward()
        # W, b 값 업데이트
        optimizer.step()

        # print
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, epochs, W.item(), b.item(), cost.item()
            ))
