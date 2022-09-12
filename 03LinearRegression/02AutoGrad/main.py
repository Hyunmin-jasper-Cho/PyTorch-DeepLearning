'''
경사 하강법 코드의 requires_grad = True,
backward() 등의 함수들이 어떠한 기능을 제공하는지 에 대한 설명.

PyTorch 의 학습 과정을 보다 잘 이해할 수 있다.
'''

# 경사 하강법 -> loss function 에서 loss 가 최소화되는 방향을 찾아내는 알고리즘

import torch

if __name__ == '__main__':

    # z = 2w^2 + 5
    # requires_grad 값이 true 이므로, 계산한 기울기 값이 w.grad 에 저장된다.
    w = torch.tensor(2.0, requires_grad=True)

    y = w ** 2
    z = 2 * y + 5

    # 해당 수식에 대한 기울기를 계산한다.
    z.backward()

    print(f'수식을 w 로 미분한 값: {w.grad}')
