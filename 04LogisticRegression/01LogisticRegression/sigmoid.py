import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(0.5*x)
    y = sigmoid(x)
    y2 = sigmoid(2*x + 1)

    plt.plot(x, y1, 'r', linestyle='--')
    plt.plot(x, y, 'g')
    plt.plot(x, y2, 'b', linestyle='--')
    plt.plot([0, 0], [1.0, 0.0], ':')
    plt.title('sigmoid function')
    plt.show()
