import sys, os
sys.path.append("./dataset")
os.chdir('C:/Users/이동학/Desktop/deep-learningmaster/ch03')
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)


def cross_entropy_error(y, t):
    if y.nidm == 1: # y가 1차원 이라면
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

