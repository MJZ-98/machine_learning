import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

# def sigmoid(z):
#     return 1/(1+np.exp(-z))
#
# def logistic_hypothesis(w,x):
#     return sigmoid(w@x)

def costs(y,predict):
    return np.sum(1/2*np.power(y - predict , 2))

a1 = np.array([1,0.05,0.1])
w1 = np.array([[0.35,0.15,0.2],[0.35,0.25,0.3]])
w2 = np.array([[0.6,0.4,0.45],[0.6,0.5,0.55]])
y = np.array([0.01,0.99])

iters = 10000
alpha = 0.5
while iters>0:
    iters -= 1
    # 正向传播
    a2 = logistic_hypothesis(w1, a1)
    a2 = np.append([1], a2)
    a3 = logistic_hypothesis(w2, a2)

    # dE_a31 = (a3[0]-y[0])
    # da31_z31 = a3[0] * (1 - a3[0])
    # dz31_w21 = a2[1]
    # dE_w21 = dE_a31 * da31_z31 * dz31_w21
    # w21_new = w2[0][1] - alpha * dE_w21
    dE_a3 = (a3 - y)
    da3_z3 = a3 * (1 - a3)
    dz3_w2 = a2
    delta3 = dE_a3 * a3 * da3_z3
    w2_new = w2 - alpha * delta3.reshape(-1,1) * dz3_w2
    # # 输入层到隐藏层的权重更新
    # dE1_a21 = delta3[0] * w2[0,1]
    # dE2_a21 = delta3[1] * w2[1,1]
    # dE_a21 = dE1_a21 + dE2_a21
    # da21_z21 = a2[1] * (1 - a2[1])
    # dz21_w1_11 = a1[1]
    # dE_w1_11 = dE_a21 * da21_z21 * dz21_w1_11
    # w1_11_new = w1[0][1] - alpha * dE_w1_11
    # 向量化
    dE_a2 = delta3 @ w2[: , 1:]
    delta2 = dE_a2 * a2[1:] * (1 - a2[1:])
    w1_new = w1 - alpha * (delta2.reshape(-1,1) * a1)
    w1 , w2 = w1_new ,w2_new
print(costs(y,a3))
print(a3)