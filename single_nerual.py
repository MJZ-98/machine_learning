import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

def cost(y,predict):
    return np.sum(1/2*np.power(y - predict,2))

data = np.array([0.05 ,0.1 ,0.1])
x = np.append([1],data[:2])
y = data[-1]
w = np.array([0.35, 0.15, 0.2])
iters = 1000
alpha = 1
costs = []
while iters > 0:
    h = logistic_hypothesis(w,x)
    g = (h - y) * (1 - h)*h*x
    w -= alpha * g
    iters -= 1
    costs.append(cost(y,h))
print(costs[-1])