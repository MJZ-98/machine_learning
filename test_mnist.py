from my_ml import *
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/handfig_train.csv")
INPUT_COUNT = data.shape[1] -1
HIDDEN_COUNT = 100
OUTPUT_COUNT = 10

nn = NeuralNetworks(INPUT_COUNT,OUTPUT_COUNT,[HIDDEN_COUNT],weight_type=1)
# input_weight = np.random.normal(0,pow(INPUT_COUNT,-0.5),(HIDDEN_COUNT,INPUT_COUNT + 1))
# out_weight = np.random.normal(0,pow(HIDDEN_COUNT,-0.5),(OUTPUT_COUNT,HIDDEN_COUNT + 1))
epoch = 1
while epoch:
    for row in data.values:
        target = row[0]
        x = row[1:] / 255 * 0.99 + 0.01
        y = np.zeros(10) + 0.01
        y[target] = 0.99
        nn.train(x,y)
    epoch-=1

test_data = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/mnist_test.csv")
score = []
for row in test_data.values:
    x = row[1:] / 255 * 0.99 + 0.01
    y = row[0]
    pre = nn.predict(x)
    label = np.argmax(pre)
    score.append(y==label)

print(np.count_nonzero(score))