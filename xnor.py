import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

def train(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return gradient_descend(100, 1, X, y, 'Logistic')

def showSamples(ax,data):
    ax.scatter(data[data.y == 0].x1, data[data.y == 0].x2)
    ax.scatter(data[data.y == 1].x1, data[data.y == 1].x2)

def showBoudary(ax,data,T):
    t = -T / T[2]
    ax.plot(data.x1, t[0] + t[1] * data.x1)

fig ,axes = plt.subplots(1,4,figsize=(10,6))

# and
data_and = pd.DataFrame([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]], columns=['x1', 'x2', 'y'])
T_and , costs_and  = train(data_and)
showSamples(axes[0],data_and)
showBoudary(axes[0],data_and,T_and)

# or
data_or = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], columns=['x1', 'x2', 'y'])
T_or , costs_or  = train(data_or)
showSamples(axes[1],data_or)
showBoudary(axes[1],data_or,T_or)

# not_and
data_notand = pd.DataFrame([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]], columns=['x1', 'x2', 'y'])
T_notand , costs_notand  = train(data_notand)
showSamples(axes[2],data_notand)
showBoudary(axes[2],data_notand,T_notand)

# -----------------XNOR
test = pd.DataFrame([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], columns=['x1', 'x2', 'y'])
X = test.iloc[:,:-1]
y = test.iloc[:,-1]
a21 = [1 if i>=0.5 else 0 for i in logistic_hypothesis(T_and,addBias(X))]
a22 = [1 if i>=0.5 else 0 for i in logistic_hypothesis(T_notand,addBias(X))]
a2 = np.array([a21,a22]).T
a3 = [1 if i>=0.5 else 0 for i in logistic_hypothesis(T_or,addBias(a2))]
print(np.count_nonzero(a3==y)/len(y))
showSamples(axes[3],test)
showBoudary(axes[3],test,T_and)
showBoudary(axes[3],test,T_notand)
# data_xnor = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=['x1', 'x2'])
# predict_and = [1 if i>0 else 0 for i in addBias(data_xnor)@T_and]
# print('and',predict_and)
# predict_notand = [1 if i>0 else 0 for i in addBias(data_xnor)@T_notand]
# print('notand',predict_notand)
# predict_xnor = np.array([predict_notand,predict_and])
# print(addBias(predict_xnor.T)@T_or)


plt.show()