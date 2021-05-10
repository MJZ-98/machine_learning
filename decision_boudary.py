import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

data = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/ex2data1.txt",sep=',',header=None)
print(data)

feature = data.iloc[:,:-1].values
target =  data.iloc[:,-1].values

scaler = StandardScaler()
scaler.fit_transform(feature)

plt.scatter(feature[data.iloc[:,-1]==0][:,0],feature[data.iloc[:,-1]==0][:,1],label='0')
plt.scatter(feature[data.iloc[:,-1]==1][:,0],feature[data.iloc[:,-1]==1][:,1],marker='D',label='1')
plt.legend()

T, cost = gradient_descend(1000,0.06,feature,target,'Logistic')

def show_decision_boudary(T,ax):
    x = np.linspace(-2,2,200)
    xx,yy = np.meshgrid(x,x)
    z = np.c_[xx.ravel(),yy.ravel()]
    z = addBias(z)@T
    z = z.reshape(xx.shape)
    ax.contour(xx,yy,z,0)

# x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# x2 = (-T[1] * x1 - T[0]) / T[2]
# ax1.plot(x1, x2)

# print(target)
# print(logistic_hypothesis(T,addBias(feature)))
show_decision_boudary(T,plt)
plt.show()
