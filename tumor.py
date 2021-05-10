from my_ml.ml_algorithms import *
import matplotlib.pyplot as plt

data = [0,0,0,0,0,0,1,0,1,1,1,1,1]
x = list(range(len(data)))
X = np.array(x).reshape((len(data),1))
y = np.array(data)
T,costs = gradient_descend(200,0.01,X,y,algo='Logistic')
plt.scatter(x,y)
# 自定义
plt.plot(x,addBias(X)@T)

# data.append(1)
# x = list(range(len(data)-1))+[40]
# X = np.array(x).reshape((len(data),1))
# y = np.array(data)
# T = normal_eq(X,y)
# plt.scatter(x,y)
# plt.plot(x,addBias(X)@T)
#
plt.show()