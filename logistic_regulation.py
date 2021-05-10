import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

def feature_mapping(x1,x2,power):
    features = {}
    for k in range(1,power+1):
        for i in range(k+1):
            features['f{}{}'.format(k-i,i)] = np.power(x1,k-i)*np.power(x2,i)
    return pd.DataFrame(features).values

# 可视化数据
def visual_show(data,ax):
    pos_data = data[data.iloc[:,-1]==1]
    neg_data = data[data.iloc[:,-1]==0]
    ax.scatter(pos_data.test1,pos_data.test2,label='Postive',marker='+')
    ax.scatter(neg_data.test1,neg_data.test2,label='Negative',marker='x')
    ax.legend()

# 预测
def predict(T,x,y):
    predicts = logistic_hypothesis(T,addBias(X))
    predicts = [1 if x>0.5 else 0 for x in predicts]
    corrects = np.count_nonzero(predicts==y)
    print(corrects/len(y))

# 显示当前决策边界
def show_decision_boudary(T,ax):
    x = np.linspace(-1,1.5,200)
    xx,yy = np.meshgrid(x,x)
    z = feature_mapping(xx.ravel(),yy.ravel(),N)
    z = addBias(z)@T
    z = z.reshape(xx.shape)
    ax.contour(xx,yy,z,0)

data = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/ex2data2.txt",sep=',',header=None,names=['test1','test2','accepted'])
figure, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 8))
visual_show(data,ax1)
# 特征提取
X = feature_mapping(data.test1,data.test2,6)
y = data.iloc[:,-1].values
# 训练模型
T , costs = gradient_descend(2000,0.45,X,y,'Logistic')
# 显示代价函数曲线
ax2.plot(range(len(costs)),costs)
N=6
predict(T,X,y)
show_decision_boudary(T,ax1)

# 使用正则化的功能
visual_show(data,ax3)
T_L , costs_L = gradient_descend(1000,0.45,X,y,'Logistic',L=1)
ax4.plot(range(len(costs_L)),costs_L)
print('after L_regulation:')
predict(T_L,X,y)
show_decision_boudary(T,ax3)

plt.show()
