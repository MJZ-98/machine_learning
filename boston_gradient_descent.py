# 使用一元梯度下降方法找到给定数据的最优解
# 数据集及其属性描述：
# 该数据集共有 506 个观察，13 个输入变量和1个输出变量housing.csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据并查看数据
boston = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/boston_housing.data",header=None,sep='\s+')
print('boston:','\n',boston)
# 1、将数据集中的数据分成两部分，随机选取80%作为训练集，剩余20%作为测试集
boston_train = boston.iloc[:int(boston.shape[0]*0.8),:].sort_values(13)
boston_train_x = boston_train.iloc[:,:boston.shape[1]-1].values
boston_train_y = boston_train.iloc[:,boston.shape[1]-1:].values
boston_test = boston.iloc[int(boston.shape[0]*0.8):,:].sort_values(13)
boston_test_x = boston_test.iloc[:,:boston.shape[1]-1].values
boston_test_y = boston_test.iloc[:,boston.shape[1]-1:].values
# 2、从中选取一个特征，使用一元梯度下降法去拟合找出最优解，并使用得到的参数使用折线图绘制其在测试集上的预测
# 并在同一图上使用折线图绘制测试集中的目标值，为方便进行对比，建议先都对特征值进行排序

# 选取第6列RM为特征（相关系数较大）,并进行归一化
print('相关系数',np.corrcoef(boston.T)[-1])
boston_train_x = boston_train_x[:,5]
boston_test_x = boston_test_x[:,5]
def normal(X):
    return (X-X.min())/(X.max()-X.min())
boston_train_x_n = normal(boston_train_x)
boston_train_y_n = normal(boston_train_y)
boston_test_x_n = normal(boston_test_x)
boston_test_y_n = normal(boston_test_y)

# 代价函数（最小二乘法）:该函数只返回一个值
def compute_cost(b,k,x_data,y_data):
    total=0
    for i in range(0,len(x_data)):
        total+=(y_data[i]-(k*x_data[i]+b))**2
    return total/float(2*len(x_data))
# 梯度下降算法函数
def gradient_descent(x_data,y_data,b,k,lr,epochs=5000,erro=1e-8):
    # 总数据量
    m=float(len(x_data))
    # 迭代epochs次
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        for j in range(0,len(x_data)):
            b_grad+= -(1/m)*(y_data[j]-((k*x_data[j])+b))
            k_grad+= -(1/m)*x_data[j]*(y_data[j]-((k*x_data[j])+b))
        # 更新b和k
        last_k = k
        last_b = b
        b= b-(lr*b_grad)
        k= k-(lr*b_grad)
        # 每迭代500次，输出一次数据
        if i%50 ==0:
            print('epochs：',i)
            print('b:',b,'k:',k)
            print('cost:',compute_cost(b,k,x_data,y_data))
        if abs(compute_cost(b,k,x_data,y_data) - compute_cost(last_b,last_k,x_data,y_data)) < erro:
            break
    return b,k

# 学习率learning rate（步长）、截距、斜率、最大迭代次数
b,k = gradient_descent(boston_train_x_n,boston_train_y_n,b=0,k=1,lr=0.01)
print('斜率:',k,'截距:',b)

# 3、观察以上图形中的预测值与目标值之间的差异，试分析并得出所选特征与房价的关联程度
# RM关联性强且成正相关
fig = plt.figure()
ax = plt.subplot()
plt.scatter(boston_test_x_n,boston_test_y_n)
plt.plot(boston_test_x_n,boston_test_x_n*k+b,color='red')
ax.set_xticklabels(boston_test_x[::len(boston_test_x)//7])
ax.set_yticklabels(boston_test_y[::len(boston_test_y)//7])
plt.xlabel('RM')
plt.ylabel('price')
plt.legend(['回归预测','真实值'])
plt.show()




