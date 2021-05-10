import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/housing.csv",sep='\s+',
                 header=None,names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

x = df[['CRIM']].values
y = df.MEDV.values*1000
fig,(ax1,ax2) = plt.subplots(2,1)

def addBias(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    return np.hstack((ones,X))
def normal_eq(X,y):
    X = addBias(X)
    return np.linalg.inv(X.T@X)@X.T@y

T = normal_eq(x,y)
ax1.scatter(x,y)
ax1.plot(x,T[0]+T[1]*x)
ax1.set_title("normal_eq")
print(T,x,T*x)

norm_y = (y-y.mean())/(y.max()-y.min())
T = normal_eq(x,norm_y)
ax2.scatter(x,norm_y)
ax2.plot(x,T[0]+T[1]*x)
ax2.set_title("after normalization")
plt.show()