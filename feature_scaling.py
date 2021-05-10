import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/housing.csv",sep='\s+',
                 header=None,names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
x = df[['CRIM']].values
y = df.MEDV.values*1000

def hypothesis(T,X):
    return X@T

def cost(T,X,Y):
    m = X.shape[0]
    return np.sum((hypothesis(T,X)-Y)**2)/(2*m)

def gradient_descend(iters,alpha,X,Y):
    costs = []
    m = X.shape[0]
    ones = np.ones((m,1))
    X = np.hstack((ones,X))
    T = np.zeros(X.shape[1])
    for i in range(iters):
        costs.append(cost(T,X,Y))
        T = T - (alpha/m)*((hypothesis(T,X)-Y)@X)
    return T,costs

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

alpha = 0.01
iters = 300
T ,costs = gradient_descend(iters,alpha,x,y)
ax1.scatter(x,y)
ax1.plot(x,T[0]+T[1]*x)
ax2.plot(range(iters),costs)
ax1.set_title('gradient_descend')

# one
norm_y = (y-y.mean())/(y.max()-y.min())
# T ,costs = gradient_descend(iters,alpha,x,norm_y)
# ax3.scatter(x,norm_y)
# ax3.plot(x,T[0]+T[1]*x)
# ax4.plot(range(iters),costs)



# -------------------x 1/x
x2 = np.hstack((x,1/x))
T,costs = gradient_descend(iters,alpha*0.05,x2,norm_y)
ax3.scatter(x,norm_y)
one = np.ones((x2.shape[0],1))
x2 = np.hstack((one,x2))
ax3.scatter(x2[:,1],x2@T)
ax4.plot(range(iters),costs)

# -----------------1/x
# norm_y = (y-y.mean())/(y.max()-y.min())
# xinv = 1/x
# T ,costs = gradient_descend(iters,alpha*0.01,xinv,norm_y)
# ax3.scatter(x,norm_y)
# ax3.scatter(x,T[0]+T[1]*xinv)
# ax4.plot(range(iters),costs)

plt.show()
