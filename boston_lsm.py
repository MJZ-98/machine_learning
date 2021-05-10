import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 最小二乘法
def lsm():
    arr = np.array([[25,27,31,33,35],[110,115,155,160,180]])
    s_x2 = sum(arr[0]**2)
    s_x = sum(arr[0])
    s_xy = sum(arr[0]*arr[1])
    s_y = sum(arr[1])
    A = np.array([[s_x2,s_x],[s_x,arr.shape[1]]])
    b = np.array([[s_xy,s_y]]).T
    res = np.linalg.solve(A,b)
    print(res)

    import pandas as pd
    df = pd.DataFrame(arr.T,columns=['tem','sales'])
    df.plot(kind='scatter',x='tem',y='sales')
    plt.plot(arr[0],[i*res[0]+res[1] for i in arr[0]],color='red')
    plt.show()

boston = pd.read_csv('/Users/MJZ/Desktop/something/machine learn/dataset/boston house.txt', delimiter=',', header=None)
boston = boston.sort_values(0)
print(boston.head(5))
arr = np.array([boston[0].values,boston[2].values])
s_x4 = sum(arr[0]**4)
s_x3 = sum(arr[0]**3)
s_x2 = sum(arr[0]**2)
s_x = sum(arr[0])

s_x2y = sum(arr[0]**2*arr[1])
s_xy = sum(arr[0]*arr[1])
s_y = sum(arr[1])
A = np.array([[s_x4,s_x3,s_x2],[s_x3,s_x2,s_x],[s_x2,s_x,arr.shape[1]]])
b = np.array([[s_x2y,s_xy,s_y]]).T
res = np.linalg.solve(A,b)
print(res)

import pandas as pd
df = pd.DataFrame(arr.T,columns=['X','price'])
df.plot(kind='scatter',x='X',y='price')
plt.plot(arr[0],[i**2*res[0]+i*res[1]+res[2] for i in arr[0]],color='red')
plt.show()
