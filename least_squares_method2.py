import numpy as np
import matplotlib.pyplot as plt

# 最小二乘法
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
