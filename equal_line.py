import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig ,[h_ax ,j_ax] = plt.subplots(2,1,figsize=(6,8))

data = [[i,i] for i in range(1,4)]
df = pd.DataFrame(data,columns=['area','price'])

h_ax.scatter(df.area,df.price)

def hypothesis2(t0,t1):
    return t0 + df.area.values * t1
def cost2(T0,T1):
    T0 = T0.reshape(T0.shape+(1,))
    T1 = T1.reshape(T1.shape + (1,))
    m = df.shape[0]
    return ((hypothesis2(T0,T1) - df.price.values)**2).sum(axis=2)/(2*m)

n = 500
t0 = np.linspace(-3,df.price.max(),n)
t1 = np.linspace(-2,5,n)
T0 , T1 = np.meshgrid(t0,t1)

C = cost2(T0,T1)
res = j_ax.contour(T0,T1,C,200)
j_ax.clabel(res,inline=True)
j_ax.set_ylabel('theta1')
j_ax.set_xlabel('theta0')
min_pos = C.argmin()
print(min_pos)
print(C.min())
pos = (min_pos // C.shape[1], min_pos%C.shape[1])
print(pos)
min_t0 , min_t1 = t0[pos[1]] , t1[pos[0]]
print(min_t0,min_t1)
h_ax.plot(df.area,[i*min_t1+min_t0 for i in df.area])
# 展示参数theta的变化 对代价函数及假设函数的影响
sorted_c = C.copy()
sorted_c = sorted_c.reshape(-1,)
sorted_c.sort()
plt.ion()
while True:
    plt.pause(0.5)
    q = input('input any character to quit:')
    if q:
        break
    t = sorted_c[np.random.randint(0,10000)]
    [pos,*_] = np.argwhere(C==t)
    m_t0,m_t1 = t0[pos[1]] , t1[pos[0]]
    j_ax.scatter(m_t0,m_t1,color='red',marker='x')
    h_ax.plot(df.area,[i*m_t1+m_t0 for i in df.area])

plt.show()
