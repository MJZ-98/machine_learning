import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# plt.figure(figsize=(8,8),dpi=72)
#
# data = pd.read_csv('C:/Users/MJZ/Desktop/something/machine learn/dataset/boston house.txt',delimiter=',',header=None,names=["area","rooms","price"])
# min_x,max_x=data.area.min()-10,data.area.max()+10
#
# def calc(k,b):
#     y1 = k*min_x + b
#     y2 = k*max_x + b
#     return [min_x,max_x],[y1,y2]
#
# plt.scatter(data.area,data.price)
# plt.autoscale(False)
# plt.ion()
#
# while True:
#     cmd = input("input the values k , b:")
#     if not cmd:
#         break
#     k,b = cmd.split(',')
#     k = eval(k)
#     b = eval(b)
#     rand=np.random.randint(min_x,max_x)
#     pre_y=k*rand+b
#     print("X:",rand,"Y:",pre_y)
#     x,y = calc(k,b)
#     plt.plot(x,y)
#     plt.pause(0.5)
#
# plt.ioff()
# plt.show()


data = [[i,i] for i in range(1,4)]
df = pd.DataFrame(data,columns=["area","price"])
fig, [h_ax,j_ax] = plt.subplots(2,1,figsize=(4,8))
h_ax.scatter(df.area,df.price)
j_ax.set_xticks(range(0,4))
j_ax.set_yticks(range(0,4))

j_ax.autoscale(False)

def hypoth(theta):
    return df.area*theta
def cost(theta):
    return ((df.price - hypoth(theta))**2).sum()/(2*df.shape[0])

plt.ion()
while True:
    plt.pause(0.5)
    t = input("请输入theta1的值:")
    if not t:
        break
    t = eval(t)
    h_y = hypoth(t)
    h_ax.plot(df.area,h_y)
    j = cost(t)
    j_ax.scatter(t,j,marker='x',color='red')

plt.ioff()
plt.show()