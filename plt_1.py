# coding=gbk
import matplotlib.pyplot as plt
import math
import random


fig = plt.figure(figsize=[8,8])
# # x = [i/1000*math.pi for i in range(0,4000,1)]
# # y = [math.sin(i) for i in x]
# #
# # ax1 = fig.add_subplot(221)
# # ax1.set_xticks([i/2*math.pi for i in range(9)])
# # ax1.set_xticklabels(['%d pi'] %( i*math.pi )for i in range(9))
# # ax1.plot(x,y)
# # plt.show()

y = [1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
y2 = [random.randint(0,3) for _ in range(len(y))]
x = range(10,10+len(y))

ax2 = fig.add_subplot(222)
ax2.set_xticks(x)
ax2.set_xticklabels(['%dyears'% i for i in x],rotation=45)
ax2.plot(x,y)

ax3 = fig.add_subplot(223)
ax3.set_xticks(x)
ax3.set_xticklabels(['%dyears'% i for i in x],rotation=45)
ax3.plot(x,y2,label="2")
ax3.plot(x,y,label="1")

ax3.annotate('i have been the peak',xy=(x[y.index(max(y))],max(y)))
ax3.legend(loc="upper left")

fig.text(0.5,0.5,'kill the two people',fontsize=80,alpha=0.2,va='center',ha='center')

plt.show()

################

fig = plt.figure(figsize=(16,9),dpi=72)

x = [t for t in range(120+1)]
y = [random.randint(20,35) for _ in range(120+1)]

ax1 = fig.add_subplot(221)
ax1.plot(x,y)

ax2 = fig.add_subplot(222)
ax2.set_xticks(x[::10])
ax2.plot(x,y)

ax3 = fig.add_subplot(223)
ax3.set_xticks(x[::10])
#ax3.set_xticksables(['10点%d分'%(t%60) if t// 60< 1 else "11点%d分" (t%60) for t in x[::10],rotation=45])
ax3.set_xticklabels(['%d点%d分'%(t//60+10,t%60) for t in x[::10]],rotation=45)
ax3.plot(x,y)

ax4 = fig.add_subplot(224)
ax4.set_xticks(x[::10])
ax4.set_xlabel("时间")
ax4.set_ylabel("温度")
ax4.set_title("10--12 温湿度")
ax4.plot(x,y)
ax4.set_xticklabels(['%d点%d分'%(t//60+10,t%60) for t in x[::10]],rotation=45)

plt.show()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8),dpi=72)
ax1 = fig.add_subplot(221)
x=[1,2,3]
y=[6,5,4]
plt.plot(x,y)
plt.show()

import numpy as ny
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8),dpi=66)
ax1 = fig.add_subplot(221)
A=[12.5,15.3,23.2,26.4,33.5,34.4,39.4,45.2,55.4,60.9]
B=[21.2,23.9,32.9,34.1,42.5,43.2,49.0,52.8,59.4,63.5]
plt.scatter(A,B)
plt.show()

f=pd.DataFrame({"A":[12.5,15.3,23.2,26.4,33.5,34.4,39.4,45.2,55.4,60.9],\
	"B":[21.2,23.9,32.9,34.1,42.5,43.2,49.0,52.8,59.4,63.5]})
print(f.corr())
