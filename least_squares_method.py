import numpy as np
import matplotlib.pyplot as plt

data = np.array([10.2,10.3,9.8,9.9,9.8])

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()

plt.ion()
y = np.random.uniform(data.min(),data.max())
span =0.01
s = 50
real_y = 0
while True:
    ax.scatter(range(len(data)),data)
    if y <= data.min() or y >= data.max():
        span = -span
    y += span
    sy = np.sum((data - y)**2)
    if s > sy:
        s = sy
        real_y = y
    ax.plot(range(len(data)),[y]*len(data),linestyle="--")
    ax.text(1,y,'y=%.2f,sum=%.2f,real_y=%.2f'%(y,s,real_y))
    for index,val in enumerate(data):
        ax.plot([index,index],[y,val])
    plt.pause(0.02)
    ax.cla()
plt.ioff()
plt.show()