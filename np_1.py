import numpy as np
import matplotlib.pyplot as plt

g1 = np.loadtxt("C:/Users/MJZ/Desktop/something/machine learn/dataset/youtube/GB_video_data_numbers.csv",delimiter=',')
u1 = np.loadtxt("C:/Users/MJZ/Desktop/something/machine learn/dataset/youtube/US_video_data_numbers.csv",delimiter=',')

fig = plt.figure(figsize=(16,8),dpi=80)
ax1 = fig.add_subplot(221)
ax1.set_title("美国youtube评论数量直方图")
plt.hist(u1[:,-1],bins=20)

ax2 = fig.add_subplot(222)
ax2.set_title("英国youtube评论数量直方图")
plt.hist(g1[:,-1],bins=20)

ax3 = fig.add_subplot(223)
ax3.set_title("英国的youtube中视频的评论数和喜欢数的关系")
ax3.set_xlabel("评论数")
ax3.set_ylabel("喜欢数")
plt.scatter(g1[:,-1],g1[:,-3])

# ax4 = fig.add_subplot(224)
# ax4.set_title("美国的youtube中视频的评论数和喜欢数的关系")
# ax4.set_xlabel("评论数")
# ax4.set_ylabel("喜欢数")
# plt.scatter(u1[:,-1],u1[:,-3])

plt.show()


#################################


np.random.seed(10)
a = np.random.randint(1,5,18).reshape(6,3)
b = np.random.randint(1,5,4).reshape(2,2)
print(a,"\n\n",b)
a1 = a.reshape(a.shape+(1,1))
c = (a1 == b)
c1 = c.any(1)
c2 = c1.any(2)
c3 = c2.all(1)
print("c:",c,"\nc1:",c1,"\nc2:",c2,"\nc3:",c3)
print(a[c3])

np.random.seed(10)
a = np.random.randint(1,10,8).reshape(2,4)
print("origin:\n",a)
a[[0,1],:] = a[[1,0],:]
print("after swap:\n",a)

# Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?

a = np.arange(1,6)
a = a.reshape(a.shape[0],1)
z = np.zeros((3,1),dtype=int)
a = np.insert(a,1,values=z,axis=1)

print(a.flatten())