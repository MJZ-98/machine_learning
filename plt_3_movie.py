# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=[16,16])
# ax1 = fig.add_subplot(221)

# a = ["战狼2","速度与激情8","功夫瑜伽","西游伏妖篇","变形金刚5：最后的骑士","摔跤吧！爸爸","加勒比海盗5：死无对证","金刚：骷髅岛","极限特工：终极回归","生化危机6：终章","乘风破浪","神偷奶爸3","智取威虎山","大闹天竺","金刚狼3：殊死一战","蜘蛛侠：英雄归来","悟空传","银河护卫队2","情圣","新木乃伊",]
# b = [56.01,26.94,17.53,16.49,15.45,12.96,11.8,11.61,11.28,11.12,10.49,10.3,8.75,7.55,7.32,6.99,6.88,6.86,6.58,6.23]

# x = range(len(a))
# y = b
# ax1.set_xticks(x)
# # ax1.set_yticks()
# ax1.set_xticklabels(a,rotation=90)
# plt.bar(x,y,width=0.2,color='blue')
# plt.show()

import matplotlib.pyplot as plt

a = ["猩球崛起3：终极之战","敦刻尔克","蜘蛛侠：英雄归来","战狼2"]
b_14 = [2358,399,2358,362]
x_14 = [i for i in range(4)]
b_15 = [12357,156,2045,168]
x_15 = [i+5 for i in x_14]
b_16 = [15746,312,4497,319]
x_16 = [i+5 for i in x_15]

fig = plt.figure(figsize=[16,16])
ax1 = fig.add_subplot(221)

x=[[]for i in range(4)]
b=[[]for i in range(4)]
for i in range(4):
    x[i] = [x_14[i]] + [x_15[i]] + [x_16[i]]
    b[i] = [b_14[i]] + [b_15[i]] + [b_16[i]]
# print(_x,_b)
ax1.set_xticklabels([9.14,9.15,9.16])
ax1.set_xticks([i+0.5 for i in x[1]])
ax1.bar(x[0],b[0],label=a[0])
ax1.bar(x[1],b[1],label=a[1])
ax1.bar(x[2],b[2],label=a[2])
ax1.bar(x[3],b[3],label=a[3])
ax1.legend(loc='upper left')
ax1.set_xlabel('movie name')
ax1.set_ylabel('box office')
ax1.set_title("three day box office")
plt.show()


# a=[131, 98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115,  99, 136, 126, 134,  95, 138, 117, 111,78, 132, 124, 113, 150, 110, 117,  86,  95, 144, 105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123,  86, 101,  99, 136,123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140,  83, 110, 102,123,107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,115, 146, 137, 116, 103, 144,  83, 123, 111, 110, 111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137,  92,121, 112, 146,  97, 137, 105,  98, 117, 112,  81,  97, 139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110,105, 129, 137, 112, 120, 113, 133, 112,  83,  94, 146, 133, 101,131, 116, 111,  84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]

# fig = plt.figure(figsize=(16,8),dpi=72)
# ax1 = fig.add_subplot(221)

# g_diff = 3
# ax1.hist(a,bins=((max(a)-min(a))//g_diff))
# plt.show()

# interval = [0,5,10,15,20,25,30,35,40,45,60,90]
# width = [5,5,5,5,5,5,5,5,5,15,30,60]
# quantity = [836,2737,3723,3926,3596,1438,3273,642,824,613,215,47]

# arr = [5]
# x1 = []
# for i in range(len(interval)-1):
# 	arr.append(interval[i+1]-interval[i])
# 	x1.append(arr[i]/2)
# print(arr,x1)
# fig = plt.figure(figsize=(16,8),dpi=72)
# ax1 = fig.add_subplot()
# ax1.bar([x+x1[i] for i,x in enumerate(interval)],quantity,width=[x for x in arr])
# ax1.set_xticks([i for i in interval])
# plt.show()

import matplotlib.pyplot as plt
interval = [0,5,10,15,20,25,30,35,40,45,60,90]
width = [5,5,5,5,5,5,5,5,5,15,30,60]
quantity = [836,2737,3723,3926,3596,1438,3273,642,824,613,215,47]

fig = plt.figure(figsize=(16,8),dpi=72)
ax1 = fig.add_subplot()
ax1.bar([width[i]/2+x for i,x in enumerate(interval)],quantity,width=width)
ax1.set_xticks([i for i in interval+[150]])
plt.show()


# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import random
fig = plt.figure(figsize=(10,10),dpi=80)
#规整化布局

# sub1 = fig.add_subplot(331)
# sub2 = fig.add_subplot(332)
# sub2 = fig.add_subplot(333)
# sub3 = fig.add_subplot(223)
# sub4 = fig.add_subplot(224)

#自定义布局
ax1 = fig.add_axes([0.1,0.1,0.5,0.5])
# ax2 = fig.add_axes([0.8,0.8,0.2,0.2])
#
# ax1.spines['bottom'].set_visible(False)

# ax1.set_xticks([1,2,3,4])
# ax1.set_xticklabels(list('abcd'))
# ax1.grid(b=True, axis='y')
# ax1.set_xlabel('x cord', fontsize=20)
ax1.plot([1,2,3,4],[1,2,3,4],label='age')
x = [random.randint(1,10) for _ in range(100)]
y = [random.randint(1,10) for _ in range(100)]
ax1.scatter(x,y)
ax1.legend(loc='upper center')
ax1.set_title('this is a test graph')
plt.show()


