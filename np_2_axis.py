import numpy as np
from functools import reduce

a = np.random.randint(0,5,24).reshape(2,3,4)
print(a)
a = a != 0

print('a:',a,'\n')
a1 = reduce(np.logical_and,a)
print('\na1:',a1,'\n',a.all(0))
a2 = [reduce(np.logical_and,i) for i in a]
print('\na2:',a2,'\n',a.all(1))
a3 =[[reduce(np.logical_and,j) for j in i ]for i in a]
j = [i for i in a]
print('\na3:',a3,'\n',a.all(2))
# print(a,"\n",a1,'\n',a2,'\n',a3)

####################

arr = np.arange(10)
print("1、构造一个一维向量的ndarray数组arr:\n",arr)
print('2、找出arr中所有的奇数:\n',arr[1::2])
print('3、计算arr中的元素的平方:\n',arr**2)

arr2 = np.arange(12).reshape(3,4)
print('4、构造一个(3,4)的二维数组arr2:\n',arr2)

arr2 = arr2.reshape(3,4,1)
print('5、怎样求arr2的每个元素与arr中的所有元素相乘的和:\n',(arr2*arr).sum(2))

a = np.arange(1,6)
b = np.arange(5,0,-1)
arr2 = arr2.reshape(3,4,1)
print('6、使用矩阵运算的方式求y = ax + b,  x 为上述的矩阵arr2， 创建a 为[1,2,3,4,5]的向量，b为[5,4,3,2,1]的向量\na:',a,'\nb',b,'\n运算结果为:\n',(a*arr2+b).sum(2))

##########################
#讨论轴运算
import numpy as np
from numpy.random import  randint

homes = np.arange(15,19)
import numpy as wdm
#居委会王大妈来统计全部人的平均年龄
wdm.mean(homes)

#5年过去了，都长大了
homes += 5
#想成家了，单间变成套房
homes = homes.reshape(homes.shape[0],-1)

#结婚了
homes = np.append(homes, homes-randint(1,5,homes.shape), 1)
#王大妈又来统计年龄了
wdm.mean(homes) #所有人的平均年龄
wdm.mean(homes, 0)#统计所有家庭中丈夫和妻子的平均年龄
wdm.mean(homes,1)#统计每一个家庭里人员的平均年龄

#又过了5年，每一家人都有了2个小孩
homes+=5
np.append(homes, randint(1,5,(homes.shape[0],2)), axis=1)
#王大妈还是会来，统计所有人的平均年龄，丈夫和妻子以及小孩的平均年龄，每一家人的平均年龄
wdm.mean(homes) #所有人的平均年龄
wdm.mean(homes,1)#统计每一个家庭里人员的平均年龄
#统计所有家庭中丈夫和妻子以及小孩的平均年龄
wdm.hstack((wdm.mean(homes,0)[:2], wdm.mean(wdm.mean(homes,0)[2:])))

#就这样含辛茹苦20年， 小孩长大了，父母变老了
homes += 20
#未完待续...

################################

#讨论轴运算
'''
1、由左侧数组转化成一个布尔数组，0转成False, 非0转成True
2、使用函数np.logical_and验证轴运算函数all()
'''

import numpy as np
from functools import reduce

def dis_axes():
    a = np.random.randint(0,5,24).reshape(2,3,4)
    a.sum(0)
    #等价于下式
    np.add(a[0],a[1])
    a.sum(1)
    #等价于下式
    [ reduce(np.add,i) for i in a]
    a.sum(2)
    #等价于下式
    [[reduce(np.add,j) for j in  i] for i in a]