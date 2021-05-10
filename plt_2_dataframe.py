import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import sklearn
import pandas as pd

y=[0.5,9.36,33.6,191,350.19,571,912]
x = np.arange(7)

b = 0
x1=0
x2=0
for i in range(len(x)):
    x1 += (x[i]-np.mean(x))*(y[i]-np.mean(y))
    x2 += (x[i] - np.mean(x)) ** 2

b = x1/x2
a = np.mean(y)-b*np.mean(x)

print(a,b,7*b+a)

plt.scatter(x,y)
plt.plot(x,b*x+a)
plt.show()

boston = datasets.load_boston()
x = boston.data
y = boston.target
df1=pd.DataFrame(x)
df2=pd.DataFrame(y)
line_r = linear_model.LinearRegression()
line_r.fit(df1,df2)
line_r.fit(df1,df2)
print("theta:",line_r.coef_)
print("theta0:",line_r.intercept_)
print(line_r.score(df1,df2))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

data = pd.read_csv('C:/Users/MJZ/Desktop/PoliceKillingsUS.csv')

races = np.sort(data['race'].dropna().unique())

fig, axes = plt.subplots(1, len(races),figsize=(20,6),sharey=True)

amax = int(data['age'].max())
amin = int(data['age'].min())

b = 17
bins = (amax-amin)//b

for i in range(len(races)):
    r = (data[data['race'] == races[i]])
    d = r['age'].sort_index()
    axes[i].hist(d,bins=bins)
    axes[i].set_title(races[i])
    axes[i].set_xticks(range(amin,amax+b,b))
plt.show()

####################

data = pd.read_csv('C:/Users/MJZ/Desktop/lesson8/911.csv')

def diff_m_cate_count():
    data['cate'] = data['title'].apply(lambda x: x.split(':')[0])
    stats = data.groupby('cate').count()
    data['timeStamp'] = pd.to_datetime(data['timeStamp'])
    data.set_index('timeStamp', inplace=True)

    fig = plt.figure(figsize=(16,9),dpi=72)
    ax = fig.add_subplot()

    for gname, gdata in data.groupby('cate'):
        cbm = gdata.resample('M').count()['title']
        x = cbm.index
        y = cbm.values
        plt.plot(range(len(x)), y, label=gname)
    xticks = [ i.strftime('%Y%m%d') for i in x ]
    plt.xticks(range(len(x)), xticks, rotation=45)
    plt.legend()
    plt.show()

diff_m_cate_count()

####################

data=[]
data_name=[]
data.append(pd.read_csv('C:/Users/MJZ/Desktop/lesson8/BeijingPM20100101_20151231.csv'))
data.append(pd.read_csv('C:/Users/MJZ/Desktop/lesson8/ChengduPM20100101_20151231.csv'))
data.append(pd.read_csv('C:/Users/MJZ/Desktop/lesson8/GuangzhouPM20100101_20151231.csv'))
data.append(pd.read_csv('C:/Users/MJZ/Desktop/lesson8/ShanghaiPM20100101_20151231.csv'))
data.append(pd.read_csv('C:/Users/MJZ/Desktop/lesson8/ShenyangPM20100101_20151231.csv'))
data_name=["Beijing","Chengdu","Guangzhou","Shanghai","Shenyang"]

fig, axes = plt.subplots(len(data),1,figsize=(20,6),sharex=True)
for i in range(len(data)):
    data[i]["Time_Stamp"] = pd.to_datetime(data[i].iloc[:,1:4])
    data[i]["PM"] = data[i]["PM_US Post"]
    data[i]["PM"].dropna(inplace=True)
    data[i].set_index("Time_Stamp",inplace=True)

    axes[i].set_title(data_name[i])
    axes[i].set_xticks(range(0,len(data[i].resample("M")["PM"].count().index),5))
    axes[i].set_xticklabels([i.strftime("%Y%m%d") for i in data[i].resample("M")["PM"].count().index[::5]],rotation=60)
    axes[i].plot(range(len(data[i].resample("M")["PM"])),data[i].resample("M").mean()["PM"])
plt.show()