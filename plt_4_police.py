import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

data = pd.read_csv('C:/Users/MJZ/Desktop/something/machine learn/dataset/PoliceKillingsUS.csv')

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
