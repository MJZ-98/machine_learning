import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

data = pd.read_csv('PoliceKillingsUS.csv')

races = np.sort(data['race'].dropna().unique())


# 场景一: 不同种族中, 逃逸方式分别是如何分布的?
def race_flee():
    fig, axes = plt.subplots(1, len(races), figsize=(20, 6), sharey=True)

    # for ax, race in zip(axes, races):
    #     data[data['race'] == race]['flee'].value_counts().sort_index()\
    #         .plot(kind='bar',ax=ax,title=race)

    for i in range(len(races)):
        r = (data[data['race'] == races[i]])
        d = r['flee'].value_counts().sort_index()
        axes[i].bar(range(len(d.index)), d.values)
        axes[i].set_title(races[i])
        axes[i].set_xticks(range(len(d.index)))
        axes[i].set_xticklabels(d.index, rotation=45)


def flee_age():
    # fig = plt.figure(figsize=(20,6),dpi=72)
    # ax = fig.add_subplot()
    data.groupby('flee')['age'].plot(kind='kde', legend=True, figsize=(20, 6))


def stats_cols():
    print(data.groupby('race').agg({'age': np.median, 'signs_of_mental_illness': np.mean}))


race_flee()
plt.show()
##########################
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

data = pd.read_csv('IMDB-Movie-Data.csv')
genres = set(data['Genre'].str.split((',')).sum())
df_zeros = pd.DataFrame(np.zeros((data.shape[0], len(genres))), columns=genres)
for i in range(data.shape[0]):
    g = data['Genre'][i]
    df_zeros.loc[i, g.split(',')] = 1
genres_counts = df_zeros.sum()

genres_counts = genres_counts.sort_values(ascending=False)
fig = plt.figure(figsize=(16, 9), dpi=72)
plt.bar(range(genres_counts.shape[0]), genres_counts.values)
plt.xticks(range(genres_counts.shape[0]), genres_counts.index, rotation=45)
plt.xlabel('电影类型')
plt.ylabel('电影数量')
plt.title('电影类型-数量一览')
plt.show()

############################

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

data = pd.read_csv('IMDB-Movie-Data.csv')
rating = data['Rating']
runtimes = data['Runtime (Minutes)']

fig = plt.figure(figsize=(16, 9), dpi=72)
ax1 = fig.add_subplot(211)
bins = 10
x = np.linspace(rating.min(), rating.max(), bins + 1).round(1)
ax1.set_xticks(x)
ax1.set_title('电影-评分分布')
ax1.set_xlabel('评分')
ax1.set_ylabel('数量')
ax1.hist(rating, bins=bins)

ax2 = fig.add_subplot(212)
bins = 20
x = np.linspace(runtimes.min(), runtimes.max(), bins + 1).round(1)
ax2.set_xticks(x)
ax2.set_title('电影-时长分布')
ax2.set_xlabel('时长')
ax2.set_ylabel('数量')
ax2.hist(runtimes, density=True, bins=bins)
plt.show()

########################
data = pd.read_csv('starbucks.csv')
fig = plt.figure(figsize=(10, 6))


def rank_cn():
    grouped = data.groupby(['Country', 'State/Province']).count().loc['CN']
    sorted = grouped['Brand'].sort_values(ascending=False)
    sorted.plot(kind='bar', figsize=(20, 5))


def top3():
    grouped = data.groupby(['Country', 'State/Province']).count().loc['CN']
    sorted = grouped['Brand'].sort_values(ascending=False)
    top3 = sorted[:3].index
    top3all = data[(data['Country'] == 'CN') & data['State/Province'].isin(top3)]
    res = top3all.groupby(['State/Province']).head(3)['Street Address']

    print(res)


top3()
plt.show()

##############
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('911.csv')


def firstsplit(x):
    return x.split(":")[0]


def cate_count():
    data['cate'] = data['title'].apply(lambda x: x.split(':')[0])
    stats = data.groupby('cate').count()['title']


def diff_m_cate_count():
    data['cate'] = data['title'].apply(lambda x: x.split(':')[0])
    stats = data.groupby('cate').count()
    data['timeStamp'] = pd.to_datetime(data['timeStamp'])
    data.set_index('timeStamp', inplace=True)

    fig = plt.figure(figsize=(16, 9), dpi=72)
    ax = fig.add_subplot()

    for gname, gdata in data.groupby('cate'):
        cbm = gdata.resample('M').count()['title']
        x = cbm.index
        y = cbm.values
        plt.plot(range(len(x)), y, label=gname)
    xticks = [i.strftime('%Y%m%d') for i in x]
    plt.xticks(range(len(x)), xticks, rotation=45)
    plt.legend()
    plt.show()


diff_m_cate_count()
