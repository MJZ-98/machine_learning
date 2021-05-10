import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

df = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/iris_3.csv",sep=',',header=0)
df.loc[df.Species=='setosa','Species'] = 1
df.loc[df.Species=='versicolor','Species'] = 2
df.loc[df.Species=='virginica','Species'] = 3

data = df.values
print(data)
train_set,train_target,test_set,test_target = train_test_split(data[:,:-1],data[:,-1])

scaler = StandardScaler()
scaler.fit_transform(train_set)
scaler.transform(test_set)

# -------
feature_pairs = []
columns = df.columns[:-1].to_list()
for i in range(len(columns)-1):
    others = columns[i+1:]
    for j in others:
        feature_pairs.append((columns[i],j))

fig , axes = plt.subplots(2,3,figsize=(10,6))
species_name = ['setosa','versicolor','virginica']
for i , pair in enumerate(feature_pairs):
    x,y = pair
    ax = axes[divmod(i,3)]
    for s in range(1,4):
        species = df[df.Species == s]
        ax.scatter(species[x],species[y],label=species_name[s-1])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
plt.show()

iters = 2000
alpha = 0.8
adjust = np.array([1,0,1,1,1])

setosa_target = np.array([1 if x==1 else 0  for x in train_target])
T,setosa_costs = gradient_descend(iters,alpha,train_set,setosa_target,'Logistic')
predicts1 = logistic_hypothesis(T,addBias(test_set))

versicolor_target = np.array([1 if x==2 else 0  for x in train_target])
T,versicolor_costs = gradient_descend(iters,alpha,train_set,versicolor_target,'Logistic')
predicts2 = logistic_hypothesis(T,addBias(test_set))

virginica_target = np.array([1 if x==3 else 0  for x in train_target])
T,virginica_costs = gradient_descend(iters,alpha,train_set,virginica_target,'Logistic')
predicts3 = logistic_hypothesis(T,addBias(test_set))

plt.plot(range(len(setosa_costs)),setosa_costs)
plt.plot(range(len(versicolor_costs)),versicolor_costs)
plt.plot(range(len(virginica_costs)),virginica_costs)
plt.legend(("setosa",'versicolor','virginica'))
plt.xlabel("iters")
plt.ylabel("costs")
plt.show()

df = np.vstack((np.vstack((predicts1,predicts2)),predicts3))
predicts = [i+1 for i in np.argmax(df,axis=0)]
print('target:',test_target)
print('predict:',[i+1 for i in np.argmax(df,axis=0)])
acc = (len(predicts)-np.count_nonzero(predicts-test_target))/len(predicts)
print('acc:',acc)