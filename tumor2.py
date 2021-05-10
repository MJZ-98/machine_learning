import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_ml import *

data = pd.read_csv("C:/Users/MJZ/Desktop/something/machine learn/dataset/breast-cancer-wisconsin.csv",sep=',',
                 header=None,names=['Sample code number','Clump Thickness','Uniformity of Cell Size',
                                    'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
print(data.info())
data = data.replace('?',np.nan)
data = data.dropna()
data = data.iloc[:,1:]
data['Bare Nuclei'] = data['Bare Nuclei'].apply(lambda x:int(x))
data['Class'] = data['Class'].apply(lambda x:0 if x==2 else 1)
feature = data.iloc[:,:-1].values
target =  data.iloc[:,-1].values

train_feature,train_target,test_feature,test_target = train_test_split(feature,target)

scaler = StandardScaler()
scaler.fit_transform(train_feature)
scaler.transform(test_feature)

T, cost = gradient_descend(1000,0.01,train_feature,train_target,'Logistic')

predicts = logistic_hypothesis(T,addBias(test_feature))
predicts = np.array([1 if x>=0.5 else 0 for x in predicts])
print(predicts)

TP = np.count_nonzero(predicts[np.argwhere(test_target == 1)] == 1)
FP = np.count_nonzero(predicts[np.argwhere(test_target == 0)] == 1)
TN = np.count_nonzero(predicts[np.argwhere(test_target == 0)] == 0)
FN = np.count_nonzero(predicts[np.argwhere(test_target == 1)] == 0)

acc = TP/(TP+FP)
print('acc:',acc)
recall = TP/(TP+FN)
print('recall:',recall)