import numpy as np
from my_ml.linear_regress import *

def sigmoid(Z):
    return  1/(1+np.exp(-Z))

def logistic_hypothesis(t,X):
    return sigmoid(X@t.T)

def logistic_cost(t,X,y,L=0):
    penalty = 0
    m = X.shape[0]
    if L != 0:
        penalty = L/(2*m)*np.sum((t**2)[1:])
    first = y@np.log(logistic_hypothesis(t,X))
    second = (1-y)@np.log(1-logistic_hypothesis(t,X))
    return -(first+second)/(X.shape[0]) + penalty

