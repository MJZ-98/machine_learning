import numpy as np

def linear_hypothesis(T,X):
    return X@T

def linear_cost(T,X,Y):
    m = X.shape[0]
    return np.sum((linear_hypothesis(T,X)-Y)**2)/(2*m)
