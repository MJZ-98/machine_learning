from my_ml.linear_regress import *
from my_ml.logistic_regression import *
def addBias(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    return np.hstack((ones,X))

Algorithms = {
    'Linear':[linear_hypothesis,linear_cost],
    'Logistic':[logistic_hypothesis,logistic_cost]
}

def gradient_descend(iters,alpha,X,Y,algo='Linear',L = 0):
    hypothesis,cost = Algorithms[algo]
    costs = []
    m = X.shape[0]
    X = addBias(X)
    T = np.zeros(X.shape[1])
    for i in range(iters):
        costs.append(cost(T,X,Y))
        if L == 0:
            T = T - (alpha/m)*((hypothesis(T,X)-Y)@X)
        else:
            penalty = np.append([0],(L*T)[1:])
            T = T - (alpha / m) * ((hypothesis(T, X) - Y) @ X + penalty )
    return T,costs

def normal_eq(X,y):
    X = addBias(X)
    return np.linalg.inv(X.T@y)@X.T@y

if __name__ == '__main__':
    print('test')