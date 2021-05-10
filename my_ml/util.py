import numpy as np

def train_test_split(feature,target,train_ratio=0.7):
    data_len = feature.shape[0]
    sep = int(data_len*train_ratio)
    indices = np.random.permutation(data_len)
    train_indices = indices[:sep]
    test_indices = indices[sep:]
    return feature[train_indices,:],target[train_indices],\
           feature[test_indices,:],target[test_indices]

class StandardScaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self,data):
        self._mean = data.mean()
        self._std = data.std()
        data[:,:] = (data-self._mean)/self._std

    def transform(self,data):
        data[:, :] = (data - self._mean) / self._std