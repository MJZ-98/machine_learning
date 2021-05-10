import matplotlib.pyplot as plt
import numpy as np
import gzip
from timeit import default_timer as timer
from my_ml import *

def read_zip_data(file_path,file_offset):
    with gzip.open(file_path) as f:
        data = np.frombuffer(f.read() , np.uint8 , offset=file_offset)
    return data

def load_data():
    y_train = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/train-labels-idx1-ubyte.gz",8)
    x_train = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/train-images-idx3-ubyte.gz",16)
    x_train = x_train.reshape(len(y_train),28,28)
    y_test = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/t10k-labels-idx1-ubyte.gz",8)
    x_test = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/t10k-images-idx3-ubyte.gz",16)
    x_test = x_test.reshape(len(y_test),28,28)
    return x_train,y_train,x_test,y_test

def visualize_images(images,labels):
    names = 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',\
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    plt.figure(figsize=(10,8))
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(names[labels[i]])
    plt.show()

x_train , y_train , x_test , y_test = load_data()
print(x_train)
# visualize_images(x_trian,y_train)
x_train = x_train.reshape(len(y_train),-1)
x_test = x_test.reshape(len(y_test),-1)
# print(x_train,y_train)

# INPUT_COUNT = x_train.shape[1]
# # HIDDEN_COUNT = 30
# OUTPUT_COUNT = 10
# nn = NeuralNetworks(INPUT_COUNT,OUTPUT_COUNT,[20],weight_type=1,learning_rate=0.12)
#
# tic = timer()
# epoch = 1
# k = epoch
# while epoch:
#     for i in range(len(x_train)):
#         target = y_train[i]
#         x = x_train[i] / 255 * 0.99 + 0.01
#         y = np.zeros(10) + 0.01
#         y[target] = 0.99
#         # print(y)
#         nn.train(x,y)
#     epoch-=1
#     print(k-epoch,"iter's cost:",nn.cost())
#
# score = []
# for i in range(len(x_test)):
#     x = x_test[i] / 255 * 0.99 + 0.01
#     y = y_test[i]
#     pre = nn.predict(x)
#     label = np.argmax(pre)
#     score.append(y==label)
#
# print(np.count_nonzero(score)/len(score))
#
# toc = timer()
# print(toc - tic,'second') # 输出的时间，秒为单位