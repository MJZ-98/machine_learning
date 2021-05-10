import numpy as np
from my_ml import *


class NeuralNetworks_wu:
    # input_dim 输入层特征个数，output_dim输出层预测值个数，hidden_dim隐藏层节点数的数组
    def __init__(self,input_dim, output_dim , hidden_dims ,weight_type, learning_rate= 0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weight_type = weight_type
        self._init_layers()

    # 初始化网络各层信息，包括各层权重矩阵
    def _init_layers(self):
        # 输出的权重, 行为第一个隐藏层的神经元个数, 列为输入特征数+1(1是偏置)
        if self.weight_type == 0:
            self.input_weights = np.zeros((self.hidden_dims[0],self.input_dim + 1))
            self.hidden_weights = []
            for index in range(len(self.hidden_dims)-1):
                weights = np.zeros((self.hidden_dims[index + 1],self.hidden_dims[index] + 1))
                self.hidden_weights.append(weights)
            self.output_weights = np.zeros((self.output_dim,self.hidden_dims[-1]+1))
        else:
            self.input_weights = np.random.normal(0, pow(self.input_dim,-0.5),(self.hidden_dims[0],self.input_dim+1))
            self.hidden_weights = []
            for index in range(len(self.hidden_dims)-1):
                weights = np.random.normal(0,pow(self.hidden_dims[index],-0.5),(self.hidden_dims[index+1],self.hidden_dims[index]+1))
                self.hidden_weights.append(weights)
            self.output_weights = np.random.normal(0,pow(self.hidden_dims[-1],-0.5),(self.output_dim,self.hidden_dims[-1]+1))
        self.hidden_nodes = [None]*len(self.hidden_dims)


    def _forward_propagation(self):
        self.hidden_nodes[0] = logistic_hypothesis(self.input_weights,self.input_nodes)
        self.hidden_nodes[0] = np.append([1],self.hidden_nodes[0])
        for i in range(1,len(self.hidden_dims)):
            self.hidden_nodes[i] = logistic_hypothesis(self.hidden_weights[i-1],self.hidden_nodes[i-1])
            self.hidden_nodes[i] = np.append([1],self.hidden_nodes[i])
        self.output_nodes = logistic_hypothesis(self.output_weights,self.hidden_nodes[-1])


    def _backward_propagation(self,pen):
        a = [None] * (len(self.hidden_dims)+2)
        delta = [None] * (len(self.hidden_dims) + 1)

        a[0] = self.input_nodes
        for i in range(1,len(self.hidden_dims)+1):
            a[i] = self.hidden_nodes[i-1]
        a[-1] = self.output_nodes

        delta[-1] = a[-1] - self.y
        Delta = delta[-1].T * a[-1]
        delta[-2] = (self.output_weights.T * delta[-1].T).T * a[-2] * (1 - a[-2])
        Delta = Delta + delta[-2].T * a[-1]
        for i in range(len(self.hidden_dims)-2,-1,-1):
            delta[i] = np.dot(self.hidden_weights[i].T , delta[i + 1][:,1:].T).T * a[i+1] * (1 - a[i+1])
            Delta = Delta + np.dot(delta[i] , a[i+1])

        # theta = np.append(self.input_weights,self.hidden_weights)
        # theta = np.append(theta, self.output_weights)
        # print(theta)
        D1 = 1/self.y.shape[0] * Delta # + pen * theta[:,1:]
        D2 = 1/self.y.shape[0] * Delta
        self.input_weights[:,1:] = self.input_weights[:,1:] - self.learning_rate * D1
        self.input_weights[:,0] = self.input_weights[:,0] - self.learning_rate * D2
        for i in range(len(self.hidden_dims)):
            self.hidden_weights[i][:,1:] = self.hidden_weights[i][:,1:] - self.learning_rate * D1
            self.hidden_weights[i][:,0] = self.hidden_weights[i][:,0] - self.learning_rate * D2
        self.output_weights[:, 1:] = self.output_weights - self.learning_rate * D1
        self.output_weights[:, 0] = self.output_weights - self.learning_rate * D2

    def init_weight(self,input_weight,output_weight,hidden_weight = None):
        self.input_weights = input_weight
        self.output_weights = output_weight
        if(hidden_weight):
            self.hidden_weights = hidden_weight

    def predict(self,x):
        input_nodes = np.append([1],x)
        hidden_nodes = [None]*len(self.hidden_dims)
        hidden_nodes[0] = logistic_hypothesis(self.input_weights,input_nodes)
        hidden_nodes[0] = np.append([1], hidden_nodes[0])
        for i in range(1,len(self.hidden_dims)):
            hidden_nodes[i] = logistic_hypothesis(self.hidden_weights[i - 1],hidden_nodes[i - 1])
            hidden_nodes[i] = np.append([1], hidden_nodes[i])
        output_nodes = logistic_hypothesis(self.output_weights, hidden_nodes[-1])
        return output_nodes

    def train(self,x,y):
        self.input_nodes = np.append([1],x)
        self.y = y
        self._forward_propagation()
        self._backward_propagation(0.1)
        return self.cost()
    def cost(self):
        return np.sum(1/2*pow(self.y - self.output_nodes , 2))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import gzip
    from timeit import default_timer as timer
    from my_ml import *

    def read_zip_data(file_path, file_offset):
        with gzip.open(file_path) as f:
            data = np.frombuffer(f.read(), np.uint8, offset=file_offset)
        return data

    def load_data():
        y_train = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/train-labels-idx1-ubyte.gz", 8)
        x_train = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/train-images-idx3-ubyte.gz", 16)
        x_train = x_train.reshape(len(y_train), 28, 28)
        y_test = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/t10k-labels-idx1-ubyte.gz", 8)
        x_test = read_zip_data("C:/Users/MJZ/Desktop/dataset/fashion mnist/t10k-images-idx3-ubyte.gz", 16)
        x_test = x_test.reshape(len(y_test), 28, 28)
        return x_train, y_train, x_test, y_test

    def visualize_images(images, labels):
        names = 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        plt.figure(figsize=(10, 8))
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images[i], cmap=plt.cm.binary)
            plt.xlabel(names[labels[i]])
        plt.show()

    x_train, y_train, x_test, y_test = load_data()
    # visualize_images(x_trian,y_train)
    x_train = x_train.reshape(len(y_train), -1)
    x_test = x_test.reshape(len(y_test), -1)

    INPUT_COUNT = x_train.shape[1]
    # HIDDEN_COUNT = 30
    OUTPUT_COUNT = 10
    nn = NeuralNetworks_wu(INPUT_COUNT, OUTPUT_COUNT, [10,20], weight_type=1, learning_rate=0.12)
    # print(np.array(nn.hidden_weights[0]).shape,np.array(nn.output_weights.shape))
    tic = timer()
    epoch = 1
    k = epoch
    while epoch:
        for i in range(len(x_train)):
            target = y_train[i]
            x = x_train[i] / 255 * 0.99 + 0.01
            y = np.zeros(10) + 0.01
            y[target] = 0.99
            nn.train(x, y)
        epoch -= 1
        print(k - epoch, "iter's cost:", nn.cost())

    score = []
    for i in range(len(x_test)):
        x = x_test[i] / 255 * 0.99 + 0.01
        y = y_test[i]
        pre = nn.predict(x)
        label = np.argmax(pre)
        score.append(y == label)

    print(np.count_nonzero(score) / len(score))

    toc = timer()
    print(toc - tic, 'second')  # 输出的时间，秒为单位