import numpy as np
from my_ml import *


class NeuralNetworks_1:
    # input输入层特征数， output输出层预测值个数， hidden隐藏层结点数的数组
    def __init__(self, input_dim, output_dim, hidden_dims, weights_type=0, learning_rate=0.1, type='mse', D=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weights_type = weights_type
        self.type = type
        self.Deltas = D
        self._init_layers()

        self.output_Delta = 0
        self.input_Delta = 0

    # 初始化网络的各层信息，包括各层的权重矩阵  |
    def _init_layers(self):
        if self.weights_type == 0:
            self.input_weights = np.zeros((self.hidden_dims[0], self.input_dim + 1))  # 行是下一层的神经元个数,列是输入层的特征数+1
            self.hidden_weights = []
            for index in range(len(self.hidden_dims) - 1):  # 隐藏层个数-1
                weights = np.zeros((self.hidden_dims[index + 1], self.hidden_dims[index] + 1))
                self.hidden_weights.append(weights)
            self.output_weights = np.zeros((self.output_dim, self.hidden_dims[-1] + 1))
        else:
            self.input_weights = np.random.normal(0, pow(self.input_dim, -0.5),
                                                  (self.hidden_dims[0], self.input_dim + 1))
            self.hidden_weights = []
            for index in range(len(self.hidden_dims) - 1):
                weights = np.random.normal(0, pow(self.hidden_dims[index], -0.5),
                                           (self.hidden_dims[index + 1], self.hidden_dims[index] + 1))
                self.hidden_weights.append(weights)
            self.output_weights = np.random.normal(0, pow(self.hidden_dims[-1], -0.5),
                                                   (self.output_dim, self.hidden_dims[-1] + 1))

        self.hidden_nodes = [None] * len(self.hidden_dims)

    def _forward_propagation(self):

        self.hidden_nodes[0] = logistic_hypothesis(self.input_weights, self.input_nodes)
        # 添加偏置项
        self.hidden_nodes[0] = np.append([1], self.hidden_nodes[0])
        for i in range(1, len(self.hidden_dims)):
            self.hidden_nodes[i] = logistic_hypothesis(self.hidden_weights[i - 1], self.hidden_nodes[i - 1])
            self.hidden_nodes[i] = np.append([1], self.hidden_nodes[i])
        self.output_nodes = logistic_hypothesis(self.output_weights, self.hidden_nodes[-1])

    def _backward_propagation(self):
        if self.type == 'mse':
            # 隐藏层到输出层的权重更新
            last_delta = (self.output_nodes - self.y) * self.output_nodes * (1 - self.output_nodes)
            last_weights = self.output_weights
            new_weights = last_weights - self.learning_rate * last_delta.reshape(-1, 1) * self.hidden_nodes[-1]
            # 隐藏层到隐藏层之间的权重更新
            for i in range(len(self.hidden_dims) - 1, 0, -1):
                last_delta = last_delta @ last_weights[:, 1:] * self.hidden_nodes[i][1:] * (
                            1 - self.hidden_nodes[i][1:])
                last_weights[:, :] = new_weights  # 索引last_weights = self.output_weights，更新output_weights
                last_weights = self.hidden_weights[i - 1]  # 改变引用对象
                new_weights = last_weights - self.learning_rate * (last_delta.reshape(-1, 1) * self.hidden_nodes[i - 1])

            last_delta = last_delta @ last_weights[:, 1:] * self.hidden_nodes[0][1:] * (1 - self.hidden_nodes[0][1:])
            last_weights[:, :] = new_weights
            last_weights = self.input_weights
            self.input_weights = last_weights - self.learning_rate * (last_delta.reshape(-1, 1) * self.input_nodes)

        if self.type == 'ANN':
            output_delta = self.output_nodes - self.y
            last_weights = self.output_weights
            self.Deltas[2] += output_delta.reshape(-1, 1) @ (np.array(self.hidden_nodes[-1]).reshape(-1, 1).T)  # 输出层Δ更新
            last_delta = output_delta
            for i in range(len(self.hidden_dims) - 1, 0, -1):
                last_delta = (last_weights[:, 1:].T @ last_delta.reshape(-1, 1)) * (
                    (self.hidden_nodes[i][1:] * (1 - self.hidden_nodes[i][1:])).reshape(-1, 1))
                last_weights = self.hidden_weights[i - 1]  # 改变引用对象
                self.Deltas[1][i - 1] += last_delta.reshape(-1, 1) @ (
                    np.array(self.hidden_nodes[i - 1]).reshape(-1, 1).T)  # 隐层Δ更新

            hidden_delta = last_delta
            input_delta = last_weights[:, 1:].T @ hidden_delta.reshape(-1, 1) * (
                (self.hidden_nodes[0][1:] * (1 - self.hidden_nodes[0][1:])).reshape(-1, 1))
            last_weights = self.input_weights
            self.Deltas[0] += input_delta.reshape(-1, 1) @ (np.array(self.input_nodes).reshape(-1, 1).T)  # 输入层Δ更新

            self.output_weights -= self.learning_rate * (1 / self.input_dim) * self.Deltas[2]
            for i in range(len(self.hidden_dims) - 1, 0, -1):
                self.hidden_weights[i - 1] -= self.learning_rate * (1 / self.input_dim) * self.Deltas[1][i - 1]
            self.input_weights -= self.learning_rate * (1 / self.input_dim) * self.Deltas[0]

            ##-----------------
            # output_delta = self.output_nodes - self.y

            # hiddden_node_delta = (self.output_weights.T @ output_delta.reshape(-1,1)) *((self.hidden_nodes[0] * ( 1 -self.hidden_nodes[0] )).reshape(-1,1))
            # D_hidden = output_delta.reshape(-1,1) @ (np.array(self.hidden_nodes).reshape(-1,1).T)
            # self.output_Delta  = 1/(self.input_dim) *(self.output_Delta + D_hidden)

            # D_input = (hiddden_node_delta[1:] . reshape(-1,1) * self.input_nodes.reshape(-1,1).T)
            # self.input_Delta = 1/(self.input_dim) *(self.input_Delta + D_input)
            # self.output_weights -= self.learning_rate* self.output_Delta
            # self.input_weights -= self.learning_rate *self.input_Delta
            ##==================

    def init_weights(self, input_weights, output_weights, hidden_weights=None):
        self.input_weights = input_weights
        self.output_weights = output_weights
        if hidden_weights:
            self.hidden_weights = hidden_weights

    def predict(self, x):
        input_nodes = np.append([1], x)
        hidden_nodes = [None] * len(self.hidden_dims)
        hidden_nodes[0] = logistic_hypothesis(self.input_weights, input_nodes)
        hidden_nodes[0] = np.append([1], hidden_nodes[0])
        for i in range(1, len(self.hidden_dims)):
            # hidden_nodes[i] =logistic_hypothesis(self.hidden_weights[i-1],self.hidden_nodes[i-1])
            hidden_nodes[i] = logistic_hypothesis(self.hidden_weights[i - 1], hidden_nodes[i - 1])
            # hidden_nodes[i] = np.append([1],self.hidden_nodes[i])
            hidden_nodes[i] = np.append([1], hidden_nodes[i])
        output_nodes = logistic_hypothesis(self.output_weights, hidden_nodes[-1])
        return output_nodes

    # 训练的目的是得到各层的权重矩阵
    def train(self, x, y):
        self.input_nodes = np.append([1], x)
        self.y = y
        self._forward_propagation()
        self._backward_propagation()
        return self.cost()

    def cost(self):
        return np.sum(1 / 2 * pow(self.y - self.output_nodes, 2))


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
    nn = NeuralNetworks_1(INPUT_COUNT, OUTPUT_COUNT, [10, 20], learning_rate=0.12)
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
