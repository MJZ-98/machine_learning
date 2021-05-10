import numpy as np
from my_ml import *


class NeuralNetworks:
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


    def _backward_propagation(self):
        # hidden--out
        last_delta = (self.output_nodes - self.y)*self.output_nodes*(1-self.output_nodes)
        last_weights = self.output_weights
        new_weights = last_weights - self.learning_rate * (last_delta.reshape(-1,1) * self.hidden_nodes[-1])

        # hidden--hidden
        for i in range(len(self.hidden_dims)-1,0,-1):
            last_delta = last_delta @ last_weights[:,1:] * self.hidden_nodes[i][1:] * (1-self.hidden_nodes[i][1:])
            last_weights[:,:] = new_weights
            last_weights = self.hidden_weights[i-1]
            new_weights = last_weights - self.learning_rate * (last_delta.reshape(-1,1) * self.hidden_nodes[i-1])
        # if len(self.hidden_dims) > 1:
        #     last_weights[:,:] = new_weights
        #     last_weights = self.hidden_weights[0]
        last_delta = last_delta @ last_weights[:,1:] * self.hidden_nodes[0][1:] * (1-self.hidden_nodes[0][1:])
        last_weights[:,:] = new_weights
        last_weights = self.input_weights
        self.input_weights = last_weights - self.learning_rate*(last_delta.reshape(-1,1) * self.input_nodes)

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
        self._backward_propagation()
        return self.cost()
    def cost(self):
        return np.sum(1/2*pow(self.y - self.output_nodes , 2))

if __name__ == '__main__':
    x = np.array([ 0.05, 0.1])
    w1 = np.array([[0.35, 0.15, 0.2], [0.35, 0.25, 0.3]])
    w2 = np.array([[0.6, 0.4, 0.45], [0.6, 0.5, 0.55]])
    y = np.array([0.01, 0.99])
    nn = NeuralNetworks(len(x),len(y), [0] ,weight_type=0, learning_rate= 1)
    nn.init_weight(w1 , w2)
    epoch = 5
    while epoch>0:
        nn.train(x,y)
        epoch-=1
    print(nn.cost(),nn.output_nodes)