from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class CNN(Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # 输入到隐层 1
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # 池化层 1
        self.pool1 = MaxPool2d((2,2), stride=(2,2))
        # 隐层 2
        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 池化层 2
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        # 全连接层
        self.hidden3 = Linear(5*5*32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # 输出层
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)
 
    # 前向传播
    def forward(self, X):
        # 输入到隐层 1
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # 隐层 2
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # 扁平化
        X = X.view(-1, 4*4*50)
        # 隐层 3
        X = self.hidden3(X)
        X = self.act3(X)
        # 输出层
        X = self.hidden4(X)
        X = self.act4(X)
        return X