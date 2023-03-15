import pandas as pd
import numpy as np


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        """
          损失函数，预测与实际相差的平方和
        """
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost


def loadData():
    # 读入训练数据
    datafile = './data/price.csv'
    data = pd.read_csv(datafile)
    np_arr = data.values

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(np_arr.shape[0] * ratio)
    training_data = np_arr[:offset]
    # 计算训练集的最大值，最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)
    # 获取列名
    colnums = data.columns.values

    # 对数据进行归一化处理
    for i in range(len(colnums)):
        np_arr[:, i] = (np_arr[:, i] - minimums[i]) / \
            (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = np_arr[:offset]
    test_data = np_arr[offset:]
    return training_data, test_data


# 输入层，加权
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, -0.5, 0.0]
w = np.array(w).reshape([14, 1])

# 获取数据
training_data, test_data = loadData()
# 价格不要
x = training_data[:, :-1]
# 只要价格
y = training_data[:, -1:]

net = Network(13)

x1 = x[0:3]
y1 = y[0:3]
z = net.forward(x1)
print('predict: ', z)
loss = net.loss(z, y1)
print('loss:', loss)
