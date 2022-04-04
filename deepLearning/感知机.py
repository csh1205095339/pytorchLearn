"""
    感知机接受多个信号输出一个信号
    将多个信号{x1, x2, ...} 分别乘以固定的权重 {w1, w2, ...} 当信号乘以权重的和超过阈值时 神经元（感知机）才会被激活
    y = 0 (w1x1+w2x2+...) <= 阈值
    y = 1 (w1x1+w2x2+...) > 阈值
"""
import numpy as np
import matplotlib.pylab as plt
# # 与门感知机
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1


# # 与非门感知机
# def NAND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1 * w1 + x2 * w2
#     if tmp <= theta:
#         return 1
#     elif tmp > theta:
#         return 0


# 导入 w权重 和 b偏置 与门感知机例子
def AND(x1, x2):
    x = np.array([x1, x2])  # 输入
    w = np.array([0.5, 0.5])  # 权重
    b = -0.7  # 偏置
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


# 导入 w权重 和 b偏置 与门感知机例子
def NAND(x1, x2):
    x = np.array([x1, x2])  # 输入
    w = np.array([-0.5, -0.5])  # 权重
    b = 0.7  # 偏置
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 异或感知机
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# 可以接收数组的阶跃函数
# y = 0  x <= 0
# y = 1  x > 0
def step_function(x):
    y = x > 0
    return y.astype(np.int64)


# sigmoid 函数的实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU  函数的实现
def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def init_network():
    network = {'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), 'b1': np.array([0.1, 0.2, 0.3]),
               'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), 'b2': np.array([0.1, 0.2]),
               'W3': np.array([[0.1, 0.3], [0.2, 0.4]]), 'b3': np.array([0.1, 0.2])}

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

# 分类问题一般用softmax函数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y



if __name__ == "__main__":
    x = np.array([1.0, 0.5])
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)