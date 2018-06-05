# -*- coding:utf-8 -*-
import numpy as np


def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    dW_each = np.zeros_like(W)
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)  # 样本数*类别数   分值
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
    # 计算对数概率  prob.shape=N*10  每一行与一个样本相对应  每一行的概率和为1
    # 其中 f_max 是每行的最大值，exp(x)中x的值过大而出现数值不稳定问题
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0  # 每行只有正确的类别处为1，其余为0
    # 遍历然后根据loss公式和loss函数对于W的偏导值进行操作
    for i in range(num_train):
        for j in range(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
        dW += dW_each
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train, dim = X.shape

    f = X.dot(W)
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))  # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0

    # 计算损失  y_trueClass是N*C维度  np.log(prob)也是N*C的维度
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)

    # 计算损失  X.T = (D*N)  y_truclass-prob = (N*C)
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W
    return loss, dW
