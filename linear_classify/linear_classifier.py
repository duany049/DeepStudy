# -*-coding:utf-8-*-
from __future__ import print_function

import numpy as np
from linear_svm import *
from softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, method=0):
        """
        使用随机梯度下降来训练
        :param X:
        :param y:
        :param learning_rate:  学习率
        :param reg:            正则化强度
        :param num_iters: 优化时训练的步数
        :param batch_size:  每一步使用的训练样本数
        :param verbose: 是否需要打印优化的过程
        :return:
        """

        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # 假设y的取值为0....k-1
        if self.W is None:
            # 初始化一个极小的W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 使用随机梯度来优化W
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            batch_index = np.random.choice(num_train, batch_size)  # 有放回采样的训练速度比无放回采样的训练速度要快
            X_batch = X[batch_index, :]
            y_batch = y[batch_index]

            loss, grad = self.loss(X_batch, y_batch, reg, method)
            loss_history.append(loss)
            self.W -= learning_rate * grad  # 使用梯度和学习率更新权重
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        num_class = self.W.shape[1]
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg, method):
        if method == 0:
            return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
        elif method == 1:
            return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
        else:
            raise TypeError('not this method')


class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg, method):
        if method == 0:
            return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
        elif method == 1:
            return softmax_loss_naive(self.W, X_batch, y_batch, reg)
        else:
            raise TypeError('not this method')
