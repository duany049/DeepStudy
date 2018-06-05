# usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ 使用L2判断距离的knn分类器 """

    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X, k=1, num_loops=0):
        """ 预测图片分类，根据参数使用不同的方法进行距离对比 """

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X_test):
        """使用两层循环来进行距离对比"""

        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j, :] - X_test[i, :])))

        print ('two loops dist shape: ', dists.shape)
        return dists

    def compute_distances_one_loop(self, X_test):
        """使用一层循环来进行距离对比"""

        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            # 利用广播的性质来比较所有的train样本
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X_test[i, :]), axis=1))

        print ('one loops dist shape: ', dists.shape)
        return dists

    def compute_distances_no_loops(self, X):
        """完全矢量化来进行距离对比"""

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.multiply(np.dot(X, self.X_train.T), -2)
        sq1 = np.sum(np.square(X), axis=1, keepdims=True)
        sq2 = np.sum(np.square(self.X_train), axis=1)
        dists = np.add(dists, sq1)
        dists = np.add(dists, sq2)
        dists = np.sqrt(dists)
        return dists

    def predict_labels(self, dists, k=1):
        """通过距离矩阵预测每个测试样本的类别"""

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            closest_y = []
            # np.sort是对数组进行从小到大排序，然后返回索引值
            closest_y = self.Y_train[np.argsort(dists[i])[:k]]
            # np.bincount是返回一个数组，数组中元素分别表示0到最大值之间的数目
            # np.argmax返回最大值的索引值
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
