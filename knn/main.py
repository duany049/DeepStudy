# usr/bin/env python
# -*- coding:utf-8 -*-
# from __future__ import print_function

import random
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
from data_utils import load_data
import KNearestNeighbor


def run_time(func, args):
    # 使用装饰器来打印执行函数所需要的时间
    import time
    before = time.time()
    func(args)
    after = time.time()
    return after - before


# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data_dir_cloba = '../data_batches_py'
X_train, y_train, X_test, y_test = load_data(data_dir_cloba)
# data_dir = '../data_batches_py'
# X_train, y_train, X_test, y_test = load_data(data_dir)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# 展示训练数据中每个类型随机的七张图片用来观摩
classnames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_class = len(classnames)
simple_class_per = 7
for y, cls in enumerate(classnames):
    tem_y_train = y_train == y
    print('tem_y_train shape: ', tem_y_train.shape)
    # 返回矩阵中非0元素的位置,即所有类型为y的索引的集合
    idxs = np.flatnonzero(tem_y_train)
    # 即从所有y的类型中，随机抽取7个类型用来展示
    idxs = np.random.choice(idxs, simple_class_per, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_class + y + 1
        plt.subplot(simple_class_per, num_class, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
# 仅供不了解图片内容者查看
# plt.show()

"""
# 为了更高效，我们取子集来进行训练，暂时这么做，一会我用全部数据再试试
num_training = 5000
mark = list(range(num_training))
# 训练数据的前5000个
X_train = X_train[mark]
y_train = y_train[mark]

num_test = 500
mark = list(range(num_test))
X_test = X_test[mark]
y_test = y_test[mark]
"""

num_test = X_test.shape[0]
# 将图像数据转置成二维的
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

knn_classifier = KNearestNeighbor.KNearestNeighbor()
knn_classifier.train(X_train, y_train)

# y_test_pred = knn_classifier.predict(X_test, 1, 2)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('get %f / %f ===> accuracy is: %f' % (num_correct, num_test, accuracy))
#
# y_test_pred_k_7 = knn_classifier.predict(X_test, 7, 2)
#
# num_correct = np.sum(y_test_pred_k_7 == y_test)
# accuracy = float(num_correct) / num_test
# print('k7 get %f / %f ===> accuracy is: %f' % (num_correct, num_test, accuracy))
#
# y_test_one_pred_k7 = knn_classifier.predict(X_test, 7, 1)
# num_correct = np.sum(y_test_one_pred_k7 == y_test)
# accuracy = float(num_correct) / num_test
# print('k7 one loop get %f / %f ===> accuracy is: %f' % (num_correct, num_test, accuracy))

# y_test_none_pred_k7 = knn_classifier.predict(X_test, 7, 0)
# num_correct = np.sum(y_test_none_pred_k7 == y_test)
# accuracy = float(num_correct) / num_test
# print('k7 none loop get %f / %f ===> accuracy is: %f' % (num_correct, num_test, accuracy))

# two_loop_time = run_time(knn_classifier.compute_distances_two_loops, X_test)
# one_loop_time = run_time(knn_classifier.compute_distances_one_loop, X_test)
none_loop_time = run_time(knn_classifier.compute_distances_no_loops, X_test)
# print ('two loop use %f seconds' % two_loop_time)
# print ('one loop use %f seconds' % one_loop_time)
print ('none loop use %f seconds' % none_loop_time)
