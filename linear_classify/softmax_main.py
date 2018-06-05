# -*- coding:utf-8 -*-
"""
分值计算公式      f(xi;W)=W∗x
第i个样本损失函数   Li=−log(efyi/∑jefj)
类被正确分类的概率   P(yi|xi;W)=efyi/∑jefj
损失函数对权值矩阵W求导得   Xi.T.dot(P)  (当前类为正确类m时：P = Pm     否则：P = Pm - 1)

注意：实际操作中，efi常因为指数太大出现数值爆炸问题，两个数值非常大的数相除会出现数值不稳定的情况，
因此我们一般分子分母同时乘以一个负数C，efyi/∑jefj=CefyiC/∑jefj=efyi+logC/∑jefj+logC
实际操作中logC一般取fi中的最大值的负数，所以fi+logC <= 0,放在e的指数上可以保证分子分布都在0~1之间
"""
import random
from data_utils import load_data
import numpy as np
import matplotlib.pyplot as plt
from softmax import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data_dir = '../data_batches_py'
X_train, y_train, X_test, y_test = load_data(data_dir)

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# 从训练数据集中随机抽取一部分作为开发数据集
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

print ('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

# 预处理，减去图像平均值
mean_img = np.mean(X_train, axis=0)
plt.figure(figsize=(4, 4))
new_mean_img = mean_img.reshape(32, 32, 3)
plt.imshow(new_mean_img.astype('uint8'))
X_train -= mean_img
X_val -= mean_img
X_test -= mean_img
X_dev -= mean_img

# 在训练数据中添加一列1，用来和W中的偏置向量相乘，这样就省去了b的书写
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

W = np.random.randn(3073, 10) * 0.0001
loss, gradient = softmax_loss_naive(W, X_dev, y_dev, 0.0)

print ('loss: %f' % loss)
# 之所以觉得这个地方获取的loss值会接近-np.log(0.1)是因为权值*0.0001后接近0，然后以此为指数得到的值接近1，共有十个类别大家都等概率，所以p接近0.1，故预测loss为-np.log(0.1)
print ('sanity check: %f' % (-np.log(0.1)))

import time

time_start = time.time()
loss_naive, gradient_naive = softmax_loss_naive(W, X_train, y_train, 5e-6)
time_end = time.time()
print ('loss naive take %f seconds - and loss is: %f' % (time_end - time_start, loss_naive))

time_start = time.time()
loss_vector, gradient_vector = softmax_loss_vectorized(W, X_train, y_train, 5e-6)
time_end = time.time()
print ('loss vector take %f seconds - and loss is: %f' % (time_end - time_start, loss_vector))

gradient_diff = np.linalg.norm(gradient_naive - gradient_vector, ord='fro')
print ('lost different: ', np.abs(loss_vector - loss_naive))
print ('gradient different: ', gradient_diff)

from linear_classifier import Softmax

softmax = Softmax()
softmax.train(X_train, y_train, learning_rate=1.67e-8, reg=1e-2, num_iters=1500, verbose=True, method=0)
y_test_pred = softmax.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
print ('final test accuracy: ', test_accuracy)
