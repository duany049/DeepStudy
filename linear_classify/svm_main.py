# -*- coding:utf-8 -*-
import random
import numpy as np
import time

from data_utils import load_data
import matplotlib.pyplot as plt
from linear_svm import svm_loss_naive
from linear_svm import svm_loss_vectorized
from linear_svm import grad_check_sparse

"""
分值计算公式      f(xi;W)=W∗x
使用svm损失函数   Li=∑j≠yimax(0,fj−fyi+Δ)
考虑整个数据集的平均损失和正则化后，公式如下：
L=1N∑i∑j≠yi[max(0,f(xi;W)j−f(xi;W)yi+Δ)]+λ∑k∑lW2k
"""

# data_dir_cloba = '/content/drive/trainData/cifar/knn/data_batches_py'
# X_train, y_train, X_test, y_test = load_data(data_dir_cloba)
data_dir = '../data_batches_py'
X_train, y_train, X_test, y_test = load_data(data_dir)

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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
# print ('mean_img shape: ', mean_img.shape)
# print ('mean_img data: ', mean_img[:10])
plt.figure(figsize=(4, 4))
new_mean_img = mean_img.reshape(32, 32, 3)
# print ('new mean img shape: ', new_mean_img.shape)
# print ('new mean img data: ', new_mean_img)
plt.imshow(new_mean_img.astype('uint8'))
# plt.show()

# 然后对数据集图像分别减去均值
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

# 生成一个很小的随机权重矩阵
W = np.random.rand(3073, 10) * 0.0001
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
# grad_check_sparse(f, W, grad)

# time_start = time.time()
# loss_naive, gradient_navie = svm_loss_naive(W, X_dev, y_dev, 5e-6)
# time_end = time.time()
# print ('Naive loss: ', loss_naive, ' use time: ', time_end - time_start)
#
# time_start = time.time()
# loss_vector, gradient_vector = svm_loss_vectorized(W, X_dev, y_dev, 5e-6)
# time_end = time.time()
# print ('Vector loss: ', loss_vector, ' use time: ', time_end - time_start)
# print ('different loss: ', loss_vector - loss_naive)

from linear_classifier import LinearSVM

svm = LinearSVM()
time_start = time.time()
loss_histroy = svm.train(X_train, y_train, learning_rate=1.5e-7, reg=3.25e4, num_iters=1500, batch_size=5000,
                         verbose=True)
time_end = time.time()
print ('train take time: ', time_end - time_start)

# 将损失和循环次数画出来，有利于debug
plt.plot(loss_histroy)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

y_val_pred = svm.predict(X_val)
print ('accuracy: %f' % (np.mean(y_val_pred == y_val)))
