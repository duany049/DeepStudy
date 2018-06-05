# usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform


def loadPick(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
        # raise ValueError("invalid python version: {}".format(version))


def load_data_batch(filename):
    with open(filename, 'rb') as f:
        datadict = loadPick(f)
        data = datadict['data']
        labels = datadict['labels']
        # 存在文本中的数据是(10000, 3072)
        print('before reshape and transpose data shape: ', data.shape)
        # 取出来之后转换成图片本来的像素存储样式(10000, 32, 32, 3)
        data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        print('after reshape and transpose data shape: ', data.shape)
        labels = np.array(labels)
        return data, labels


def load_data(ROOT):
    # 加载所有数据
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        print('filename: %s' % f)
        X, Y = load_data_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_data_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_special_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    # 从磁盘加载数据并且做数据的归一化处理

    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_data(cifar10_dir)

    # 截取子数据，(一会使用全部数据)
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 归一化数据：剪去平均值
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 把数据存入字典中返回
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
