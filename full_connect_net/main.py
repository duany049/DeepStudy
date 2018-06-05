# -*- coding:utf-8 -*-
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data
from gradient_check import eval_numerical_gradient_array
from layers import *
from test import TestSmallData
from full_net import FullConnectNet
from solver import Solver
from features import *
import random

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """ 返回相对误差，加上1e-8是为了防止分母为零 """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def genera_small_data(data):
    x_train, y_train = data['x_train'], data['y_train']
    num_train = 100
    num_val = 100
    x_small_train = x_train[0:num_train]
    x_small_train = np.reshape(x_small_train, (x_small_train.shape[0], -1))
    y_small_train = y_train[0:num_train]
    x_small_val = x_train[num_train:num_train + num_val]
    x_small_val = np.reshape(x_small_val, (x_small_val.shape[0], -1))
    y_small_val = y_train[num_train:num_train + num_val]
    return {
        'x_train': x_small_train,
        'y_train': y_small_train,
        'x_val': x_small_val,
        'y_val': y_small_val
    }


testSmall = TestSmallData()
# testSmall.start_test()

data_dir = '../data_batches_py'
data = load_data(data_dir)
for k, v in data.items():
    print('key: %s - value shape: ' % k, v.shape)

# 先使用小批量数据来验证代码正确性
small_data = genera_small_data(data)
for k, v in small_data.items():
    print('key: %s - value shape: ' % k, v.shape)

# solvers = {}
# for update_rule in ['sgd', 'sgd_moment']:
#     print('running with: ', update_rule)

x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
num_train = 49000
num_val = 1000
x_val = x_train[num_train:num_train + num_val]
y_val = y_train[num_train:num_train + num_val]
x_train = x_train[:num_train]
y_train = y_train[:num_train]

# num_color_bins = 10  # 颜色空间bin的数目
# feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
# X_train_feats = extract_features(x_train, feature_fns, verbose=True)  # extract_features是把梯度方向直方图和颜色空间抽取的特征结合起来，比当个的效果更好
# X_val_feats = extract_features(x_val, feature_fns)
# X_test_feats = extract_features(x_test, feature_fns)

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_val = np.reshape(x_val, (x_val.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
val_train_data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_val': x_val,
    'y_val': y_val
}

# X_train_feats = np.reshape(X_train_feats, (X_train_feats.shape[0], -1))
# X_val_feats = np.reshape(X_val_feats, (X_val_feats.shape[0], -1))
# X_test_feats = np.reshape(X_test_feats, (X_test_feats.shape[0], -1))
# val_train_feature_data = {
#     'x_train': X_train_feats,
#     'y_train': y_train,
#     'x_val': X_val_feats,
#     'y_val': y_val
# }

update_rule = 'sgd_moment'
data = val_train_data
# data = val_train_feature_data
input_dims = data['x_train'].shape[1]

# weight_scale, reg, learning_rate,
# best_acc = -1
# best_solver = None
# w_num = 20
# reg_num = 20
# learning_num = 20
# reg_history = []
# rate_history = []
# acc_history = []
# weight_history = []
# for index_w in range(w_num):
#     weight_scale = 10 ** random.uniform(-3, 1)
#     for index_reg in range(reg_num):
#         reg = 10 ** random.uniform(-6, -3)
#         for index_learning in range(learning_num):
#             learning_rate = 10 ** random.uniform(-2, 2)
#
#             fullModel = FullConnectNet([100, 100, 100, 100, 100, 100, 100, 100, 100, 100], input_dims=input_dims,
#                                        weight_scale=weight_scale, reg=reg,
#                                        use_batchnorm=True)
#             solver = Solver(fullModel, data, num_epochs=250, batch_size=200,
#                             update_rule=update_rule,
#                             optim_config={
#                                 'learning_rate': learning_rate,
#                             },
#                             verbose=True)
#             solver.train()
#             test_acc = solver.check_accuracy(x_test, y_test)
#             print('epoch weight: %s - reg: %s - learning: %s - test acc: %s ' % (
#                 weight_scale, reg, learning_rate, test_acc))
#             if test_acc > best_acc:
#                 best_acc = test_acc
#                 best_solver = solver
#                 print('cur best weight: %s - reg: %s - leaning: %s - acc: %s' % (
#                     weight_scale, reg, learning_rate, test_acc))
#
#             weight_history.append(weight_scale)
#             reg_history.append(reg)
#             rate_history.append(learning_rate)
#             acc_history.append(test_acc)
#
# for it in range(len(reg_history)):
#     print('weight scale: %f - learning rate: %f - reg: %f - acc: %f' % (
#     weight_history[it], rate_history[it], reg_history[it], acc_history[it]))
#
# test_acc = best_solver.check_accuracy(x_test, y_test)
# print('Final acc: ', test_acc)

fullModel = FullConnectNet([30, 30, 30], input_dims=input_dims,
                           weight_scale=5e-2, reg=0,
                           use_batchnorm=True)
solver = Solver(fullModel, data, num_epochs=250, batch_size=200,
                update_rule=update_rule,
                optim_config={
                    'learning_rate': 1e-2,
                },
                verbose=True)
solver.train()
test_acc = solver.check_accuracy(x_test, y_test)
print('Final test acc: ', test_acc)

plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

# plt.subplot(3, 1, 1)
# plt.plot(solver.loss_history, 'o', label=update_rule)
#
# plt.subplot(3, 1, 2)
# plt.plot(solver.train_acc_history, '-o', label=update_rule)
#
# plt.subplot(3, 1, 3)
# plt.plot(solver.val_acc_history, '-o', label=update_rule)

for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
# plt.show()
