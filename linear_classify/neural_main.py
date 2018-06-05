# -*- coding:utf-8 -*-
# from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from neural_net import *
from data_utils import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data():
    np.random.seed(1)
    x = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return x, y


"""
    策略：先使用极小一部分数据验证模型是否有异常，如果确认没有异常，再使用大规模数据进行训练
"""
ver_neural = init_toy_model()
ver_X_data, ver_y_data = init_toy_data()
scores = ver_neural.loss(ver_X_data)
# print('Your scores:', scores)
correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])
# print('correct scores:', correct_scores)
# print ('shape: ', scores.shape)
# print ( 'shape: ', correct_scores.shape)
# 不同值应该非常小，我们得到的值是-6.762279500249768e-09 < 1e-7，可以认为模型在获取分数实现上没有问题
print('Difference between your scores and correct scores:', np.sum(scores - correct_scores))

loss, _ = ver_neural.loss(ver_X_data, ver_y_data, reg=5e-2)
correct_loss = 1.30378789133
# 通过小数据集验证模型在前向传播中是否有问题，此处defferent极小，验证前向传播没问题
print('Difference between your loss and correct loss:', np.sum(np.abs(loss - correct_loss)))

from gradient_check import eval_numerical_gradient

loss, gradient = ver_neural.loss(ver_X_data, ver_y_data, reg=5e-2)
for param_name in gradient:
    f = lambda W: ver_neural.loss(ver_X_data, ver_y_data, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, ver_neural.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, gradient[param_name])))

ver_states = ver_neural.train(ver_X_data, ver_y_data, ver_X_data, ver_y_data, learning_rate=1e-1, reg=5e-6,
                              num_iters=100, verbose=False)
# 在极小数据集上loss几乎为0，准确率百分之百，可以认为模型代码没有问题
print ('Final train loss: ', ver_states['loss_histroy'][-1])

# 画出迭代过程的损失值变化过程的图像
plt.plot(ver_states['loss_histroy'])
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.title('Training loss histroy')
# plt.show()

"""
    上面用极小量数据验证了正确性，下面使用大数据进行正式训练
"""
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

mean_imag = np.mean(X_train, axis=0)
X_train -= mean_imag
X_val -= mean_imag
X_test -= mean_imag

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

print ('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

input_size = X_train.shape[1]
hidden_size = 50
num_classes = 10
neural_model = TwoLayerNet(input_size, hidden_size, num_classes)
states = neural_model.train(X_train, y_train, X_val, y_val, learning_rate=1e-4, learning_rate_decay=0.95, reg=0.25,
                            num_iters=10000, batch_size=200, verbose=True)
y_val_pred = neural_model.predict(X_val)
val_acc = np.mean(y_val == y_val_pred)
y_train_pred = neural_model.predict(X_train)
train_acc = np.mean(y_train == y_train_pred)
y_test_pred = neural_model.predict(X_test)
test_acc = np.mean(y_test == y_test_pred)
print ('val acc: %f - train acc: %f - test acc: %f' % (val_acc, train_acc, test_acc))

plt.subplot(2, 1, 1)
plt.plot(states['loss_histroy'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(states['train_acc_histroy'], label='train')
plt.plot(states['val_acc_histroy'], label='val')
plt.title('Acc history')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.show()

from vis_utils import visualize_grid
# 可视化二层神经网络W1权重
def show_two_layer_net_weight(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
show_two_layer_net_weight(neural_model)



