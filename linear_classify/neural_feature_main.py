# -*- coding:utf -8-
import random
import numpy as np
from data_utils import load_data
import matplotlib.pyplot as plt
from features import *
import neural_net

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = '../data_batches_py'
X_train, y_train, X_test, y_test = load_data(cifar10_dir)
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = list(range(num_training, num_training + num_validation))
X_val = X_train[mask]
y_val = y_train[mask]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

num_color_bins = 10  # 颜色空间bin的数目
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)  # extract_features是把梯度方向直方图和颜色空间抽取的特征结合起来，比当个的效果更好
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# 数据预处理，去中心化
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# 数据预处理，除以标准差，这样可以移除数据的scale
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# 增加一个偏置向量的位置
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

input_dim = X_train_feats.shape[1]
hidden_num = 500
class_num = 10

net = neural_net.TwoLayerNet(input_dim, hidden_num, class_num)
history = net.train(X_train_feats, y_train, X_val_feats, y_val, learning_rate=1.4e-1, num_iters=5000, batch_size=200,
                    verbose=True, reg=2)
y_test_pred = net.predict(X_test_feats)
print('Test acc', np.mean(y_test_pred == y_test))

# X_train = np.reshape(X_train, (X_train.shape[0], -1))
# X_val = np.reshape(X_val, (X_val.shape[0], -1))
# X_test = np.reshape(X_test, (X_test.shape[0], -1))
# net = neural_net.TwoLayerNet(X_train.shape[1], hidden_num, class_num)
# history = net.train(X_train, y_train, X_val, y_val, learning_rate=1e-4, num_iters=5000, batch_size=200,
#                     verbose=True, reg=0.25)
print('Final loss: %f - final train acc: %f - final val acc: %f'
      % (history['loss_histroy'][-1], history['train_acc_histroy'][-1], history['val_acc_histroy'][-1]))

plt.subplot(2, 1, 1)
plt.plot(history['loss_histroy'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(history['train_acc_histroy'], label='train')
plt.plot(history['val_acc_histroy'], label='val')
plt.title('Acc history')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.show()

best_acc = -1
best_net = None
reg_history = []
rate_history = []
acc_history = []
max_count = 55
for count in range(max_count):
    rate = 10 ** random.uniform(-1, 0.5)
    reg = 10 ** random.uniform(-2, 1)
    net = neural_net.TwoLayerNet(input_dim, hidden_num, class_num)
    net.train(X_train_feats, y_train, X_val_feats, y_val, rate, reg=reg, num_iters=1500, batch_size=200,
              verbose=False)
    y_val_pred = net.predict(X_val_feats)
    val_acc = np.mean(y_val_pred == y_val)
    if val_acc > best_acc:
        best_acc = val_acc
        best_net = net
    reg_history.append(reg)
    rate_history.append(rate)
    acc_history.append(val_acc)
    print('reg: %f - rate: %f - acc: %f' % (reg, rate, val_acc))

for it in range(len(reg_history)):
    print('learning rate: %f - reg: %f - acc: %f' % (rate_history[it], reg_history[it], acc_history[it]))

y_test_pred = best_net.predict(X_test_feats)
print('Test acc', np.mean(y_test_pred == y_test))
