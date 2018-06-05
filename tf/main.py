# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from data_utils import *

data_dir = '../data_batches_py'
data = load_data(data_dir)
X_train, y_train, X_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

num_training = 49000
num_validation = 1000
num_test = 10000
X_val = X_train[num_training:num_training + num_validation]
y_val = y_train[num_training:num_training + num_validation]
X_train = X_train[:num_training]
y_train = y_train[:num_training]
X_test = X_test[:num_test]
y_test = y_test[:num_test]

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
# 设置输入，比如每个batch要输入的数据
# 第一维是None, 可以根据输入的batch size自动改变。

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


# def simple_model(X, y):
#     # define our weights (e.g. init_two_layer_convnet)
#     #  定义权重W
#     # setup variables
#     # 设置变量
#     Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
#     bconv1 = tf.get_variable("bconv1", shape=[32])
#     W1 = tf.get_variable("W1", shape=[5408, 10])
#     b1 = tf.get_variable("b1", shape=[10])
#
#     # define our graph (e.g. two_layer_convnet)
#     # 定义我们的图
#
#     # 这里我们需要用到conv2d函数，建议大家仔细阅读官方文档
#     # tf.nn.conv2d()  https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
#     # conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,data_format=None,name=None)
#     # input ： [batch, in_height, in_width, in_channels]
#     # filter/kernel: [filter_height, filter_width, in_channels, out_channels]
#     # strides: 长度为4的1维tensor，用来指定在每一个维度上的滑动的窗口滑动的步长
#     # 水平或者垂直滑动通常会指定strides = [1,stride,,stride,1]
#     # padding: 'VALID' 或者是 'SAME'
#     # data_format: 数据的输入格式，默认是‘NHWC’
#
#     # 根据输出的大小的公式：(W-F+2P)/S + 1
#     # W: 图像宽度   32
#     # F：Filter的宽度  7
#     # P: padding了多少  0
#     # padding='valid' 就是不padding  padding='same' 自动padding若干个行列使得输出的feature map和原输入feature map的尺寸一致
#     # S: stride 步长  2
#
#     a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1
#     # (W-F+2P)/S + 1 = (32 - 7 + 2*0)/2 + 1 = 13
#     # 那么输出的feature map的尺寸就是 13 * 13 * 32 = 5408   （Wconv1 有32个out channels, 也就是说有32个filters）
#
#     h1 = tf.nn.relu(a1)  # 对a1中的每个神经元加上激活函数relu
#     h1_flat = tf.reshape(h1, [-1, 5408])  # reshape h1，把feature map展开成 batchsize * 5408
#     y_out = tf.matmul(h1_flat, W1) + b1  # 得到的输出是每个样本在每个类型上的得分
#     return y_out
#
#
# y_out = simple_model(X, y)
#
# # define our loss
# # 定义我们的loss
#
# total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
# mean_loss = tf.reduce_mean(total_loss)  # loss求平均
#
# # define our optimizer
# # 定义优化器，设置学习率
# optimizer = tf.train.AdamOptimizer(5e-4)  # select optimizer and set learning rate
# train_step = optimizer.minimize(mean_loss)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    '''
    run model函数主要是控制整个训练的流程，需要传入session，调用session.run(variables)会得到variables里面各个变量的值。
    这里当训练模式的时候，也就是training!=None，我们传入的training是之前定义的train_op，调用session.run(train_op)会自动完成反向求导，
    整个模型的参数会发生更新。
    当training==None时,是我们需要对验证集合做一次预测的时候(或者是测试阶段)，这时我们不需要反向求导，所以variables里面并没有加入train_op
    '''
    # have tensorflow compute accuracy
    # 计算准确度（ACC值）
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    # 对训练样本进行混洗
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    # 设置需要计算的变量
    # 如果需要进行训练，将训练过程(training)也加进来
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    # 进行迭代
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        # 记录损失函数和准确度的变化
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        # 确保每个训练样本都被遍历
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            # 产生一个minibatch的样本
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            # 生成一个输入字典(feed dictionary)
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            # 获取minibatch的大小
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            # 计算损失函数和准确率
            # 如果是训练模式的话，执行训练过程
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            # 记录本轮的训练表现
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            # 定期输出模型表现
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct


def custom_model(X, y, is_training):
    def conv_relu_pool(X, num_filter=32, conv_strides=1, kernel_size=[3, 3], pool_size=[2, 2], pool_strides=2):
        conv1 = tf.layers.conv2d(inputs=X, filters=num_filter, kernel_size=kernel_size, strides=conv_strides,
                                 padding='same', activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=pool_strides)
        return pool

    def conv_relu_conv_relu_pool(X, num_filter1=32, num_filter2=32, conv_strides=1, kernel_size=[5, 5],
                                 pool_size=[2, 2], pool_strides=2):
        conv1 = tf.layers.conv2d(X, num_filter1, kernel_size=kernel_size, strides=conv_strides, padding='same',
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, num_filter2, kernel_size=kernel_size, strides=conv_strides, padding='same',
                                 activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(conv2, pool_size=pool_size, strides=pool_strides)
        return pool

    def affine(X, num_unit, act):
        return tf.layers.dense(X, num_unit, activation=act)

    def batchnorm_relu_conv(X, num_filters=32, conv_strides=2, kernel_size=[5, 5], is_training=True):
        bn = tf.layers.batch_normalization(X, training=is_training)
        act = tf.nn.relu(bn)
        conv = tf.layers.conv2d(act, num_filters, kernel_size=kernel_size, strides=conv_strides, padding='same',
                                activation=None)
        return conv

    num_conv_layer = 3
    num_affine_layer = 1
    conv = tf.layers.conv2d(inputs=X, filters=64, kernel_size=[5, 5], strides=1, padding="same", activation=None)
    for i in range(num_conv_layer):
        print('cur input shape', conv.get_shape())
        conv = batchnorm_relu_conv(conv, is_training=is_training)

    print('out shape of conv layers: ', conv.get_shape())
    global_average_shape = conv.get_shape()[1:3]
    print('global_average_shape: ', global_average_shape)
    avg_layer = tf.reduce_mean(conv, [1, 2])  # 计算第1、2纬度的均值
    fc = avg_layer
    # keep_prob = tf.constant(0.5)
    for i in range(num_affine_layer):
        fc = affine(fc, 100, tf.nn.relu)
        # fc = tf.nn.dropout(fc, keep_prob)

    fc = affine(fc, 10, None)
    return fc


tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = custom_model(X, y, is_training)
total_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=tf.one_hot(y, 10))
mean_loss = tf.reduce_mean(total_loss)  # 计算均值

global_step = tf.Variable(0, trainable=False, name="Global_Step")
starter_learning_rate = 1e-2
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           750, 0.96, staircase=True)  # 学习率逐步衰减

# learning_rate = starter_learning_rate
# define our optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)  # 选择一个优化器，并且设置学习率

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss, global_step=global_step)

print([x.name for x in tf.global_variables()])

with tf.Session() as sess:
    print('Training')
    sess.run(tf.global_variables_initializer())
    run_model(sess, y_out, mean_loss, X_train, y_train, 2, 64
              , 100, train_step, True)
    print('Validation')
    run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)
