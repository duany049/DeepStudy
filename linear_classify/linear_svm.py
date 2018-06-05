# -*- coding:utf-8 -*-
import numpy as np
from random import randrange

def svm_loss_naive(W, X, y, reg):
    """
    输入纬度为D，有C类，使用N个样本作为第一批输入
    :param W:   一个numpy array，形状为(D,C)存储权重
    :param X:   一个numpy array，形状为(N,D)存储一个小批数据
    :param y:   一个numpy array，存储训练标签
    :param reg: float，表示正则化强度
    :return:
    """
    dW = np.zeros(W.shape)
    num_class = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                # 之所以这么做是因为损失函数对Wy[i]求偏导得到的是-Xi
                dW[:, y[i]] += -X[i, :].T
                # 之所以这么做是因为损失函数对Wj求偏导得到的是+Xi
                dW[:, j] += X[i, :].T

    loss /= num_train
    dW /= num_train
    #     加入正则项
    loss += reg * np.sum(W * W)
    dW += reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    完全矢量化版本
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    scores = X.dot(W)
    num_train = X.shape[0]

    scores_correct = scores[np.arange(num_train), y]  # 所有样例正确类对应的分数
    scores_correct = np.reshape(scores_correct, (num_train, -1))
    margins = scores - scores_correct + 1.0  # 计算scores矩阵中每一处的损失
    margins = np.maximum(0, margins)
    margins[np.arange(num_train), y] = 0.0  # 每个样本的正确类别损失置0
    loss += np.sum(margins) / num_train  # 累加所有损失除以总数
    loss += 0.5 * reg * np.sum(W * W)  # 正则一般乘以1/2

    # 计算梯度
    margins[margins > 0] = 1.0  # 为了让后面的np.dot(X.T, margins)计算
    row_sum = np.sum(margins, axis=1)  # N*1  每个样本累加
    margins[np.arange(num_train), y] = -row_sum  # 类正确的位置 = -梯度累加
    """
    np.dot(X.T, margins)相当于非矢量化中，对所有margins大于0预测进行dW[:, y[i]] += -X[i, :].T
    / num_train是取均值
    """
    dW += np.dot(X.T, margins) / num_train + reg * W
    return loss, dW

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """

  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] = oldval - h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))