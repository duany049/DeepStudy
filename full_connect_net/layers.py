# -*- coding:utf-8 -*-
import numpy as np
from utils import *


def affine_forward(x_data, w, b):
    """
    计算并且输出分数和缓存

    :param x_data:
    :param w:
    :param b:
    :return:
    """
    out = None
    reshaped_x = np.reshape(x_data, (x_data.shape[0], -1))
    out = reshaped_x.dot(w) + b
    # 并不需要全局保存，返回给train函数，以计算梯度值
    cache = (x_data, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    返回x，w，b的梯度

    :param dout:
    :param cache: 仿射层的输入x,w,b
    :return:
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.reshape(dout.dot(w.T), x.shape)
    dw = x.T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    return out, (fc_cache, relu_cache)


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache  # fc_cache = (x, w, b)   relu_cache = a
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)  # 防止分母为0
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        # 在 train 模式里，running_mean, running_var 会在每经过一个BN层进行一次迭代计算
        cur_mean = np.mean(x, axis=0, keepdims=True)
        cur_var = np.var(x, axis=0, keepdims=True)
        running_mean = momentum * running_mean + (1 - momentum) * cur_mean
        running_var = momentum * running_var + (1 - momentum) * cur_var
        x_hat = x - running_mean / np.sqrt(running_var + eps)
        out = x_hat * gamma + beta
        # 参数(gamma, beta)是我们新的待优化学习的参数
        cache = (x, cur_mean, cur_var, x_hat, eps, gamma, beta)
    elif mode == 'test':
        x_hat = x - running_mean / np.sqrt(running_var + eps)
        out = x_hat * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    # 把值存入bn_param中
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    x, mean, var, x_hat, eps, gamma, beta = cache
    dx, dgamma, dbeta = None, None, None
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * dgamma
    N = x.shape[0]
    dx_hat_numerator = dx_hat / np.sqrt(var + eps)
    dx_hat_denominator = np.sum(dx_hat * (x - mean), axis=0)
    dx_1 = dx_hat_numerator
    dvar = -0.5 * ((var + eps) ** (-1.5)) * dx_hat_denominator
    dmean = -1.0 * np.sum(dx_hat_numerator, axis=0) + \
            dvar * np.mean(-2.0 * (x - mean), axis=0)
    dx_var = dvar * 2.0 / N * (x - mean)
    dx_mean = dmean * 1.0 / N
    dx = dx_1 + dx_var + dx_mean
    return dx, dgamma, dbeta


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relue_cache = cache
    drelu_input = relu_backward(dout, relue_cache)
    dbn_input, dgamma, dbeta = batchnorm_backward(drelu_input, bn_cache)
    dx, dw, db = affine_backward(dbn_input, fc_cache)
    return dx, dw, db, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask, out = None, None
    if mode == 'train':
        keep_prop = 1 - p
        mask = (np.random.rand(*x.shape) < keep_prop) / keep_prop  # 之所以再除以keep_prop是为了让最终平均期望值不变
        out = mask * x
    elif mode == 'test':
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    dx = None
    drop_param, mask = cache
    mode = drop_param['mode']
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def softmax_loss(x, y):
    """
    他的正则表达式部分的损失，根据是否使用正则表达式分开计算
    :param x:
    :param y:
    :return:
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
