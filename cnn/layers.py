# -*- coding:utf-8 -*-
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    padding = conv_param['padding']
    new_H = int((H + 2 * padding - HH) / stride) + 1
    new_W = int((W + 2 * padding - WW) / stride) + 1
    new_C = F
    new_N = N
    out = np.zeros((new_N, new_C, new_H, new_W))
    cache = (x, w, b, conv_param)

    for index_n in range(N):
        for index_f in range(F):
            # 输出的每个深度提前赋上b值
            conv_newH_newW = np.ones([new_H, new_W]) * b[index_f]
            for index_c in range(C):
                # 从每个样本的每个深度增加pad
                padded_c_x = np.lib.pad(x[index_n, index_c], pad_width=padding, mode='constant', constant_values=0.0)
                for index_h in range(new_H):
                    for index_w in range(new_W):
                        # 获取每个样本对应每个过滤器的输出值的每个深度,不过因为会遍历所有深度C,所以最后的conv_newH_newW值为
                        # 某个样本对应某个过滤器输出值的一个点的值
                        test_pad = padded_c_x[index_h * stride: index_h * stride + HH,
                                   index_w * stride: index_w * stride + WW] * w[index_f, index_c, :, :]
                        conv_newH_newW[index_h, index_w] += np.sum(test_pad)
                out[index_n, index_f] = conv_newH_newW
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    stride = conv_param['stride']
    pad = conv_param['padding']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    out_N, out_C, out_H, out_W = dout.shape
    # 恢复填充状态,也就是和W卷积的值
    padded_x = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    padded_dx = np.zeros_like(padded_x)

    for index_N in range(out_N):  # 其实就等于N
        for index_C in range(out_C):  # 其实就等于F
            for index_H in range(out_H):
                for index_W in range(out_W):
                    db[index_C] += dout[index_N, index_C, index_H, index_W]
                    # x对应此w相乘的x部分
                    dw[index_C] += dout[index_N, index_C, index_H, index_W] \
                                   * padded_x[index_N, :, index_H * stride: index_H * stride + HH,
                                     index_W * stride: index_W * stride + WW]
                    padded_dx[index_N, :, index_H * stride: index_H * stride + HH,
                    index_W * stride: index_W * stride + WW] += w[index_C] * dout[index_N, index_C, index_H, index_W]
    # 去掉pad得到dx
    dx = padded_dx[:, :, pad: H + pad, pad: W + pad]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    N, C, H, W = x.shape
    stride = pool_param['stride']
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    new_H = int((H - pool_height) / stride) + 1
    new_W = int((W - pool_width) / stride) + 1
    out = np.zeros((N, C, new_H, new_W))

    for index_n in range(N):
        for index_c in range(C):
            for index_h in range(new_H):
                for index_w in range(new_W):
                    # 在二维数据上取最大值,进行降维操作
                    out[index_n, index_c, index_h, index_w] \
                        = np.max(x[index_n, index_c, index_h * stride: index_h * stride + pool_width,
                                 index_w * stride: index_w * stride + pool_width])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    stride = pool_param['stride']
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    new_H = int((H - pool_height) / stride) + 1
    new_W = int((W - pool_width) / stride) + 1
    dx = np.zeros_like(x)

    for index_n in range(N):
        for index_c in range(C):
            for index_h in range(new_H):
                for index_w in range(new_W):
                    window = x[index_n, index_c, index_h * stride: index_h * stride + pool_height,
                             index_w * stride: index_w * stride + pool_width]
                    dx[index_n, index_c, index_h * stride: index_h * stride + pool_height,
                    index_w * stride: index_w * stride + pool_width] = (window == np.max(window)) * dout[
                        index_n, index_c, index_h, index_w]
    return dx


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = (x > 0) * dout
    return dx


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    conv_out, conv_cache = conv_forward_naive(x, w, b, conv_param)
    relu_out, relu_cache = relu_forward(conv_out)
    out, pool_cache = max_pool_forward_naive(relu_out, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    pool_dx = max_pool_backward_naive(dout, pool_cache)
    relu_dx = relu_backward(pool_dx, relu_cache)
    dx, dw, db = conv_backward_naive(relu_dx, conv_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    conv_out, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(conv_out)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    relu_dx = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(relu_dx, conv_cache)
    return dx, dw, db


def affine_relu_forward(x, w, b):
    affine_out, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(affine_out)
    cache = (affine_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    affine_cache, relu_cache = cache
    relu_dx = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(relu_dx, affine_cache)
    return dx, dw, db


def affine_forward(x, w, b):
    out = x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = (x > 0) * dout
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