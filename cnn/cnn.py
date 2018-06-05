# -*- coding:utf-8 -*-
import numpy as np
from layers import *
from time import *


class ThreeLayersConvNet(object):
    """
    结构如:conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """

    def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7, hidden_dims=100,
                 num_classed=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """
        初始化网络
        :param input_dims:
        :param num_filters:
        :param filter_size:
        :param hidden_dims:
        :param num_classed:
        :param weight_scale:
        :param reg:
        :param dtype:
        """
        self.param = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dims
        """
        w1,b1是卷积层的参数
        """
        self.param['w1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.param['b1'] = np.zeros(num_filters, )
        """
        w2,b2是第一层仿射层的参数
        w2[0]之所以是值num_filters * H * W / 4，是因为进行此操作前，现将单个三维数据二维化(num_filters, num_filters * H * W / 4)
        """
        self.param['w2'] = weight_scale * np.random.randn(int(num_filters * H * W / 4),
                                                          hidden_dims) * weight_scale
        self.param['b2'] = np.zeros(hidden_dims, )
        """
        w3,b3是第二层放射层的参数
        """
        self.param['w3'] = weight_scale * np.random.randn(hidden_dims, num_classed)
        self.param['b3'] = np.zeros(num_classed, )

        for k, v in self.param.items():
            self.param[k] = v.astype(np.float32)

    def loss(self, x, y=None):
        """
        在此方法中计算loss和梯度
        :param x:
        :param y:
        :return:
        """
        w1, w2, w3 = self.param['w1'], self.param['w2'], self.param['w3']
        b1, b2, b3 = self.param['b1'], self.param['b2'], self.param['b3']
        filter_size = w1.shape[2]
        reg = self.reg
        conv_param = {'stride': 1, 'padding': int((filter_size - 1) / 2)}
        pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
        # conv_relu_pool_forward_time = time()
        conv_out, conv_cache = conv_relu_pool_forward(x, w1, b1, conv_param, pool_param)
        # print ('conv_relu_pool_forward time: ', time() - conv_relu_pool_forward_time)
        reshaped_conv_out = conv_out.reshape(conv_out.shape[0], int(conv_out.size / conv_out.shape[0]))
        # affine_relu_forward_time = time()
        affine_relu_out, affine_relu_cache = affine_relu_forward(reshaped_conv_out, w2, b2)
        # print ('affine_relu_forward time: ', time() - affine_relu_forward_time)
        # affine_forward_time = time()
        scores, cache = affine_forward(affine_relu_out, w3, b3)
        # print ('affine_forward time: ', time() - affine_forward_time)
        if y is None:
            return scores

        loss, grads = 0, {}
        # softmax_loss_time = time()
        loss, loss_dx = softmax_loss(scores, y)
        # print ('softmax_loss time: ', time() - softmax_loss_time)
        loss += 0.5 * reg * np.sum(w1 ** 2) + 0.5 * reg * np.sum(w2 ** 2) + 0.5 * reg * np.sum(w3 ** 2)

        # affine_backward_time = time()
        dx3, dw3, db3 = affine_backward(loss_dx, cache)
        # print ('affine_backward time: ', time() - affine_backward_time)
        grads['w3'] = dw3 + reg * dw3
        grads['b3'] = db3
        # affine_relu_backward_time = time()
        dx2, dw2, db2 = affine_relu_backward(dx3, affine_relu_cache)
        # print ('affine_relu_backward time: ', time() - affine_relu_backward_time)
        grads['w2'] = dw2 + reg * dw2
        grads['b2'] = db2
        dx2 = dx2.reshape(*conv_out.shape)
        # conv_relu_pool_backward_time = time()
        dx1, dw1, db1 = conv_relu_pool_backward(dx2, conv_cache)
        # print ('conv_relu_pool_backward time: ', time() - conv_relu_pool_backward_time)
        grads['w1'] = dw1 + reg * dw1
        grads['b1'] = db1
        return loss, grads
