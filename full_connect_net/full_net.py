# -*- coding:utf-8 -*-
import numpy as np
from layers import *


class FullConnectNet(object):
    """
    形式如：{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    """

    def __init__(self, hidden_dims, input_dims=3 * 32 * 32, num_class=10, dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
           hidden_dims: ndarray，数据数目代表有多少隐藏层，每个值代表所在隐藏层的神经元数目
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}  # 存储大部分待调整的神经网络参数
        in_dim = input_dims
        # 循环初始化隐藏层的参数
        for index, h_dim in enumerate(hidden_dims):
            self.params['w%d' % (index + 1,)] = weight_scale * np.random.randn(in_dim, h_dim)
            self.params['b%d' % (index + 1,)] = np.zeros(h_dim)
            if use_batchnorm:
                self.params['gamma%d' % (index + 1)] = np.ones(h_dim)  # 使用1初始化
                self.params['beta%d' % (index + 1)] = np.zeros(h_dim)  # 使用0初始化
            in_dim = h_dim
        # 初始化输出层的参数
        self.params['w%d' % (self.num_layers,)] = weight_scale * np.random.randn(in_dim, num_class)
        self.params['b%d' % (self.num_layers,)] = np.zeros(num_class)

        #     当开启 dropout 时，我们需要在每一个神经元层中传递一个相同的dropout 参数字典 self.dropout_param
        self.dropout_param = {}
        if dropout:  # 如果取值为(0,1)之间，认为启用dropout
            self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.params['seed'] = seed
        # 如果开启批量归一化时，我们要定义一个BN算法的参数列表 self.bn_params ，以用来跟踪记录每一层的平均值和标准差
        self.bn_params = []
        if self.use_batchnorm:  # 如果开启，默认每层为训练模式
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

            #     最后调整所有待学习的神经网络超参数
        for key, value in self.params.items():
            self.params[key] = value.astype(dtype)

    def loss(self, x, y=None):
        """
        #
        在训练模式下：
        输出loss和一个grads字典,其中存有loss关于隐藏层和输出层的参数(W,B,gamma,beta)的梯度值.
        #
        在测试模式下：
        只给出输出层最后的得分
        #
        :param x:  训练数据
        :param y:
        :return:
        """
        x = x.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        secores = None
        """
        %前向传播%
        如果开启了dropout，我们需要将dropout的参数字典 self.dropout_param 
        在每一个dropout层中传递。
        如果开启了批量归一化，我们需要指定BN算法的参数列表 self.bn_params[0]
        对应前向传播第一层的参数，self.bn_params[1]对应第二层的参数，以此类推。
        """
        fc_mix_cache = {}  # 初始化每层前向传播的缓冲字典
        if self.use_dropout:  # 如果开启dropout，初始化do缓冲字典，因为dp和其他的缓冲数据（fc_mix_cache）没放在一块，架构这么设计的(这样函数的数目可以减少，具体原因可以看下面实现)
            dp_cache = {}
        # 从第一个隐藏层开始循环每一个隐藏层，传递数据out，保存每一层的缓冲cache
        out = x
        for i in range(self.num_layers - 1):
            w, b = self.params['w%d' % (i + 1,)], self.params['b%d' % (i + 1,)]
            if self.use_batchnorm:
                gamma, beta = self.params['gamma%d' % (i + 1,)], self.params['beta%d' % (i + 1,)]
                out, fc_mix_cache[i] = affine_bn_relu_forward(out, w, b, gamma, beta, bn_param)
            else:
                out, fc_mix_cache[i] = affine_relu_forward(x, w, b)
            if self.use_dropout:
                out, dp_cache[i] = dropout_forward()

        # 输出层
        w = self.params['w%d' % (self.num_layers,)]
        b = self.params['b%d' % (self.num_layers,)]
        out, out_cache = affine_forward(out, w, b)
        secores = out

        if mode == 'test':
            return secores

        loss, grads = 0.0, {}  # 初始化此次训练loss值和grads梯度字典
        loss, dout = softmax_loss(secores, y)
        loss += 0.5 * self.reg * np.sum(self.params['w%d' % (self.num_layers,)] ** 2)
        dout, dw, db = affine_backward(dout, out_cache)
        grads['w%d' % (self.num_layers,)] = dw + self.reg * self.params['w%d' % (self.num_layers,)]
        grads['b%d' % (self.num_layers,)] = db + self.reg * self.params['b%d' % (self.num_layers,)]
        for i in range(self.num_layers - 1):
            hi = self.num_layers - 2 - i
            loss += 0.5 * self.reg * np.sum(self.params['w%d' % (hi + 1,)] ** 2)
            if self.use_dropout:
                dout = dropout_backward(dout, dp_cache[hi])
            if self.use_batchnorm:
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, fc_mix_cache[hi])
                grads['gamma%d' % (hi + 1,)] = dgamma
                grads['beta%d' % (hi + 1,)] = dbeta
            else:
                dout, dw, db = affine_relu_backward(dout, fc_mix_cache[hi])
            grads['w%d' % (hi + 1,)] = dw + self.reg * self.params['w%d' % (hi + 1,)]
            grads['b%d' % (hi + 1,)] = db + self.reg * self.params['b%d' % (hi + 1,)]

        return loss, grads  # 输出训练模式下的损失值和梯度
