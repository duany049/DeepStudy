# -*- coding:utf-8 -*-
import numpy as np
from layers import *
from gradient_check import *
from layers import *


def rel_error(x, y):
    """ 返回相对误差，加上1e-8是为了防止分母为零 """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# 大数据训练前先使用小量数据验证代码是否正确
class TestSmallData(object):
    def start_test(self):
        # 使用小量数据测试仿射代码是否正确
        num_inputs = 2  # 样本数
        input_shape = (4, 5, 6)  # 输入样本的shape
        output_dim = 3  # 下一个隐藏层的神经元数目

        input_size = num_inputs * np.prod(input_shape)  # 输入样本的元素数
        weight_size = np.prod(input_shape) * output_dim

        # 根据指定的纬度和数目以及生成x，w，b

        # 在指定区间中等差生成输入样本的总元素数的ndarray，然后再reshape成以样本数为行，其他纬度为样本本身纬度的ndarra，即生成2个样本
        x_small_data = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        weight_small = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)
        out, _ = affine_forward(x_small_data, weight_small, b)
        correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                                [3.25553199, 3.5141327, 3.77273342]])

        # 比较两者的区别，来验证代码的正确性，如果差异小于1e-9，那么可以认为代码没问题
        print('Testing affine_forward function:')
        print('difference: ', rel_error(out, correct_out))

        np.random.seed(231)
        x_small_data = np.random.randn(10, 2, 3)
        weight_small = np.random.randn(6, 5)
        b_small = np.random.randn(5)
        dout_small = np.random.randn(10, 5)

        dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x_small_data, weight_small, b_small)[0],
                                               x_small_data,
                                               dout_small)
        dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x_small_data, weight_small, b_small)[0],
                                               weight_small,
                                               dout_small)
        db_num = eval_numerical_gradient_array(lambda b: affine_forward(x_small_data, weight_small, b_small)[0],
                                               b_small,
                                               dout_small)
        _, cache = affine_forward(x_small_data, weight_small, b_small)
        dx, dw, db = affine_backward(dout_small, cache)

        print('Testing affine_backward function: shape: ', dx_num.shape)
        print('dx error: ', rel_error(np.reshape(dx_num, (dx_num.shape[0], -1)), dx))
        print('dw error: ', rel_error(dw_num, dw))
        print('db error: ', rel_error(db_num, db))

        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        out, _ = relu_forward(x)
        correct_out = np.array([[0., 0., 0., 0., ],
                                [0., 0., 0.04545455, 0.13636364, ],
                                [0.22727273, 0.31818182, 0.40909091, 0.5, ]])
        # Compare your output with ours. The error should be around 5e-8
        print('Testing relu_forward function:')
        print('difference: ', rel_error(out, correct_out))

        np.random.seed(231)
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
        _, cache = relu_forward(x)
        dx = relu_backward(dout, cache)
        # The error should be around 3e-12
        print('Testing relu_backward function:')
        print('dx error: ', rel_error(dx_num, dx))

        np.random.seed(231)
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        dout = np.random.randn(2, 10)

        out, cache = affine_relu_forward(x, w, b)
        dx, dw, db = affine_relu_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)
        print('Testing affine_relu_forward:')
        print('dx error: ', rel_error(np.reshape(dx_num.shape[0], -1), dx))
        print('dw error: ', rel_error(dw_num, dw))
        print('db error: ', rel_error(db_num, db))

        np.random.seed(231)
        num_classes, num_inputs = 10, 50
        x = 0.001 * np.random.randn(num_inputs, num_classes)
        y = np.random.randint(num_classes, size=num_inputs)

        dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
        loss, dx = softmax_loss(x, y)
        # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
        print('\nTesting softmax_loss:')
        print('loss: ', loss)
        print('dx error: ', rel_error(dx_num, dx))
