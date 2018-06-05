# -*- coding:utf-8 -*-
import numpy as np
from utils import *
from cnn import *
from layers import *
from gradient_check import *
from cnn import ThreeLayersConvNet
from data_utils import *
from solver import *


class SmallDataTest(object):
    def test(self):
        # 验证卷积层前向传播
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=w_shape[0])
        conv_param = {'stride': 2, 'padding': 1}
        print('x shape: ', x.shape)
        out, _ = conv_forward_naive(x, w, b, conv_param)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                  [-0.18387192, -0.2109216]],
                                 [[0.21027089, 0.21661097],
                                  [0.22847626, 0.23004637]],
                                 [[0.50813986, 0.54309974],
                                  [0.64082444, 0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                  [-1.19128892, -1.24695841]],
                                 [[0.69108355, 0.66880383],
                                  [0.59480972, 0.56776003]],
                                 [[2.36270298, 2.36904306],
                                  [2.38090835, 2.38247847]]]])
        # 通过少量数据测试代码是否正确
        print('Testing conv_forward_naive difference: ', rel_error(out, correct_out))

        # 验证卷积层反向传播
        np.random.seed(231)
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2, )
        dout = np.random.randn(4, 2, 5, 5)
        conv_param = {'stride': 1, 'padding': 1}
        dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

        out, cache = conv_forward_naive(x, w, b, conv_param)
        dx, dw, db = conv_backward_naive(dout, cache)
        print('Testing conv_backward_naive function')
        print('dx error: ', rel_error(dx, dx_num))
        print('dw error: ', rel_error(dw, dw_num))
        print('db error: ', rel_error(db, db_num))

        # 验证池化层前向传播
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
        pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
        out, _ = max_pool_forward_naive(x, pool_param)
        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [0.03157895, 0.04631579]]],
                                [[[0.09052632, 0.10526316],
                                  [0.14947368, 0.16421053]],
                                 [[0.20842105, 0.22315789],
                                  [0.26736842, 0.28210526]],
                                 [[0.32631579, 0.34105263],
                                  [0.38526316, 0.4]]]])
        print('Testing max_pool_forward_naive function:')
        print('difference: ', rel_error(out, correct_out))

        # 验证池化层反向传播
        np.random.seed(231)
        x = np.random.randn(3, 2, 8, 8)
        dout = np.random.randn(3, 2, 4, 4)
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

        out, cache = max_pool_forward_naive(x, pool_param)
        dx = max_pool_backward_naive(dout, cache)
        print('Testing max_pool_backward_naive function:')
        print('dx error: ', rel_error(dx, dx_num))

        # 验证卷积 relu pool的反向传播代码
        np.random.seed(231)
        x = np.random.randn(2, 3, 16, 16)
        w = np.random.randn(3, 3, 3, 3)
        b = np.random.randn(3, )
        dout = np.random.randn(2, 3, 8, 8)
        conv_param = {'stride': 1, 'padding': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
        dx, dw, db = conv_relu_pool_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x,
                                               dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w,
                                               dout)
        db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b,
                                               dout)
        print('Testing conv_relu_pool')
        print('dx error: ', rel_error(dx_num, dx))
        print('dw error: ', rel_error(dw_num, dw))
        print('db error: ', rel_error(db_num, db))

        # 验证卷积 relu的反向传播代码
        np.random.seed(231)
        x = np.random.randn(2, 3, 8, 8)
        w = np.random.randn(3, 3, 3, 3)
        b = np.random.randn(3, )
        dout = np.random.randn(2, 3, 8, 8)
        conv_param = {'stride': 1, 'padding': 1}
        out, cache = conv_relu_forward(x, w, b, conv_param)
        dx, dw, db = conv_relu_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)
        print('Testing conv_relu:')
        print('dx error: ', rel_error(dx_num, dx))
        print('dw error: ', rel_error(dw_num, dw))
        print('db error: ', rel_error(db_num, db))

        model = ThreeLayersConvNet()
        N = 50
        X = np.random.randn(N, 3, 32, 32)
        y = np.random.randint(10, size=N)
        loss, grads = model.loss(X, y)
        print('Initial loss (no regularization): ', loss)
        model.reg = 0.5
        loss, grads = model.loss(X, y)
        print('Initial loss (with regularization): ', loss)

        # 检查卷积模型loss函数得到的梯度
        num_inputs = 2
        input_dim = (3, 16, 16)
        reg = 0.0
        num_classes = 10
        np.random.seed(231)
        X = np.random.randn(num_inputs, *input_dim)
        y = np.random.randint(num_classes, size=num_inputs)
        model = ThreeLayersConvNet(num_filters=3, filter_size=3,
                                   input_dims=input_dim, hidden_dims=7,
                                   dtype=np.float64)
        loss, grads = model.loss(X, y)
        for param_name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            param_grad_num = eval_numerical_gradient(f, model.param[param_name], verbose=False, h=1e-6)
            e = rel_error(param_grad_num, grads[param_name])
            print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

        data_dir = '../data_batches_py'
        data = load_data(data_dir)
        np.random.seed(231)
        num_train = 100
        val_num = 30
        small_x_train = data['x_train'][:num_train]
        small_y_train = data['y_train'][:num_train]
        small_x_val = data['x_train'][num_train:num_train + val_num]
        small_y_val = data['y_train'][num_train:num_train + val_num]
        small_x_train = small_x_train.transpose(0, 3, 1, 2)
        small_x_val = small_x_val.transpose(0, 3, 1, 2)

        small_data = {
            'x_train': small_x_train,
            'y_train': small_y_train,
            'x_val': small_x_val,
            'y_val': small_y_val,
        }
        model = ThreeLayersConvNet(weight_scale=1e-2)
        solver = Solver(model, small_data,
                        num_epochs=40, batch_size=50,
                        update_rule='sgd_moment',
                        optim_config={
                            'learning_rate': 1e-3,
                        },
                        verbose=True, print_every=1)
        solver.train()
