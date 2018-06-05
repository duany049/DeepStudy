# -*- coding:utf-8 -*-
import optimize
import numpy as np


class Solver(object):
    """
    定义solver对模型框架类(比如：FullConnectNet)进行封装,我们将在训练集和验证数据集中训练我们的模型，
    并周期性地检查准确率以避免过拟合
    train()函数是最重要的，调用他之后会训练模型并且自动启用模型优化程序
    """

    def __init__(self, model, data, **kwargs):
        """

        :param model:  模型
        :param data:    数据
        :param kwargs:
        # 可选参数
          # 优化算法：默认为sgd update_rule
          # 设置优化算法的超参数：optim_config
          # 学习率在每次epoch时衰减率 lr_decay
          # 在训练时，模型输入层接收样本图片的大小，默认100 batch_size
          # 在训练时，让神经网络模型一次全套训练的遍数 num_epochs
          # 在训练时，打印损失值的迭代次数 print_every
          # 是否在训练时输出中间过程 verbose
        """
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_val = data['x_val']
        self.y_val = data['y_val']
        """
        下面是可选的输入参数
        """
        self.update_rule = kwargs.pop('update_rule', 'sgd_moment')  # 默认优化算法sgd_moment
        self.optim_config = kwargs.pop('optim_config', {})  # 默认设置优化算法参数为空字典
        self.lr_decay = kwargs.pop('lr_decay', 0.95)  # 默认学习衰减率为0.95
        self.batch_size = kwargs.pop('batch_size', 20)  # 训练时默认每次使用样本数目200
        self.num_epochs = kwargs.pop('num_epochs', 10)  # 默认的训练的次数100次
        self.print_every = kwargs.pop('print_every', 10)  # 默认每迭代10次打印损失值
        self.verbose = kwargs.pop('verbose', True)  # 是否打印训练的中间过程
        # 如果除了上述参数，还有其他参数则报异常
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        if not hasattr(optimize, self.update_rule):  # 如果optimize中没有update_rule对应的优化算法就报错
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optimize, self.update_rule)  # self.update_rule转化为优化算法的函数
        self._reset()

    def _reset(self):
        """
        重置一些用于记录优化的变量
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        # 做一次深拷贝
        self.optim_configs = {}
        for p in self.model.param:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        """ 上面根据模型中待学习的参数，创建了新的优化字典self.optim_configs，
        形如：{'b': {'learnning_rate': 0.0005}
             ,'w': {'learnning_rate': 0.0005}}，为每个模型参数指定了相同的超参数。
        """

    def _step(self):
        """
        仅被train函数调用,在训练模式下正向传播和反向传播一次，且更新模型参数一次
        """
        num_train = self.x_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        loss, grads = self.model.loss(x_batch, y_batch)
        self.loss_history.append(loss)
        # 执行一次模型参数的更新
        for k, w in self.model.param.items():
            dw = grads[k]  # 从临时梯度字典中，取出模型参数k对应的梯度值
            config = self.optim_configs[k]  # 取出模型参数k对应的优化超参数
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.param[k] = next_w  # 新参数替换掉旧的
            self.optim_configs[k] = next_config  # 新超参数替换掉旧的，如动量v

    def check_accuracy(self, X, y, num_samples=None, batch_size=20):
        """
        根据某图片样本数据，计算某与之对应的标签的准确率
        :param X:
        :param y:
        :param num_samples:
        :param batch_size:
        :return:
        """
        N = X.shape[0]  # 样本图片X的总数
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        print ('test check_accuracy N %d, batch_size: %d' % (N, batch_size))
        num_batches = N // batch_size
        print ('test check_accuracy num_batches: ', num_batches)
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    def train(self):
        # 首先要确定下来总共要进行的迭代的次数num_iterations
        num_train = self.x_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)  # 每个epoch要进行几遍迭代
        num_iterations = self.num_epochs * iterations_per_epoch  # 总迭代次数
        """
        开始迭代循环
        """
        for it in range(num_iterations):
            self._step()  # 进行一次神经网络迭代，并且更新模型的参数，存储最新loss值
            if self.verbose and it % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (it + 1,
                                                        num_iterations, self.loss_history[-1]))
            epoch_end = (it + 1) % iterations_per_epoch == 0  # 是否为当前epoch最后一次迭代
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:  # 第一遍之后开始，每遍给学习率自乘一个衰减率
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
            first_it = (it == 0)  # 起始的t
            last_it = (it == num_iterations - 1)  # 最后的iter
            if first_it or last_it or epoch_end:  # 在最开始／最后／每遍epoch结束时
                train_acc = self.check_accuracy(self.x_train, self.y_train,
                                                num_samples=100)  # 随机取1000个训练图看准确率
                val_acc = self.check_accuracy(self.x_val, self.y_val)  # 计算全部验证图片的准确率
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc))
            """
            """
            # 在最开始／最后／每遍结束时，比较当前验证集的准确率和过往最佳验证集
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {}
                for k, v in self.model.param.items():
                    self.best_params[k] = v.copy()  # copy()仅复制值过来

        """
        结束迭代循环！
        """
        self.model.param = self.best_params  # 最后把得到的最佳模型参数存入到模型中
