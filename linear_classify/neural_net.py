# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        W1: 隐藏层第一层的权重矩阵shape为 (D, H)
        b1: 隐藏层第一层的偏置向量shape为 (H,)
        W2: 隐藏层第二层的权重矩阵shape为 (H, C)
        b2: 隐藏层第二层的偏置向量shape为 (C,)

        :param input_size:  特征纬度D
        :param hidden_size: 隐藏层神经元的数目H
        :param output_size: 类型的数量C
        :param std: 对初始权重取极小值的系数(小型神经网络这么初始化没啥问题，大型神经网络这么初始化容易导致神经元饱和问题)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        一次对批量的数据(这批次的所有样本)进行处理，求平均loss,而不是对一批数据中一个个的样本进行梯度更新？？？？？
        :param X:
        :param y:
        :param onlyScore:
        :param reg:
        :return:
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        scores = None
        z1 = X.dot(W1) + b1  # 第一层对输入进行仿射
        a1 = np.maximum(0, z1)  # 第二层使用ReLu激活函数处理第一层的输入
        scores = a1.dot(W2) + b2  # 激活函数处理的结果进行仿射得到各个类的分值
        if y is None:  # 如果没有指定标签，就输出分值
            print ('score shape', scores.shape)
            return scores

        """
            下面进入输出层(也就是分类层),此处使用的分类器是softmax函数
        """
        loss = None
        # scores_exp = np.exp(scores)
        # probability = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)  # 得到各个样本为各个类的概率

        # 使用下面的方法而不是上面的方法，是因为这样得到的值一样，还可以减小最大值，减小性能损耗，防止溢出
        shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
        shift_exp = np.exp(shift_scores)
        probability = shift_exp / np.sum(shift_exp, axis=1, keepdims=True)  # 得到各个样本为各个类的概率

        loss = -np.sum(np.log(probability[range(N), y]))  # 得到所有样本的总的loss
        loss /= N
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  # loss得加上最后的正则化式子

        """
            下面进行反向传播求梯度
        """
        dScores = probability
        dScores[range(N), y] -= 1  # Loss函数对于正确的类的导数为P-1，其他的类为P
        dScores /= N
        # W2, b2
        gradients = {}
        gradients['W2'] = np.dot(a1.T, dScores)  # Lost函数对W求导结果为输入的转置*dScores,此处矩阵运算相当于所有样本的梯度之和
        gradients['b2'] = np.sum(dScores, axis=0)  # 所有样本关于b的梯度之和
        # 反向传播的第二个隐藏层
        dHidden = np.dot(dScores, W2.T)  # 激活函数ReLu的梯度
        dHidden[a1 <= 0] = 0  # 激活函数ReLu中输出值为0的梯度为0
        # W1, b1
        gradients['W1'] = np.dot(X.T, dHidden)
        gradients['b1'] = np.sum(dHidden, axis=0)
        # 加上对应梯度的正则化求导部分
        gradients['W2'] += reg * W2
        gradients['W1'] += reg * W1

        return loss, gradients

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        loss_histroy = []
        train_acc_histroy = []
        val_acc_histroy = []
        for it in range(num_iters):
            sample_indexs = np.random.choice(range(num_train), batch_size)
            X_batch = X[sample_indexs]
            y_batch = y[sample_indexs]
            loss, gradients = self.loss(X_batch, y_batch, 5e-2)
            loss_histroy.append(loss)
            # 使用SGD梯度下降法
            self.params['W1'] += -learning_rate * gradients['W1']
            self.params['b1'] += -learning_rate * gradients['b1']
            self.params['W2'] += -learning_rate * gradients['W2']
            self.params['b2'] += -learning_rate * gradients['b2']

            if verbose and it % 100 == 0:
                print ('iteration %f/%f loss: %f' % (it, num_iters, loss))

            # 没进行一次整体数据量的梯度下降，就用当前得到的权值和偏置向量对所有训练数据以及val数据进行预测，
            # 并且对预测得到的精确度进行记录，并且降低学习率
            if it % iterations_per_epoch == 0:
                y_pred = self.predict(X)
                y_val_pred = self.predict(X_val)
                train_acc = np.mean(y_pred == y)
                val_acc = np.mean(y_val_pred == y_val)
                train_acc_histroy.append(train_acc)
                val_acc_histroy.append(val_acc)

                # 随着随机梯度下降的进行，对学习率逐渐降低的策略
                learning_rate *= learning_rate_decay
        return {
            'loss_histroy': loss_histroy,
            'train_acc_histroy': train_acc_histroy,
            'val_acc_histroy': val_acc_histroy
        }

    def predict(self, X):
        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1)
        scores = np.dot(a1, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred
