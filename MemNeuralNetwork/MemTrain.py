#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "H.YL"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MemNeuralNetwork.ActivationFunction import *

class NeuralNetwork(object):

    def __init__(self, Network, X, label, mu=0.01, needG=False, GinitValue=1, sd=0.01, Bias=True):
        """
        深度神经网络初始化
        :param Network: int[] 神经网络的形状
        :param X: int[int[]] 训练数据集，shape[0]
        :param label: int[int[]] 训练数据标签
        :param batch: int 批次大小
        :param mu: float 学习率
        :param needG: bool 是否模拟硬件训练
        :param GinitValue: int 模拟硬件训练的初始值
        :param sd: float 初始化标准差大小
        """
        b = Network[1:]
        c = Network[:-1]
        d = np.ones(len(b), dtype=np.int)

        self.Weight = [np.random.normal(0, sd, i) for i in list(zip(b, c))]
        if Bias:
            self.Bias = [np.random.normal(0, sd*0.1, i) for i in list(zip(b, d))]
        else:
            self.Bias = False

        # weight = list(zip(b, c))
        # bias = list(zip(b, d))
        self.Network = Network
        self.X = X
        self.label = label
        self.mu = mu  # learning rate

        self.num_input_feature = X.shape[1]
        self.num_output_feature = label.shape[1]
        self.num_train_sample = X.shape[0]
        self.num_layer = len(self.Weight)

        self.y_eachlayer = []  # Every layer's value of output every times in ForwardPro process
        self.delta_eachlayer = []  # Every layer's value of delta which is using for propagation process

        if needG == True:
            self.W_G_p = [np.ones(i) * GinitValue for i in list(zip(b, c))]
            self.W_G_d = [np.random.normal(0, sd, i) + GinitValue for i in list(zip(b, c))]

            self.Weight = []

            if self.Bias:
                self.B_G_p = [np.ones(i) * GinitValue for i in list(zip(b, d))]
                self.B_G_d = [np.random.normal(0, sd, i) + GinitValue for i in list(zip(b, d))]

                self.Bias = []

            for i in range(self.num_layer):
                self.Weight.append(self.W_G_p[i] - self.W_G_d[i])
                if self.Bias:
                    self.Bias.append(self.B_G_p[i] - self.B_G_d[i])

        # use for save data
        # A = self.Weight[0]
        # B = self.Weight[1]
        # self.W_save = [[A, B]]

    def ForwardPro(self, X, beta=1):
        """
        前向传播过程
        """
        self.batch = len(X[0])
        self.y_eachlayer = [X]

        z = X  # z shape must be (feature, batch)
        for i in range(self.num_layer):
            if self.Bias:
                sum = np.dot(self.Weight[i], z) + self.Bias[i] * np.ones((1, self.batch))
            else:
                sum = np.dot(self.Weight[i], z)
            # print(sum)
            # z = sigmoid(sum)   # 11.513/np.average(np.abs(sum))
            z = sigmoid(sum, beta)
            # z = softmax(np.dot(self.Weight[i], z) + self.Bias[i] * np.ones((1, self.batch)))
            self.y_eachlayer.append(z)
        return None

    def GetDelta(self, label):
        self.delta_eachlayer = ["None"]  # The 1st layer no delta
        d = (self.y_eachlayer[-1] - label) * self.y_eachlayer[-1] * ( 1 - self.y_eachlayer[-1])
        value = [d]

        for i in range(self.num_layer - 1):
            d = np.dot(self.Weight[-1 - i].T, d) * self.y_eachlayer[-2 - i] * ( 1 - self.y_eachlayer[-2 - i])
            value.append(d)

        for j in range(len(value)):
            self.delta_eachlayer.append(value[-1 - j])
        # return self.delta_eachlayer
        return None

    def BackPropagation(self, Wupdate=True):
        if Wupdate:
            self.dW = []
            self.dB = []
            for i in range(self.num_layer):
                dW = np.dot(self.delta_eachlayer[i + 1], self.y_eachlayer[i].T) / self.batch
                new_W = self.Weight[i] - self.mu * dW
                self.Weight[i] = new_W
                self.dW.append(dW)

                if self.Bias:
                    dB = np.dot(self.delta_eachlayer[i + 1], np.ones((self.batch, 1))) / self.batch
                    new_B = self.Bias[i] - self.mu * dB
                    self.Bias[i] = new_B
                    self.dB.append(dB)

        else:
            # use for MemUpdate
            self.dW = []
            self.dB = []
            for i in range(self.num_layer):
                dW = np.dot(self.delta_eachlayer[i + 1], self.y_eachlayer[i].T) / self.batch
                new_W = self.Weight[i] - self.mu * dW
                self.Weight[i] = new_W
                self.dW.append(dW)

                if self.Bias:
                    dB = np.dot(self.delta_eachlayer[i + 1], np.ones((self.batch, 1))) / self.batch
                    new_B = self.Bias[i] - self.mu * dB
                    self.Bias[i] = new_B
                    self.dB.append(dB)
        return None

    def Train(self, X, label):
        self.ForwardPro(X)
        self.GetDelta(label)
        self.BackPropagation()

        devide = (self.y_eachlayer[-1] - label)
        loss = np.sum(devide * devide / 2) / (self.batch * len(devide))
        return loss

    def GetAccuracy(self, X, label, beta=1):
        li = X.shape[1]
        z = X
        for i in range(self.num_layer):
            if self.Bias:
                sum = np.dot(self.Weight[i], z) + self.Bias[i] * np.ones((1, li))
            else:
                sum = np.dot(self.Weight[i], z)
            # z = sigmoid( sum)
            z = sigmoid(sum, beta)
            # z = softmax(np.dot(self.Weight[i], z) + self.Bias[i] * np.ones((1, li)))

        a = z.argmax(axis=0)
        b = label.argmax(axis=0)
        num = len(a)
        acc = np.sum(np.equal(a, b) + 0) / num

        print(a)
        print(b)
        return acc

    def ShuffleDataset(self, X, label):
        index = np.arange(self.num_train_sample)
        np.random.shuffle(index)

        X = X[index]
        label = label[index]
        return X, label

# ----------------------  Mem Conductance State Update ----------------------- #
    def GetIndex(self):
        self.index_Gp = []
        self.index_Gd = []

        if self.Bias:
            self.index_Bp = []
            self.index_Bd = []

        for layer in range(self.num_layer):
            index_Gp = np.zeros(self.Weight[layer].shape, dtype=np.int)
            self.index_Gp.append(index_Gp)
            index_Gd = np.zeros(self.Weight[layer].shape, dtype=np.int)
            self.index_Gd.append(index_Gd)

            if self.Bias:
                index_Bp = np.zeros(self.Bias[0].shape, dtype=np.int)
                self.index_Bp.append(index_Bp)
                index_Bd = np.zeros(self.Bias[0].shape, dtype=np.int)
                self.index_Bd.append(index_Bd)

