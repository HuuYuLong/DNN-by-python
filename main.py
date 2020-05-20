#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "H.YL"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MemNeuralNetwork import *


def main_PureSoftware():

    epoch_array = []
    acc_array = []
    loss_array = []
    for epoch in range(400):
        loss_all = 0
        #
        a = int( NN.num_train_sample/ batch)
        for i in range(a):
            loss = NN.Train(train_ima[i * batch: (i+1) * batch].T , train_lab[i * batch: (i+1) * batch].T)
            loss_all = loss_all + loss

        acc = NN.GetAccuracy(test_ima.T, test_lab.T)

        epoch_array.append(epoch)
        acc_array.append(acc)
        loss_array.append(loss_all)

        print(loss_all)
        print(acc)

    # dic = {"epoch": epoch_array, "acc": acc_array, "loss_all": loss_array}
    # output_data = pd.DataFrame(dic)
    # output_data.to_csv("./accuracy.csv")


if __name__ == '__main__':
    batch = 220
    train_ima, train_lab, test_ima, test_lab = get_MnistData()  # Which row is data sample
    NN = NeuralNetwork([784, 200, 10],train_ima, train_lab, mu=1)
    main_PureSoftware()
