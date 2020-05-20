#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "H.YL"

import numpy as np
import pandas as pd
from mnist import MNIST


def one_hot(index_matrix, len_enconding):
    """
    #  把数组（可多维）转换为 one_hot 编码
    :param index_matrix:   数组
    :param len_enconding:  one_hot的长度
    :return: one_hot_arr
    """
    one_hot_arr = np.eye(len_enconding)[index_matrix]
    return one_hot_arr.astype(np.uint8)


def get_MnistData(dir='E:/graduate/article/my work/HW-NN/MNIST/'):
    mndata = MNIST(dir, return_type="numpy")
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = train_images / 255
    train_labels = one_hot(train_labels, 10)
    test_images = test_images/255
    test_labels = one_hot(test_labels, 10)

    return train_images, train_labels, test_images, test_labels    # 返回mnist所有的数据与标签用于测试


def get_HardwareData():
    """
    get the hardware data
    :return: dict.  "1.0v", "1.2v", "1.4v", "1.6v", "1.8v", "2.0v", "50ge", "100ge", "150ge", "200ge"
    """
    dir_path = "E:/graduate/article/my work/dataset2020/"
    file = ("DiferentVoltage", "DiferentNum", )
    Vds = 0.01 #v
    DiferentNum = ("1.0v", "1.2v", "1.4v", "1.6v", "1.8v", "2.0v", )
    DiferentVoltage = ("50ge", "100ge", "150ge", "200ge")

    a = pd.read_excel(dir_path+file[0]+".xlsx")
    b = pd.read_excel(dir_path+file[1]+".xlsx")

    DN_1_0v = a["1.0v"][0:51].values / Vds
    DN_1_2v = a["1.2v"][0:51].values / Vds
    DN_1_4v = a["1.4v"][0:51].values / Vds
    DN_1_6v = a["1.6v"][0:51].values / Vds
    DN_1_8v = a["1.8v"][0:51].values / Vds
    DN_2_0v = a["2.0v"][0:51].values / Vds

    DV_50ge = b["50ge"][0:51].values / Vds
    DV_100ge = b["100ge"][0:101].values / Vds
    DV_150ge = b["150ge"][0:151].values / Vds
    DV_200ge = b["150ge"][0:201].values / Vds

    HadwareData = {}
    HadwareData[0] = DN_1_0v
    HadwareData[1] = DN_1_2v
    HadwareData[2] = DN_1_4v
    HadwareData[3] = DN_1_6v
    HadwareData[4] = DN_1_8v
    HadwareData[5] = DN_2_0v
    HadwareData[6] = DV_50ge
    HadwareData[7] = DV_100ge
    HadwareData[8] = DV_150ge
    HadwareData[9] = DV_200ge

    return HadwareData


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False



# def get_MnistData(Use_pd=False):
#     """
#     get the MNIST dataset
#     :param    Use_pd:  default is Ture,
#               if Use_pd=True then return pd.DataFrame() data
#               else return np.array() data
#     :return:  train_images, train_labels, test_images, test_labels respectively
#     """
#     from tensorflow.examples.tutorials.mnist import input_data
#     mnist = input_data.read_data_sets("E:/ssstudy/mnist/", one_hot=True)
#     train_images = mnist.train.images
#     train_labels = mnist.train.labels
#     test_images = mnist.test.images
#     test_labels = mnist.test.labels
#     if Use_pd:
#         return pd.DataFrame(train_images), pd.DataFrame(train_labels), pd.DataFrame(test_images), pd.DataFrame(
#             test_labels)
#     else:
#         return train_images, train_labels, test_images, test_labels  # 返回mnist所有的数据与标签用于测试
