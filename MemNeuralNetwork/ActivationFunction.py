#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "H.YL"

import numpy as np

def sigmoid(z, beta=1):
    """
    Active function which is sigmoid
    :param z: could be an array
    :return: Active out
    """
    return 1. / (1. + np.exp(-z*beta))