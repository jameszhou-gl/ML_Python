#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-21 14:43:14
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

import os
import numpy as np
import pandas as pd


def compute_loss_function():
    print('aa')


def loadDataSet():
    XMat = []
    yMat = []
    fr = open('data.csv')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        XMat.append(float(lineArr[0]))
        yMat.append(float(lineArr[1]))
    return XMat, yMat


if __name__ == '__main__':
    learning_rate = 0.1
    XMat, yMat = loadDataSet()
    dataMatrix = np.mat(XMat)
    yMatrix = np.mat(yMat)
