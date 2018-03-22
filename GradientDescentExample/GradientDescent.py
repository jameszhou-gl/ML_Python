#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-21 14:43:14
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

import numpy as np
import random as random


def batchGradientDescent(alpha, XMatrix, yMatrix, iterNum):
    m, n = np.shape(XMatrix)
    theta = np.ones((n, 1))
    for i in range(iterNum):
        error = yMatrix - np.dot(XMatrix, theta)
        theta = theta + alpha * np.dot(error.transpose(), XMatrix).transpose() / m
    return theta


def stochasticGradientDescent(alpha, XMatrix, yMatrix, iterNum):
    m, n = np.shape(XMatrix)
    theta = np.ones((n, 1))
    randomNum = random.randint(0, m - 1)
    for i in range(iterNum):
        error = yMatrix[randomNum] - np.dot(XMatrix[randomNum], theta)
        theta = theta + alpha * (error * XMatrix[randomNum]).transpose()
    return theta


def miniBatchGradientDescent(alpha, XMatrix, yMatrix, iterNum, batch):
    m, n = np.shape(XMatrix)
    theta = np.ones((n, 1))
    indexList = list(range(m))
    slice = random.sample(indexList, batch)
    XMatrix = XMatrix[slice]
    yMatrix = yMatrix[slice]
    for i in range(iterNum):
        error = yMatrix - np.dot(XMatrix, theta)
        theta = theta + alpha * (error.transpose() * XMatrix).transpose()
    return theta


def loadDataSet():
    Xlist = []
    ylist = []
    fr = open('data2.csv')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        Xlist.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        ylist.append(float(lineArr[-1]))
    Xmat = np.mat(np.array(Xlist))
    ymat = np.mat(np.array(ylist)).transpose()
    return Xmat, ymat


if __name__ == '__main__':
    alpha = 0.1
    Xmat, ymat = loadDataSet()
    print(batchGradientDescent(alpha, Xmat, ymat, 100))
    print(stochasticGradientDescent(alpha, Xmat, ymat, 100))
    print(miniBatchGradientDescent(alpha, Xmat, ymat, 100, 4))
