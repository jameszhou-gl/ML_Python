#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-21 14:43:14
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

import numpy as np
import random as random
import matplotlib.pyplot as plt


def batchGradientDescent(alpha, XMatrix, yMatrix, iterNum):
    m, n = np.shape(XMatrix)
    theta = np.ones((n, 1))
    for i in range(iterNum):
        error = yMatrix - np.dot(XMatrix, theta)
        # 损失函数的值
        # loss = sum(pow(np.array(error), 2)) * 2 / m
        # print(loss)
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


def loadDataSet(filename):
    Xlist = []
    ylist = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        Xlist.append([round(float(lineArr[0]), 2), 1])
        ylist.append(round(float(lineArr[-1]), 2))
    Xarray = np.array(Xlist)
    yarray = np.array(ylist)
    return Xarray, yarray


def plot(Xmat, ymat, thetaUpdate):
    Xarray = np.array(Xmat)
    yarray = np.array(ymat)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Linear Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(Xarray[:, 0], yarray, c='b', marker='o')
    plt.plot(Xarray[:, 0], np.dot(Xmat, thetaUpdate).tolist(), c='r')
    plt.show()
