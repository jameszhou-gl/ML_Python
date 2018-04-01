#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-25 10:15:12
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

'''
使用sklearn中make_moons数据，构建了个LR二分类模型
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def logistic(alpha, XMatrix, yMatrix, iterNum):
    m, n = np.shape(XMatrix)
    print(m)
    print(n)
    theta = np.ones((n, 1))
    loss_list = []
    for i in range(iterNum):
        hypothsis = np.squeeze(np.asarray(np.dot(XMatrix, theta)))
        hypothsis = sigmoid(hypothsis)
        loss = 0
        for j in range(m):
            if (yMatrix[j][0] == 1):
                loss += np.log(hypothsis[j])
            else:
                loss += np.log(1 - hypothsis[j])
        loss = -loss / m
        loss_list.append(loss)
        error = np.mat(hypothsis).transpose() - yMatrix
        theta = theta - alpha * np.dot(XMatrix.transpose(), error)
    return theta, loss_list


def plotLossFunction(loss_list, iterNum):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Loss Value')
    plt.xlabel('iter num')
    plt.ylabel('loss')
    plt.plot(list(range(iterNum)), loss_list, c='r')
    plt.show()


def plotLR(Xmat, ymat, thetaUpdate):
    Xarray = np.array(Xmat)
    yarray = np.array(ymat)
    col = {0: 'r', 1: 'b'}
    plt.figure()
    for i in range(Xarray.shape[0]):
        plt.plot(Xarray[i, 0], Xarray[i, 1], col[yarray[i][0]] + 'o')
    plt.ylim([-1.5, 1.5])
    plt.plot(Xarray[:, 0],
             (-(thetaUpdate[0][0] * Xarray[:, 0] + thetaUpdate[2][0] * Xarray[:, 2]) / thetaUpdate[1][0]).transpose(),
             c='g')
    plt.show()


if __name__ == '__main__':
    x, y = make_moons(250, noise=0.25)
    XarrayLess = x
    x = np.ones((XarrayLess.shape[0], XarrayLess.shape[1] + 1))
    x[:, :-1] = XarrayLess
    x = np.mat(x)
    y = np.mat(y).transpose()
    alpha = 0.01
    iterNum = 500
    theta, loss_list = logistic(alpha, x, y, iterNum)
    print(theta.shape)
    plotLR(x, y, theta)
    plotLossFunction(loss_list, iterNum)
