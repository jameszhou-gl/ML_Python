#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-23 13:42:03
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

from GradientDescentExample.GradientDescent import *

if __name__ == '__main__':
    alpha = 0.0001
    Xarray, yarray = loadDataSet('data1.csv')
    Xmat = np.mat(Xarray)
    ymat = np.mat(yarray).transpose()
    thetaUpdate = batchGradientDescent(alpha, Xmat, ymat, 50)
    print(thetaUpdate)
    plot(Xmat, ymat, thetaUpdate)
