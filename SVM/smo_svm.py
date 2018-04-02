#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-29 17:06:57
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import math


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    return dataMat, labelMat


# 定义一个类作为数据结构，存储一些关键值
class classSMO:
    def __init__(self, dataMat, labelMat, C, tol):  # Initialize the structure with the parameters
        self.dataMat = dataMat
        self.labelMat = labelMat
        self.C = C
        self.tol = tol
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 1)))  # first column is valid flag
        self.kernelType = 'linear'
        # self.kernelType = 'Gaussian'
        # 储存K(xi,xj)的值
        self.kernelFunctionValue = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.kernelFunctionValue[:, i] = kernel(i, self)


def kernel(i, objectSMO):
    if (objectSMO.kernelType == 'linear'):
        return np.dot(objectSMO.dataMat[:, :], objectSMO.dataMat[i, :].transpose())
    if (objectSMO.kernelType == 'Gaussian'):
        sigma = 1
        temp = -np.square(objectSMO.dataMat[:, :] - objectSMO.dataMat[i, :]) / (2 * math.pow(sigma, 2))
        ndarr = np.array(temp)
        ndarr = np.sum(ndarr, axis=1)
        temp = np.mat(ndarr).transpose()
        return np.exp(temp)


# 计算g(xi)的值
def computeGx(i, objectSMO):
    if (objectSMO.kernelType == 'linear'):
        g_xi1 = np.multiply(objectSMO.alphas, objectSMO.labelMat)
        g_xi2 = np.multiply(objectSMO.dataMat, objectSMO.dataMat[i])
        g_xi = np.sum(np.multiply(g_xi1, g_xi2)) + objectSMO.b
    elif (objectSMO.kernelType == 'Gaussian'):
        g_xi = np.multiply(objectSMO.alphas, objectSMO.labelMat)
        g_xi = np.multiply(g_xi, objectSMO.kernelFunctionValue[:, i])
        g_xi = np.sum(g_xi) + objectSMO.b
    else:
        g_xi = 0
    return g_xi


# 计算Ei的值
def computeEk(k, objectSMO):
    print('computeEi')
    g_xi = computeGx(k, objectSMO)
    return (g_xi - objectSMO.labelMat[k]).item()


# 检查误分类的个数
def checkClassifierErrorSample(objectSMO):
    errorNum = 0
    for i in range(objectSMO.dataMat.shape[0]):
        predi = np.sign(computeGx(i, objectSMO))
        yi = objectSMO.labelMat[i].item()
        if (predi != yi):
            errorNum += 1
    print('核函数为：{}, 样本总数为：{}，分类错误的样本个数为：{}'.format(objectSMO.kernelType, objectSMO.m, errorNum))


# 绘制原始数据
def plotData(Xmat, ymat):
    Xarray = np.array(Xmat)
    yarray = np.array(ymat)
    col = {+1: 'r', -1: 'b'}
    plt.figure()
    for i in range(Xarray.shape[0]):
        plt.plot(Xarray[i, 0], Xarray[i, 1], col[yarray[i][0]] + 'o')
    plt.show()


# 更新Ei
def updateEk(objectSMO, k):
    objectSMO.eCache[k] = computeEk(k, objectSMO)


# 更新alpha1和alpha2
def takeStep(i, j):
    if (i == j):
        return 0
    # y1 y2 E1 E2为标量形式
    y1 = objectSMO.labelMat[i].item()
    y2 = objectSMO.labelMat[j].item()
    E1 = computeEk(i, objectSMO)
    E2 = computeEk(j, objectSMO)
    # alpha1old alpha2old为标量形式
    alpha1old = objectSMO.alphas[i].item()
    alpha2old = objectSMO.alphas[j].item()
    s = y1 * y2
    if (labelMat[i] != labelMat[j]):
        L = max(0, alpha2old - alpha1old)
        H = min(C, C + alpha2old - alpha1old)
    else:
        L = max(0, alpha2old + alpha1old - C)
        H = min(C, alpha2old + alpha1old)
    if (L == H):
        return 0
    # Kij为标量形式
    K11 = objectSMO.kernelFunctionValue[1, 1]
    K22 = objectSMO.kernelFunctionValue[2, 2]
    K12 = objectSMO.kernelFunctionValue[1, 2]
    eta = K11 + K22 - 2 * K12
    if (eta > 0):
        alpha2New = alpha2old + y2 * (E1 - E2) / eta
        if (alpha2New < L):
            alpha2New = L
        elif (alpha2New > H):
            alpha2New = H
    else:
        print('eta<=0')
        return 0
    if (np.abs(alpha2New - alpha2old) < 0.00001):
        print('alpha2 step size too small')
        return 0
    alpha1New = alpha1old + s * (alpha2old - alpha2New)
    b1 = -E1 - y1 * K11 * (alpha1New - alpha1old) - y2 * K12 * (alpha2New - alpha2old) + objectSMO.b
    b2 = -E2 - y1 * K12 * (alpha1New - alpha1old) - y2 * K22 * (alpha2New - alpha2old) + objectSMO.b
    # update
    objectSMO.b = (b1 + b2) / 2
    updateEk(objectSMO, i)
    updateEk(objectSMO, j)
    objectSMO.alphas[i] = alpha1New
    objectSMO.alphas[j] = alpha2New
    return 1


# 选择第2个变量
def secondChoice(i1, objectSMO):
    maxAbsEDelta = 0
    i2OfMaxAbsEDelta = 0
    Ei1 = objectSMO.eCache[i1].item()
    for i2 in range(objectSMO.eCache.shape[0]):
        Ei2 = objectSMO.eCache[i2].item()
        absEDelta = np.abs(Ei1 - Ei2)
        if (absEDelta > maxAbsEDelta):
            maxAbsEDelta = absEDelta
            i2OfMaxAbsEDelta = i2
    return i2OfMaxAbsEDelta


# 内层循环
def examineExample(i1, objectSMO, C):
    y1 = objectSMO.labelMat[i1]
    alpha1Old = objectSMO.alphas[i1]
    E1 = computeEk(i1, objectSMO)
    r1 = E1 * y1
    if ((r1 < -tol and alpha1Old < C) or (r1 > tol and alpha1Old > 0)):
        set1 = set(np.where(objectSMO.alphas > 0)[0])
        set2 = set(np.where(objectSMO.alphas < C)[0])
        indexAlphaNot0andC = set1 & set2
        if (len(indexAlphaNot0andC) > 1):
            i2 = secondChoice(i1, objectSMO)
            if (takeStep(i1, i2)):
                return 1
        # loop over all no-bound alpha,starting at a random point
        indexAlphaNot0andCList = list(indexAlphaNot0andC)
        shuffle(indexAlphaNot0andCList)
        for i2 in indexAlphaNot0andCList:
            if (takeStep(i1, i2)):
                return 1
        # loop over all possible i1,starting at a random point
        indexAllDataSet = list(range(objectSMO.dataMat.shape[0]))
        shuffle(indexAllDataSet)
        for i2 in indexAllDataSet:
            if (takeStep(i1, i2)):
                return 1
    return 0


# 外循环
def mainRoutine(dataMat, C, maxIter):
    m, n = np.shape(dataMat)
    numChanged = 0
    examineAll = True
    iterNum = 0
    while ((iterNum < maxIter) and (numChanged > 0 or examineAll)):
        numChanged = 0
        if (examineAll):
            for i in range(m):
                numChanged += examineExample(i, objectSMO, C)
            iterNum += 1
        else:
            set1 = set(np.where(objectSMO.alphas > 0)[0])
            set2 = set(np.where(objectSMO.alphas < C)[0])
            indexAlphaNot0andC = set1 & set2
            for i in indexAlphaNot0andC:
                numChanged += examineExample(i, objectSMO, C)
            iterNum += 1
        if (examineAll == True):
            examineAll = False
        elif (numChanged == 0):
            examineAll = True


# 计算linear的权重
def computeTheta(objectSMO):
    theta = np.dot(np.multiply(objectSMO.alphas, objectSMO.labelMat).transpose(), objectSMO.dataMat)
    b = objectSMO.b
    thetaList = theta.tolist()
    thetaList[0].append(b)
    theta = np.mat(thetaList).transpose()
    return theta


# 绘制核为linear的超平面
def plotHyperplaneLinear(Xmat, ymat, theta, objectSMO):
    m, n = np.shape(Xmat)
    Xarray = np.array(Xmat)
    yarray = np.array(ymat)
    col = {+1: 'r', -1: 'b'}
    plt.figure()
    for i in range(Xarray.shape[0]):
        plt.plot(Xarray[i, 0], Xarray[i, 1], col[yarray[i][0]] + 'o')
    plt.ylim([-6, 6])
    plt.plot(Xarray[:, 0],
             (-(theta[0][0] * Xarray[:, 0] + np.multiply(theta[2][0], np.ones((m, 1)))) / theta[1][0]).transpose(),
             c='g')
    # 标出支持向量
    for i in range(objectSMO.m):
        if (round(objectSMO.alphas[i].item(), 3) < objectSMO.C and round(objectSMO.alphas[i].item(), 3) > 0):
            plt.plot(Xarray[i, 0], Xarray[i, 1], 'y' + 'o')
    plt.show()


if __name__ == '__main__':
    '''
    通过更改此处文件名，以及类别classSMO中成员kernelType，切换linear kernel和RBF kernel
    '''
    fileName = 'testSetLinear.txt'
    # fileName = 'testSetRBF.txt'
    dataMat, labelMat = loadDataSet(fileName)
    print(np.shape(dataMat))
    C = 200
    tol = 0.001
    maxIter = 200
    objectSMO = classSMO(dataMat, labelMat, C, tol)
    mainRoutine(dataMat, C, maxIter)
    plotData(objectSMO.dataMat, objectSMO.labelMat)
    theta = computeTheta(objectSMO)
    plotHyperplaneLinear(objectSMO.dataMat, objectSMO.labelMat, theta, objectSMO)
    checkClassifierErrorSample(objectSMO)
