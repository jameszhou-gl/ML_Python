# coding: utf-8

from random import random
from math import log, exp, pow, fabs
from time import strftime
from multiprocessing import Pool, cpu_count
from sys import exit
import datetime
from sklearn.datasets import make_moons
import numpy as np


# 数据列数 = 1 + 132

# 自定义异常
class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# sigmoid函数
def sigmoid(X, m, pool):
    result = X
    for i in range(m):
        if X[i][0] > 100:
            result[i][0] = 1
        if X[i][0] < -100:
            result[i][0] = 0
        try:
            result[i][0] = 1 / (1 + exp(-X[i][0]))
        except Exception as e:
            print(e)
            print(X[i][0])
            exit(0)
    return result


# sigmoid函数（并行化）任务
def sigmoidTask(X):
    result = X
    m = len(X)
    for i in range(m):
        if X[i][0] > 100:
            result[i][0] = 1
        if X[i][0] < -100:
            result[i][0] = 0
        try:
            result[i][0] = 1 / (1 + exp(-X[i][0]))
        except Exception as e:
            print(e)
            print(X[i][0])
            exit(0)
    return result


# sigmoid函数（并行化）
def sigmoidP(X, m, pool):
    result = []
    cpu_cnt = cpu_count()
    results = []
    print('res')
    for i in range(cpu_cnt):
        res = pool.apply_async(sigmoidTask, (X[int(m * i / cpu_cnt):int(m * (i + 1) / cpu_cnt)][:],))
        print(res)
        results.append(res)
    for res in results:
        result.extend(res.get())
    return result


# 矩阵点乘
def dotMultiply(X, Y, m, n):
    X_row = m
    X_col = n
    Y_row = n
    Y_col = 1
    result = []
    for i in range(X_row):
        line = []
        for j in range(Y_col):
            ans = 0
            for a in range(X_col):
                ans += X[i][a] * Y[a][j]
            line.append(ans)
        result.append(line)
    return result


# theta转秩
def T(X):
    result = []
    for i in range(n):
        line = []
        line.append(X[i])
        result.append(line)
    return result


# 获取代价函数值
def getCost(m, n, lmd, h, theta, train_y, pool):
    ans = 0
    for i in range(m):
        if train_y[i] == 1:
            ans += log(h[i][0]) * (-1.0) / m
        else:
            ans += log(1.0 - h[i][0]) * (-1.0) / m
    regularzilation = 0
    for i in range(n):
        regularzilation += pow(theta[i], 2)
    regularzilation *= lmd / (2 * m)
    return ans + regularzilation


# 获取代价函数值（并行化）任务
def costTask(m, h, train_y):
    ans = 0
    n = len(h)
    for i in range(n):
        if train_y[i] == 1:
            ans += log(h[i][0]) * (-1.0) / m
        else:
            ans += log(1.0 - h[i][0]) * (-1.0) / m
    return ans


# 获取代价函数值（并行化）
def getCostP(m, n, lmd, h, theta, train_y, pool):
    ans = 0
    cpu_cnt = cpu_count()
    results = []
    for i in range(cpu_cnt):
        result = pool.apply_async(costTask, (m, h[int(m * i / cpu_cnt):int(m * (i + 1) / cpu_cnt)][:],
                                             train_y[int(m * i / cpu_cnt):int(m * (i + 1) / cpu_cnt)]))
        results.append(result)
    for result in results:
        ans += result.get()
    regularzilation = 0
    for i in range(n):
        regularzilation += pow(theta[i], 2)
    regularzilation *= lmd / (2 * m)
    return ans + regularzilation


# 获取新的theta
def getNewTheta(m, n, lmd, alpha, train_x, train_y, h, pool):
    global theta
    for i in range(n):
        gradient = 0
        for j in range(m):
            gradient += (h[j][0] - train_y[j]) * train_x[j][i]
        gradient -= lmd * theta[i]
        gradient *= ((- alpha) / m)
        theta[i] += gradient


# 获取新的theta（并行化）任务
def newThetaTask(m, lmd, alpha, train_x, train_y, h, theta):
    n = len(theta)
    ans = []
    for i in range(n):
        gradient = 0
        for j in range(m):
            try:
                z = train_x[j][i]
            except Exception as e:
                print(e)
                print(j, i)
                exit(0)
            gradient += (h[j][0] - train_y[j]) * z
        gradient -= lmd * theta[i]
        gradient *= ((- alpha) / m)
        ans.append(theta[i] + gradient)
    return ans


# 获取新的theta（并行化）
def getNewThetaP(m, n, lmd, alpha, train_x, train_y, h, pool):
    global theta
    answer = []
    cpu_cnt = cpu_count()
    results = []
    for i in range(cpu_cnt):
        result = pool.apply_async(newThetaTask, (m,
                                                 lmd,
                                                 alpha,
                                                 [line[int(n * i / cpu_cnt):int(n * (i + 1) / cpu_cnt)] for line in
                                                  train_x],
                                                 train_y,
                                                 h,
                                                 theta[int(n * i / cpu_cnt):int(n * (i + 1) / cpu_cnt)],))
        results.append(result)
    for result in results:
        answer.extend(result.get())
    theta = answer


if __name__ == "__main__":
    train_x, train_y = make_moons(250, noise=0.25)
    m, n = np.mat(train_x).shape
    # print(m)
    # print(n)
    # 一些参数的初始化
    alpha = 0.1
    threshold = 0.000001
    lmd = 0.5
    step = 50
    # theta = [0.5] * n
    theta = [random() for i in range(n)]

    # 训练模型
    startTime = datetime.datetime.now()
    print(strftime("%Y-%m-%d %H:%M:%S") + " 开始训练", flush=True)
    cost = 0
    # change = 1
    cnt = 0
    pool = Pool(cpu_count())
    iterNum = 1000
    while (iterNum > 0):
        # h = sigmoid(dotMultiply(train_x, T(theta), m, n), m, pool)
        # new_cost = getCost(m, n, lmd, h, theta, train_y, pool)
        # getNewTheta(m, n, lmd, alpha, train_x, train_y, h, pool)
        h = sigmoidP(dotMultiply(train_x, T(theta), m, n), m, pool)
        new_cost = getCostP(m, n, lmd, h, theta, train_y, pool)
        getNewThetaP(m, n, lmd, alpha, train_x, train_y, h, pool)
        # change = fabs(cost - new_cost)
        iterNum -= 1
        cost = new_cost
        cnt += 1
        if (cnt % step == 0):
            print('cost:', cost, flush=True)
    print('cost:', cost, flush=True)
    endTime = datetime.datetime.now()
    print(strftime("%Y-%m-%d %H:%M:%S") + " 训练结束，用时" + str((endTime - startTime).seconds) + "秒", flush=True)
    print(theta)
    pool.close()
    pool.join()
