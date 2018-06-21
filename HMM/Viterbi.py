#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 20:13:04
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

import numpy as np


# 需要为每层状态建立两个列表，分别是概率列表和最大概率对应的前一个节点的列表
def Viterbi(start_probability, transition_probability, emission_probability, observations):
    bestWay = []
    probList = []
    nodeList = []
    for t in range(len(observations)):
        prob = []
        node = []
        for i in range(len(states)):
            if (t == 0):
                prob.append(start_probability[i] * emission_probability[i][obseIndex[observations[t]]])
                node.append(0)
            else:
                prob.append(np.max(
                    np.array(probList[t - 1]) * np.array(transition_probability)[:, i] * emission_probability[i][
                        obseIndex[observations[t]]]))
                node.append(np.argmax(np.array(probList[t - 1]) * np.array(transition_probability)[:, i]))
        probList.append(prob)
        nodeList.append(node)
    for t in reversed(range(len(observations))):
        if (t == len(observations) - 1):
            bestWay.append(np.argmax(probList[t]))
        else:
            bestWay.append(nodeList[t + 1][bestWay[t - 1]])
    bestWay = list(reversed(bestWay))
    statusList = [states[i] for i in bestWay]
    print('状态序列为：{}'.format(statusList))
    return statusList


if __name__ == '__main__':
    states = ('1', '2', '3')

    observations = ('红', '白', '红')
    obseIndex = {'红': 0, '白': 1}
    start_probability = [0.2, 0.4, 0.4]

    transition_probability = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    emission_probability = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Viterbi(start_probability, transition_probability, emission_probability, observations)
