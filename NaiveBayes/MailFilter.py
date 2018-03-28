#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-28 10:29:00
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $Id$

import re
from collections import defaultdict
import math

'''
先对SMSCollection.txt每行提取label(ham和spam),并进行分词，利用正则化去除标点符号，生成wordSpam.txt和wordHam.txt
对wordSpam.txt和wordHam.txt做reduce操作，统计各个词在正常垃圾和正常邮箱的词频，放在result.txt
如下所示，依次为word，在垃圾邮件出现次数，在正常邮件出现次数
Free,43,4
'''


def loadDataSet():
    regEx = re.compile('\\W*')
    wordSpamList = []
    wordHamList = []
    numSpamMail = 0
    numHamMail = 0
    with open("SMSCollection.txt", "r") as file:
        for line in file.readlines():
            wordString = line.strip()
            wordLine = regEx.split(wordString)
            if (wordLine[0] == 'spam'):
                numSpamMail += 1
                for word in wordLine[1:-1]:
                    wordSpamList.append(word)
            else:
                numHamMail += 1
                for word in wordLine[1:-1]:
                    wordHamList.append(word)
        print('训练集有{}封垃圾邮件'.format(numSpamMail))
        print('训练集有{}封正常邮件'.format(numHamMail))
        wordSpamList = [tok for tok in wordSpamList if (len(tok) > 0)]
        wordHamList = [tok for tok in wordHamList if (len(tok) > 0)]

    with open("wordSpam.txt", "w") as file:
        for word in wordSpamList:
            file.write(word)
            file.write('\n')

    with open("wordHam.txt", "w") as file:
        for word in wordHamList:
            file.write(word)
            file.write('\n')

    wordDict = defaultdict(lambda: defaultdict(lambda: 0))
    with open("wordSpam.txt", 'r') as file:
        for line in file.readlines():
            word = line.strip()
            wordDict[word]['Spam'] += 1

    with open("wordHam.txt", 'r') as file:
        for line in file.readlines():
            word = line.strip()
            wordDict[word]['Ham'] += 1

    with open("result.txt", 'w') as file:
        for word in wordDict:
            file.write(word)
            file.write(',')
            file.write(str(wordDict[word]['Spam']))
            file.write(',')
            file.write(str(wordDict[word]['Ham']))
            file.write('\n')

    return wordDict, numSpamMail, numHamMail


# 测试某封邮件是否是垃圾邮件
def testMail(testFileName):
    regEx = re.compile('\\W*')
    wordTestList = []
    with open(testFileName, "r") as file:
        for line in file.readlines():
            wordString = line.strip()
            wordLine = regEx.split(wordString)
            for word in wordLine:
                wordTestList.append(word)
        wordTestList = [tok for tok in wordTestList if (len(tok) > 0)]
    spamProbaList = []
    hamProbaList = []
    # 对于测试邮件的每个word都计算其条件概率，并保存在probaList中
    for wordTest in wordTestList:
        spamProbaList.append(probaCompute(wordDict, wordTest, numSpamMail, numHamMail)[0])
        hamProbaList.append(probaCompute(wordDict, wordTest, numSpamMail, numHamMail)[1])

    TopK = 15
    spamLogProbaResult = 0
    for logproba in spamProbaList[:TopK]:
        spamLogProbaResult += logproba
    hamLogProbaResult = 0
    for logproba in hamProbaList[:TopK]:
        hamLogProbaResult += logproba
    spamProbaResult = math.exp(spamLogProbaResult)
    hamProbaResult = math.exp(hamLogProbaResult)
    return spamProbaResult, hamProbaResult


# 返回某个词对应垃圾邮件和正常邮件的概率
def probaCompute(wordDict, wordTest, numSpamMail, numHamMail):
    if (wordTest not in wordDict.keys()):
        probaWordOfSpam = 0.5
        probaWordOfHam = 0.5
    else:
        probaWordOfSpam = wordDict[wordTest]['Spam'] / numSpamMail
        if (probaWordOfSpam == 0):
            probaWordOfSpam += 1
        probaWordOfHam = wordDict[wordTest]['Ham'] / numHamMail
        if (probaWordOfHam == 0):
            probaWordOfHam += 1

    probaHam = numHamMail / (numSpamMail + numHamMail)
    probaSpam = numSpamMail / (numSpamMail + numHamMail)
    probaWord = probaWordOfSpam * probaSpam + probaWordOfHam * probaHam

    return math.log(probaWordOfSpam * probaSpam / probaWord), math.log(probaWordOfHam * probaSpam / probaWord)


if __name__ == '__main__':
    wordDict, numSpamMail, numHamMail = loadDataSet()
    spamProbaResult, hamProbaResult = testMail('testMail.txt')
    print("该测试邮件为垃圾邮件概率为{}".format(spamProbaResult / (spamProbaResult + hamProbaResult)))
    print("该测试邮件为正常邮件概率为{}".format(hamProbaResult / (spamProbaResult + hamProbaResult)))
