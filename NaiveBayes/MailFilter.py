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
先对EnglishMailFile.txt每行提取label(ham和spam),并进行分词，去除标点符号且对单词小写处理，生成EnglishMailFileMap.txt
对EnglishMailFileMap.txt做reduce操作，统计各个词在正常垃圾和正常邮箱的词频，放在result.txt
如下所示，依次为word，在垃圾邮件出现次数，在正常邮件出现次数
Free,43,4
'''


def loadDataSet():
    regEx = re.compile('\\W*')
    wordSpamList = []
    wordHamList = []
    numSpamMail = 0
    numHamMail = 0
    with open("EnglishMailFile.txt", "r") as file:
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
        print('有{}封垃圾邮件'.format(numSpamMail))
        print('有{}封正常邮件'.format(numHamMail))
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

    TopK = 5
    spamLogprobaResult = 0
    for logproba in spamProbaList[:TopK]:
        spamLogprobaResult += logproba
    print(spamLogprobaResult)
    print(math.exp(spamLogprobaResult))

    hamLogProbaResult = 0
    for logproba in hamProbaList[:TopK]:
        hamLogProbaResult += logproba
    print(hamLogProbaResult)
    print(math.exp(hamLogProbaResult))


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
    testMail('testMail.txt')
