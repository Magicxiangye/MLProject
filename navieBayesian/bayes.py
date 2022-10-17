# -*- coding: utf-8 -*-
"""
Created on 6 13, 2022
content: Navie Bayesian
@author: magicye18
"""
from numpy import *
import re


def loadDataSet():
    # 测试的语句矩阵（实验样本）
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性词条 0 代表正常言论词条
    return postingList, classVec


# 词典List
def createVocabList(dataSet):
    vocabSet = set([])  # 空set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个set的并集（集合的合并操作）
    return list(vocabSet)


# 检验输入的文档中的单词是否存在于词典vocablist中
# 词集模型--将每个词的出现与否作为一个特征
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 每一个输出的是等词典长度的向量，若存在相应的位置为1，不存在为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 训练朴素贝叶斯方法的条件概率向量
# 输入的是训练矩阵，以及矩阵中每篇文档对应的分类标签向量
# 这里是二分类的朴素贝叶斯，所以计算的是p1的概率， p0 = 1- p1
def trainNB0(trainMatrix, trainCategory):
    # 文档个数，以及词典向量的长度
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 初始化概率
    # 属于1的标签数量 / 文档总数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 每个类别下，每个词的出现次数
    # 每个词的条件概率相乘，才是总词向量的条件概率，当一个为零时概率为零
    # 因此初始值都设置为1
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 各类别出现的总词条数量
    # 初始化为2
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 计算的是类别1的概率
        if trainCategory[i] == 1:
            # 矩阵的相加
            # 类别1中，各词条出现的次数
            p1Num += trainMatrix[i]
            # 总次数
            p1Denom += sum(trainMatrix[i])
        else:
            # 矩阵的相加
            # 类别0中，各词条出现的次数
            p0Num += trainMatrix[i]
            # 总次数
            p0Denom += sum(trainMatrix[i])
    # 套上log,为后续的预测时，条件概率中累乘变累加提供方便
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
# 这里是二分类的函数，多分类也是同样的步骤，计算每个类别下的条件概率向量来进行预测
# 选取最后概率最大的类别
# 需要分类的词特征向量，p0类条件概率向量
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 相当于贝叶斯公式的分子
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # 分母都相同就不用除了，直接进行比较
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数
# 封装了朴素贝叶斯的所有流程
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    # 各类别下各特征的条件概率向量，以及各类别的初始概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # 两个测试函数
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 词袋模型--将每个词出现的次数作为一个特征
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 统计的是每个词出现的次数
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 文本分类
# 数据预处理函数-文本的词切割函数
# 只有基础的切割单词以及大小写转化的功能
def textParse(bigString):
    # 正则的文本切割的方式
    # 切割符号匹配任意非单词数字字符 \W
    listOfTokens = re.split(r'\W', bigString)
    # 统一词字典格式，同时去掉空字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 垃圾邮件检测函数
def spamTest():
    # 词典list
    docList = []
    # 类别list
    classList = []
    # 出现的全部词
    fullText = []

    # 一共拿50份邮件进行测试
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding='ISO-8859-1').read())
        # 每份邮件中的词列表
        docList.append(wordList)
        # 将新的邮件词使用expend list拆分后加入新的list
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)  # create 词典
    # 训练集数据列表的Index
    trainingSet = list(range(50))

    # 随机选取测试数据
    testSet = []  # create test set
    for i in range(10):
        # random.uniform将随机生成一个实数，它在 [x,y] 范围内。
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    # 训练数据
    trainMat = []
    trainClasses = []
    # 向量化
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 训练贝叶斯的参数
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 进行预测
    # 计算错误率
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


# 计算统计最常用的30个词汇
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []

    # 使用RSS源作为数据的来源
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 建立词典
    vocabList = createVocabList(docList)  # create vocabulary
    # 统计最常用的30个词
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words

    # 排除词典中的30个词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    # 这里和垃圾邮件的代码是一样的
    trainingSet = range(2 * minLen)
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


# 返回每一个地区的Top词汇
# 排序的依据是每一个地区的词汇出现的条件概率的大小
def getTopWords(ny, sf):
    # 使用载入这两个地区的函数方法
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    # 返回大于某个阈值的词的list
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
