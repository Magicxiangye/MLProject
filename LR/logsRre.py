"""
Created on 9 03, 2022
Logistic Regression Working Module
@author: magicye18
"""
import numpy as np
import matplotlib.pyplot as plt


# read files
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# 激活函数（跃迁函数）
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# 梯度上升的优化流程
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose()  # 转置成为列向量
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 初始化回归参数的设置
    # 迭代优化
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)  # 每个样本的误差情况
        weights = weights + alpha * dataMatrix.transpose() * error  # 公式
    return weights


# 计算出回归方程画出可视化边界
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]  # 样本量
    # type_01
    xcord1 = []
    ycord1 = []
    # type_02
    xcord2 = []
    ycord2 = []
    # 分别画出每个类别的散点图
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 决策边界
    # 根据回归方程得出两特征之间的关系式
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    y = np.matrix(y).transpose()  # 转置
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度下降
def stocGradAscent0(dataMatrix, classLabels):
    # 转化为矩阵
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)  # 初始化各特征的系数
    # 对数据集中的每一个样本进行训练，更新各回归系数的值
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度下降算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 类型的转化
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)  # initializing
    for j in range(numIter):  # 迭代次数
        # 样本集index的list
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # 改进点1 步长的不断调整
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 改进点2 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])  # 不放回抽样选取
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print
    "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print
    "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))
