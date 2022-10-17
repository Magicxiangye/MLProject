"""
Created on 7 29, 2022
Adaboost is short for Adaptive Boosting
@author: magicye18
"""
import numpy as np
import matplotlib.pyplot as plt


# 简单的单层决策树测试矩阵
def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    # 简单数据集的分类
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# 老生常谈的读取文件函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 单层决策树的分类函数
# input
# matrix , 特征， 阈值， 划分方式
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    # 不等号的选取
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    # 返回分类的向量
    return retArray


# 最佳单层决策树的构建函数
# input---D：权重向量
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T  # shape is (m, 1)
    m, n = np.shape(dataMatrix)
    # 分箱个数（数值型的特征使用最大最小值结合分箱个数来确定可以划分的阈值）
    numSteps = 10.0
    # 保存最佳的单层决策树信息
    bestStump = {}
    # 最佳的分类结果
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 初始化最小错误率-- 直接无限大
    # 三层嵌套循环生成最佳的单层决策树
    for i in range(n):  # 特征数
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 确定步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # 特征下可以取到的阈值
            # 不等式的选取方式（大于小于阈值的类别划分）
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 验证分类的错误率
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 加权错误值
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightedError))
                # 判断是否为最佳的单层分类函数
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # return 树、最小错误值、最佳的分类结果
    return bestStump, minError, bestClasEst


# train function
# numInt 迭代的次数
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    # 弱分类器数组
    weakClassArr = []
    # 样本量
    m = np.shape(dataArr)[0]
    # 初始化样本权重向量
    D = np.mat(np.ones((m, 1)) / m)
    # 每个样本的类别估计累计值
    # 当累计值都分类正确（错误率为0时，停止迭代）
    aggClassEst = np.mat(np.zeros((m, 1)))
    # training
    for i in range(numIt):
        # build decision stump
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # 计算分类器的权重
        # 避免0溢出 max(error, 1e-16)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)

        # 迭代当前的权重向量
        # np.multiply 逐元素的相乘
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 每个样本的估计累计值
        aggClassEst += alpha * classEst
        # print("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        # 训练错误率
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)

        if errorRate == 0.0:
            break

    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    # 测试集
    dataMatrix = np.mat(datToClass)
    # 样本个数
    m = np.shape(dataMatrix)[0]
    # 样本估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 每个分类器的累计分类
    for i in range(len(classifierArr)):
        # 每个弱分类器的预测
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        # 累计
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    # 预测出最后的结果
    return np.sign(aggClassEst)


# roc曲线以及AUC面积
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)  # 画图坐标点
    ySum = 0.0  # calculate AUC
    numPosClas = sum(np.array(classLabels) == 1.0)  # 正样本个数
    yStep = 1 / float(numPosClas)  # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)  # x轴步长
    # 预测强度的排序--从小到大
    sortedIndicies = predStrengths.argsort()

    # draw figure
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历每一个点
    # 这里是二分类的问题（或者是某个重要类别的判定曲线）
    for index in sortedIndicies.tolist()[0]:
        # 样本为正，降低真阳率
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        # 降低假阳率
        else:
            delX = xStep
            delY = 0
            # 累计高，算近似的曲线下面积（AUC）
            ySum += cur[1]
        # 点的移动连线
        # ax.plot([x1, x2], [y1, y2], 线样式)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        # 当前点的变化
        cur = (cur[0] - delX, cur[1] - delY)
    # 随机猜测的结果曲线
    ax.plot([0, 1], [0, 1], 'b--')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()

    print("the Area Under the Curve is: ", ySum * xStep)
