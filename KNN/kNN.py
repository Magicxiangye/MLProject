"""
Created on 2022-5-31
kNN: k Nearest Neighbors

Input:      inx: 输入的特征向量（需要进行预测的特征向量）
            dataSet: 训练样本集
            labels: 标签向量
            k:初始K值的设置

Output:     the most popular class label
@author: magicye18
"""
import numpy as np
import operator
from os import listdir


# 预测分类方法
# k近邻的主方法


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 主要的功能是重复某个数组
    # (n,m) 行方向上复制N次，列方向上复制M次，进行距离的度量
    # 直接进行矩阵计算
    # 这里距离的度量是欧式距离L2，也可以用曼哈顿L1或者汉明距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 返回排序索引的nparray
    sortedDistIndicies = distances.argsort()
    # k近邻的选择
    classCount = {}
    for i in range(k):
        # 统计前K个的类别，给出最后的预测
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对于类别的数量进行逆序，输出发生最高频率的类别
    # classCount.items()将字典分解为元组列表，key按元组的第二个元素进行逆序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回最多的类别
    return sortedClassCount[0][0]


# 自己生成的一个简易的训练集和分类的标签向量（测试用）
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 文件的格式转化
# 其实直接的pd.read_csv()就很香，这样做大文件量下肯定浪费资源
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 文件的行数
    returnMat = np.zeros((numberOfLines, 3))  # 3是特征的数量，准备相应格式的np零矩阵
    classLabelVector = []  # 分类标签 Y值的向量
    # 读取文件
    fr = open(filename)
    index = 0  # 下标
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 分类标签页存在分类向量中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回的是特征矩阵，以及相应的标签向量
    return returnMat, classLabelVector


# 数据的归一化处理
# 需要处理的列进行处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0--列方向
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]  # 行
    # 处理为同样的矩阵形态，进行最大最小归一化的计算处理
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 便捷版的归一化处理
# 最大最小的标准化
def col2Norm(data):
    # data 是pd.Dataframe类型的数据
    return data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


# 函数验证方法
def datingClassTest():
    hoRatio = 0.90  # 10%随机选取的测试用例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 这里简单的数据集就无需数据的预处理阶段，直接进行归一化的处理
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # 每个数据的分类预测
    for i in range(numTestVecs):
        # 主近邻函数
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    # 最后的分类器错误率
    print(errorCount)


# 单条数据的输入预测
# 命令行输入各纬度版本
def classifyPerData():
    # 用于输出预测结果的labels array
    # 测试数据集中的labels是用户的喜好
    result_labels = ['not at all', 'in small doses', 'in large like']

    # input 各个特征
    percentTats = float(input('特征一'))
    ffmiles = float(input('input two'))
    icecream = float(input('input three'))

    # 使用存在的数据集进行K近邻的计算
    # classify0(inX, dataSet, labels, k) 直接可以返回出结果，后面懒得写了


# 图片转化为训练向量
# 训练集中的图片是已经转化为数据的 ， 宽高为32*32的黑白（0，1）图片
def img2vector(filename):
    # 32*32，转化为1024个特征的向量
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    # 行
    for i in range(32):
        # 逐行读取
        lineStr = fr.readline()
        # 列
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回处理好的图片向量
    return returnVect


# 使用算法来识别手写的数字
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 取训练集目录下的文件名的set
    m = len(trainingFileList)  # 训练文件的个数
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 去掉.txt后缀
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 生成预测用的数据矩阵
    testFileList = listdir('testDigits')  # 测试集

    errorCount = 0.0
    mTest = len(testFileList)
    # 计算预测的精度
    # 测试集中，每一个图片进行预测
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 去掉.txt后缀
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
