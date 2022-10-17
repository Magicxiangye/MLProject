"""
Created on 6 23, 2022
Decision Tree Source Code
@author: magicye18
"""
import math
import operator
import pickle


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算数据子集合中每一个类别的信息熵
def calcShannonEnt(dataSet):
    # 子集合中数据的个数
    numEntries = len(dataSet)
    # 类别个数的统计
    labelCounts = {}
    # 每一条数据进行分类的统计
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    # 信息熵的计算
    shannonEnt = 0.0
    for key in labelCounts:
        prob = round(float(labelCounts[key]) / numEntries, 5)
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# 按照给定的特征的特征值划分数据集
# axis 给定的特征
# value 相应的特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将划分的特征切割出去
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择当前最好的划分特征
def chooseBestFeatureToSplit(dataSet):
    # 除了labels列的特征数量
    numFeatures = len(dataSet[0]) - 1
    # 初始信息熵的值
    baseEntropy = calcShannonEnt(dataSet)
    # 初始最好的信息增益
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 当前特征中的所有特征值
        # 直接pd.dataframe的unique() 更快
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 累计这个属性分支后的条件熵之和
        newEntropy = 0.0
        for value in uniqueVals:
            # 进行集合的切割
            # 多分支的树，不是二分法
            # 切割每个特征中的每一个属性值
            subDataSet = splitDataSet(dataSet, i, value)
            # 该属性值的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 该属性值的条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 替换为当前最好的属性
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 都计算完成后返回最好的属性的Index
    return bestFeature


# 叶子节点中也没有存在唯一的类别时，使用出现频率最大的类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # 当前数据集的类别
    classList = [example[-1] for example in dataSet]
    # 递归回归树的第一种结束的方式
    # 当集合中只有一种类别的时候，直接返回当前的类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归回归树的第二种结束方式
    # 使用完了所有的特征
    # 但叶子节点中还是有不同的类
    # 使用出现概率最大的那个类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 递归的主程序
    # 选取当前信息增益最大的属性
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 当前属性所有的属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 进行分割
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    # 返回递归嵌套的字典
    return myTree


# 分类预测
# input
# 1.训练完成的dict类型的决策树
# 2.特征标签列表和预测向量index一致就行
# 3.预测向量
def classify(inputTree, featLabels, testVec):
    # 强制转成 list 类型即可使用
    # 训练好的决策数的根节点
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 拿到测试向量当前特征的值或分类
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    # 获取决策树当前节点的结果
    valueOfFeat = secondDict[key]
    # 判断非叶子节点，递归的调用分类决策函数
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        # 是叶子节点的话，输出该分类
        classLabel = valueOfFeat
    return classLabel


# 序列化的存储训练好的决策树信息
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


# 拿到序列化对象的数据
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)
