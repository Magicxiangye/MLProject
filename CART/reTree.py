"""
Created on 7 13, 2022
Tree-Based Regression Methods
@author: magicye18
"""
import numpy as np


# 数据的预处理
# 数据类型的转化
# 直接pandas大法好，直接apply就行，或者直接astype()也很快
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = []
        for i in curLine:
            fltLine.append(float(i))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    # 分别返回符合切分条件的左右子树的子集
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 回归树叶子节点的值
# 就是目标变量的一个均值
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 切分误差估计函数
# 使用的是总方差
# 方差*数据集个数
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


# 线性模型
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    # 创造自变量和目标变量
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    # x的第零列为偏差
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    # 线性模型的矩阵
    xTx = X.T * X
    # 确保矩阵的逆是存在的
    # 计算输入矩阵的行列式
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    # 回归系数（回归模型的系数）
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


# 平方误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


# 寻找最好的切分方式
# 返回切分特征以及相应的特征值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最小样本数
    # 回归数当前集合所有的回归值都相同时
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        # 返回叶子节点
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # 寻找最好的切分特征以及最好的切分值
    S = errType(dataSet)  # 初始的总平方差值
    # 初始的最好切分总方差
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    # 遍历进行寻找
    # 特征编号
    for featIndex in range(n - 1):
        # 相应的特征值
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断切分的子集是否符合最小的样本数量
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 切分后的总方差大小
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 判断总方差下降是否小于阈值
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 是否符合最小的切分样本量
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回特征index以及切分的特征值
    return bestIndex, bestValue


# CART树构建函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 选取最佳的切分特征以及特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 切分的停止条件
    if feat is None:
        return val
    # 树节点的构建
    retTree = {'spInd': feat, 'spVal': val}
    # 数据矩阵的左右子树的切分
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 左右子树的递归进入创建节点函数
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 是否为分支节点
def isTree(obj):
    return type(obj).__name__ == 'dict'


# 从上向下的遍历树，直到找到两个叶子结点，计算两节点的平均值
# 树的塌陷处理
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 后剪枝函数
# 递归处理CART树
def prune(tree, testData):
    # 没有测试数据的话，对数进行塌陷处理
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 使用CART数对于测试集进行预测
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    # 对分支节点递归的调用后剪枝的函数
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 当左右子数都为叶子节点时
    # 对合并前后的预测误差进行比较
    # 判断时候进行剪枝，合并叶子节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 进行切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 总平方误差
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        # 塌陷处理的树平均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 总平方误差
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        # 判断是否剪枝
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            # 不合并
            return tree
    else:
        return tree


# 回归树预测
def regTreeEval(model, inDat):
    return float(model)


# 模型树预测
def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    # 在原数据矩阵上加上第0列
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


# 树预测函数
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    # 进入树结构
    # 左大右小
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    # 进入右子树
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 预测执行函数
# 默认叶子节点为回归树的叶子节点预测
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
