import numpy as np
import tkinter

def loadDataSet(filename):
    dataMat =[]
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            curLine = list(map(float, line.strip().split('\t')))
            dataMat.append(curLine)

    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat1 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat2 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat1, mat2

# 二分树测试驱动
def binSplitDataSetTestMy():
    testMat = np.mat(np.eye(4))
    print(testMat)
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print(mat0)
    print(mat1)


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal

    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree

# 下面是对生成分类回归树的三次测试
def createTreeTestMy1():
    myData = loadDataSet('ex00.txt')
    myMatrix = np.mat(myData)
    myTree = createTree(myMatrix)
    print(myTree)

def createTreeTestMy2():
    myData = loadDataSet('ex0.txt')
    myMatrix = np.mat(myData)
    myTree = createTree(myMatrix)
    print(myTree)

def createTreeTestMy3():
    myData = loadDataSet('ex2.txt')
    myMatrix = np.mat(myData)
    myTree = createTree(myMatrix)
    print(myTree)

def isTree(obj):
    if type(obj).__name__ == 'dict':
        return True
    else:
        return False

def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    
    return (tree['left'] + tree['right']) / 2


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging') 
            return treeMean
        else:
            return tree
    else:
        return tree

def pruneTestMy():
    myData = loadDataSet('ex2.txt')
    myMatrix = np.mat(myData)
    myTree = createTree(myMatrix, ops = (0, 1))

    myDataTest = loadDataSet('ex2Test.txt')
    myMat2Test = np.mat(myDataTest)
    print(prune(myTree, myMat2Test))


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1 : n] = dataSet[:, 0 : n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular, cannot do inverse, try increase the second value of ops')
    ws = xTx.I * X.T * Y
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))

def modelTreeTestMy():
    myMat = np.mat(loadDataSet('exp2.txt'))
    print(createTree(myMat, modelLeaf, modelErr, (1, 10)))


def regTreeEval(model, inData):
    return float(model)

def modelTreeEval(model, inData):
    n = np.shape(inData)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1 : n + 1] = inData
    return float(X * model)

def treeForceCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForceCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForceCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForceCast(tree, np.mat(testData[i]), modelEval)
    return yHat

def createForeCastTestMy():
    trainMatrix = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMatrix = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    # 回归树
    myTree = createTree(trainMatrix, ops = [1, 20])
    yHat = createForeCast(myTree, testMatrix[:, 0])
    print(np.corrcoef(yHat, testMatrix[:, 1], rowvar = 0)[0, 1])

    # 模型树
    myTree = createTree(trainMatrix, modelLeaf, modelErr, ops = [1, 20])
    yHat = createForeCast(myTree, testMatrix[:, 0], modelTreeEval)
    print(np.corrcoef(yHat, testMatrix[:, 1], rowvar = 0)[0, 1])

    # 线性回归
    ws, X, Y = linearSolve(trainMatrix)
    for i in range(np.shape(testMatrix)[0]):
        yHat[i] = testMatrix[i, 0] * ws[1, 0] + ws[0, 0]
    print(np.corrcoef(yHat, testMatrix[:, 1], rowvar = 0)[0, 1])
          

# test
if __name__ == '__main__':
    # binSplitDataSetTestMy()
    # createTreeTestMy1()
    # createTreeTestMy2()
    # createTreeTestMy3()
    # pruneTestMy()
    # modelTreeTestMy()
    # createForeCastTestMy()
    pass
