import numpy as np
import operator

import matplotlib
import matplotlib.pyplot as plt

import os

# 准备数据集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 分类方法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 获取数据集行数，即样本数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet # 按列复制样本行数列个测试样本，并与每个样本做差
    sqDiffMat = diffMat ** 2  # 开平方
    sqDistances = sqDiffMat.sum(axis = 1) # 按行相加
    distances = sqDistances ** 0.5 # 开根号
    sortedDistIndicies = np.argsort(distances) # 获得对应数据集从小到大的索引号

    # 统计类别数
    classCount = {}

    #依据索引号来获取前k个标签，并统计数量
    for i in range(k): 
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 对获取到的类别及其数量按照数量从大到小排列
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) 
    
    return sortedClassCount[0][0]

# 读取文本文件
def file2Matrix(filename):
    arrayLines = []
    with open(filename, 'r') as f:
        arrayOfLines = f.readlines()
    
    numberOfLines = len(arrayOfLines)
    
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0 : 3]
        classLabelVector.append(int(listFromLine[-1]))
        index = index + 1
     
    return returnMat, classLabelVector

# 将数据集归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals, (m, 1)))/np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

# 约会数据分类方法
def datingClassTest():
    hoRatio = 0.30
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        # print('The classifier return %d, the real is %d.' % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount = errorCount + 1

    print('The total error rate is: %f' % (errorCount / float(numTestVecs)))

# 约会数据分类测试
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    timeInGame = float(input('How much time in playing games?'))
    flyMiles = float(input('How much miles fly per year?'))
    iceCream= float(input('How many ice-cream eat per year!'))

    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArray = np.array([flyMiles, timeInGame, iceCream])
    classifierResult = classify0((inArray - minVals) / ranges, normMat, datingLabels, 3)
    print(resultList[classifierResult - 1])

# 图像转矩阵
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(line[j])
    return returnVect

#手写数据测试
if __name__ == '__main__':
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)

    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)

        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('Classifier come back %s and the True is %s.' % (classifierResult, classNum))
        if classifierResult != classNum:
            errorCount = errorCount + 1.0

    print('The total number of errors is: %d' % errorCount)
    print('The total error rate is %f' % (errorCount/float(mTest)))