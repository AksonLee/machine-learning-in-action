# 2020-10-13
# Akson

import numpy as np
import matplotlib.pyplot as plt

# 加载一个简单的数据集
def loadSimData():
    dataMatrix = np.matrix(
        [[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return dataMatrix, classLabels

# 加载文件中的数据集
def loadDataSet(filename):
    dataMatrix = []
    labelMatrix = []
    numFeat = 0
    with open(filename, 'r') as fr:
        numFeat = len(fr.readline().split('\t'))
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(curLine[i]))
            dataMatrix.append(lineArr)
            labelMatrix.append(float(curLine[-1]))
    
    return dataMatrix, labelMatrix


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray

# 构造单层决策树桩
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.matrix(dataArr)
    labelMatrix = np.matrix(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')

    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictVals == labelMatrix] = 0
                weightedError = D.T * errArr

                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weight error is %.3f' % (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    endErrorRate = 0.0

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print('D:', D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst:', classEst.T)
        expon = np.multiply(-1 * alpha * np.matrix(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        # print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.matrix(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        endErrorRate = errorRate
        # print('total error:', errorRate)
        if errorRate == 0.0:
            break
    
    return weakClassArr,aggClassEst, endErrorRate

# 在测试集上运行adaboost
def adaClassify(dataToClass, classLabels, classifierArray):
    dataMatrix = np.matrix(dataToClass)
    labelMatrix = np.matrix(classLabels)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(len(classifierArray)):
        classEst = stumpClassify(dataMatrix, classifierArray[i]['dim'], classifierArray[i]['thresh'], classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha'] * classEst
        # print(aggClassEst)

    errorRate = 0.0
    errArr = np.mat(np.ones((m, 1)))
    errorRate = float(errArr[np.sign(aggClassEst) != labelMatrix.T].sum()) / float(m)
    
    return errorRate

def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 1.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        cur = (cur[0] - delX, cur[1] - delY)

    print('The area under the curve is: ', ySum * xStep)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()

def testHorseColicClassify():
    dataArr, trainingLabelArr = loadDataSet('horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    i = 1
    while i < 1001:
        classifierArray, aggClassEst, trainingErrorRate = adaBoostTrainDS(dataArr, trainingLabelArr, i)
        testErrorRate = adaClassify(testArr, testLabelArr, classifierArray)
        print('iter number: %d, trainErrorRate: %.3f, testErrorRate: %.3f' % (i, trainingErrorRate, testErrorRate))
        i += 10

def testPlotRoc():
    dataArr, trainingLabelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst, trainingErrorRate = adaBoostTrainDS(dataArr, trainingLabelArr, 50)
    plotROC(aggClassEst.T, trainingLabelArr)


# test
if __name__ == '__main__':
    # dataMatrix, classLabels = loadSimData()
    # classifierArray = adaBoostTrainDS(dataMatrix, classLabels, 30)
    # print(adaClassify([[5, 5], [0, 0]], classifierArray))
    testPlotRoc()

