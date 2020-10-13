import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numFeat = 0
    dataList = []
    labelList = []

    with open(filename, 'r') as fr:
        numFeat = len(fr.readline().split('\t')) - 1
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataList.append(lineArr)
            labelList.append(float(curLine[-1]))
    
    return dataList, labelList


def standRegres(xArr, yArr):
    xMat = np.matrix(xArr)
    yMat = np.matrix(yArr).T
    xTx = xMat.T * xMat

    if np.linalg.det(xTx) == 0.0:
        print('This matrix can not do inverse')
        return
    
    ws = xTx.I * xMat.T * yMat
    return ws

def standRegresTest():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    
    xMat = np.matrix(xArr)
    yMat = np.matrix(yArr)
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat))

    # 排序
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    ax.plot(xCopy[:, 1], yHat)
    # plt.show()

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.matrix(xArr)
    yMat = np.matrix(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))

    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    
    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix can not do inverse')
        return
    ws = xTx.I * xMat.T * weights * yMat

    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(xArr[i], xArr, yArr, k)
    
    return yHat


def lwlrMyTest():
    xArr, yArr = loadDataSet('ex0.txt')
    xMat = np.matrix(xArr)
    yMat = np.matrix(yArr)

    yHat = lwlrTest(xArr, xArr, yArr, 0.01)

    # 排序
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s = 2, c = 'red')

    ax.plot(xSort[:, 1], yHat[srtInd])
    # plt.show()

    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def abaloneAgeTest():
    abX, abY = loadDataSet('abalone.txt')
    trainingStart = 0
    trainingEnd = 99

    testStart = 100
    testEnd = 199

    for k in [0.1, 1.0, 10.0]:
        yHat = lwlrTest(abX[testStart: testEnd], abX[trainingStart: trainingEnd], abY[trainingStart: trainingEnd], k)
        rssErr = rssError(abY[testStart: testEnd], yHat.T)
        print('Start: %d, End: %d, k: %.2f, rssError: %.2f' % (trainingStart, trainingEnd, k, rssErr))


# test
if __name__ == '__main__':
    # standRegresTest()
    # lwlrMyTest()
    abaloneAgeTest()
