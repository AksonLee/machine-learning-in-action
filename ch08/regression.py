import numpy as np
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

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

# 岭回归
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam

    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, connot do inverse')
        return
    
    ws = denom.I * xMat.T * yMat
    return ws

# 岭回归测试
def ridgeTest(xArr, yArr):
    xMat = np.matrix(xArr)
    yMat = np.matrix(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    
    return wMat


# 岭回归测试驱动
def ridgeTestMy():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

# 按列规则化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)   #calc mean then subtract it off
    inVar = np.var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat

# 前项逐步线性回归
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)

    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))

    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        print(ws.T)
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    
    return returnMat

# 前项逐步线性回归测试驱动
def stageWiseTestMy():
    xArr, yArr = loadDataSet('abalone.txt')
    stageWise(xArr, yArr, 0.0001, 50000)

# 书上的方法不能用了，在最新源码发现了这个，就复制了过来
def scrapePage(inFile, outFile, yr, numPce, origPrc):
    with open(inFile, 'r') as fr:
        soup = BeautifulSoup(fr.read(), 'html.parser')
        i=1
        currentRow = soup.findAll('table', r = "%d" % i)

        while(len(currentRow) != 0):
            title = currentRow[0].findAll('a')[1].text
            lwrTitle = title.lower()
            if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
                newFlag = 1.0
            else:
                newFlag = 0.0
            soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
            if len(soldUnicde)==0:
                print('item #%d did not sell' % i)
            else:
                soldPrice = currentRow[0].findAll('td')[4]
                priceStr = soldPrice.text
                priceStr = priceStr.replace('$','') #strips out $
                priceStr = priceStr.replace(',','') #strips out ,
                if len(soldPrice)>1:
                    priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
                print('%s\t%d\t%s' % (priceStr,newFlag,title))
                with open(outFile, 'a') as fw:
                    fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
            i += 1
            currentRow = soup.findAll('table', r="%d" % i)

def setDataCollect():
    scrapePage('setHtml/lego8288.html', 'legoData.txt', 2006, 800, 49.99)
    scrapePage('setHtml/lego10030.html', 'legoData.txt', 2002, 3096, 269.99)
    scrapePage('setHtml/lego10179.html', 'legoData.txt', 2007, 5195, 499.99)
    scrapePage('setHtml/lego10181.html', 'legoData.txt', 2007, 3428, 199.99)
    scrapePage('setHtml/lego10189.html', 'legoData.txt', 2008, 5922, 299.99)
    scrapePage('setHtml/lego10196.html', 'legoData.txt', 2009, 3263, 249.99)

# 乐高模型回归方程测试
def legoStandRegres():
    lgX, lgY = loadDataSet('legoData.txt')
    m, n = np.shape(lgX)

    lgX1 = np.mat(np.ones((m, n + 1)))
    lgX1[:, 1 : 5] = np.mat(lgX)

    ws = standRegres(lgX1, lgY)

# k折交叉验证
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))

    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []

        np.random.shuffle(indexList)
        
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])

        wMat = ridgeTest(trainX, trainY)

        for k in range(30):
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)

            # 用训练数据将测试数据标准化
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain

            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
    
    meanErrors = np.mean(errorMat, 0)
    minMean = float(np.min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]

    # 数据还原
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX

    print('the best model from Ridge Regression is:', unReg)
    print('with constant term: ', -1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat))


# k折交叉验证测试驱动
def crossValidationTestMy():
    lgX, lgY = loadDataSet('legoData.txt')
    crossValidation(lgX, lgY, 10)

# test
if __name__ == '__main__':
    # standRegresTest()
    # lwlrMyTest()
    # abaloneAgeTest()
    # ridgeTestMy()
    # stageWiseTestMy()
    # setDataCollect()
    # legoStandRegres()
    crossValidationTestMy()
