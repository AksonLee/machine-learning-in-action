import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(filename, delim = '\t'):
    with open(filename, 'r') as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        dataArr = [list(map(float, line)) for line in stringArr]
        # print(dataArr)
        return np.mat(dataArr)

def pca(dataMat, topNfeat = 9999999):
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    # print('meanRemoves: ', meanRemoved)

    covMat = np.cov(meanRemoved, rowvar = 0)
    # print('covMat: ', covMat)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # print('eigVals: ', eigVals)
    # print('eigVects: ', eigVects)
    eigValInd = np.argsort(eigVals)
    # print('eigValInd1: ', eigValInd)
    eigValInd = eigValInd[: -(topNfeat + 1) : -1]
    # print('eigValInd2: ', eigValInd)
    redEigVects = eigVects[:, eigValInd]
    # print('redEigVects: ', redEigVects)
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def pcaTestMy():
    dataMat = loadDataSet('testSet.txt')
    lowDataMat, reconMat = pca(dataMat, 1)
    print(np.shape(lowDataMat))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker = '^', s = 90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker = 'o', s = 50, c = 'red')

    plt.show()

def replaceNaNToMean():
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])

        dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal

    return dataMat

def replaceNaNToMeanTestMy():
    dataMat = replaceNaNToMean()
    meanVals = np.mean(dataMat, axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar = 0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))

    print(eigVals)

# test
if __name__ == '__main__':
    # pcaTestMy()
    replaceNaNToMeanTestMy()
