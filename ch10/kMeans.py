import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat =[]
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            curLine = list(map(float, line.strip().split('\t')))
            dataMat.append(curLine)

    return dataMat

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def randCentTestMy():
    dataMat = np.mat(loadDataSet('testSet.txt'))
    print(randCent(dataMat, 2))
    print(distEclud(dataMat[0], dataMat[1]))

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True

    iterCount = 0

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        
        # print(centroids)
        for cent in range(k):
            ptsInclust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInclust, axis = 0)
        
        iterCount += 1
    
    # print(iterCount)
    return centroids, clusterAssment

def kMeansTestMy():
    dataMat = np.mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(dataMat, 4)
    # print(myCentroids)
    # print(clustAssing)

def biKmeans(dataSet, k, distMeas = distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2

    while len(centList) < k:
        lowestSSE = float('inf')
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])

            print('sseSplit = %.6f, sseNotSplit = %.6f' % (sseSplit, sseNotSplit))

            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
            
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClusterAss is: ', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    # 整理一下坑爹的myCentroids数据格式
    myCentroidsList = []
    for myCentroid in centList:
        myCentroidsList.append(myCentroid.flatten().A[0])
    centList = np.mat(myCentroidsList)

    return centList, clusterAssment

def biKmeansTestMy():
    myData = np.mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(myData, 3)
    print(centList)


def distSLC(vecA, vecB):
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * np.cos(np.pi *(vecB[0, 0] - vecA[0, 0]) / 180)

    return np.arccos(a + b) * 6371.0


def clusterClubs(numClust = 5):
    dataList = []
    with open('places.txt', 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataList.append([float(lineArr[4]), float(lineArr[3])])
    dataMatrix = np.mat(dataList)
    myCentroids, clustAssing = biKmeans(dataMatrix, numClust, distMeas = distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks = [], yticks = [])
    ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label = 'ax1', frameon = False)
    for i in range(numClust):
        ptsInCurrCluster = dataMatrix[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker = markerStyle, s = 90)
    
    ax1.scatter(myCentroids[:][:, 0].flatten().A[0], myCentroids[:][:, 1].flatten().A[0], marker = '+', s = 300)
    plt.show()
    

def clusterClubsTestMy():
    clusterClubs(12)


# test
if __name__ == '__main__':
    # randCentTestMy()
    # kMeansTestMy()
    # biKmeansTestMy()
    clusterClubsTestMy()
    pass