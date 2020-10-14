import numpy as np

def loadDataSet(filename):
    dataMat =[]
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            dataMat.append(map(float, curLine))

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
        for splitVal in set(dataSet[:, featIndex]):
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



# test
if __name__ == '__main__':
    # binSplitDataSetTestMy()
    pass
