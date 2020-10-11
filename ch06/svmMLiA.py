import numpy as np

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append(float(lineArr[0]), float(lineArr[1]))
            labelMat.append(float(lineArr[2]))
    
    return dataMat, labelMat

def selectJrand(i, m):
    j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L

    return aj