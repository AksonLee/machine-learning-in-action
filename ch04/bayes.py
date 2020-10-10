import numpy as np
import re
import operator
import feedparser

# 生成测试用数据集
def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    classVec = [0, 1, 0, 1, 0, 1]  # 0代表没有侮辱性文字，1代表有

    return postingList, classVec

# 返回某篇文档含有的所有词（不重复）
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 根据词汇表看某句话中词是否出现
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('The word %s is not in my Vocabulary!' % word)
    
    return returnVec

# 根据词汇表看某句话中词出现的次数
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('The word %s is not in my Vocabulary!' % word)
    
    return returnVec
   
# 依据训练集与对应标签训练分类器
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive

# 依据训练参数做分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)

    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    
    p0, p1, pAb = trainNB0(trainMat, listClasses)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWord2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0, p1, pAb))

    testEntry = ['stupid', 'garbage', 'dalmation']
    thisDoc = setOfWord2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0, p1, pAb))

def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 0]

# 垃圾邮件训练与分类测试
def spemTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        with open('email/spam/%d.txt' % i, 'r') as fr:
            allWords = fr.read()
            wordList = textParse(allWords)
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
        
        with open('email/ham/%d.txt' % i, 'r') as fr:
            allWords = fr.read()
            wordList = textParse(allWords)
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    # 建立训练用的文档矩阵与标签
    trainMat = []
    trainClasses =[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    # 训练
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)

    # 测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('The error rate is: ', float(errorCount) / len(testSet))
 

def calcMostFreq(vocabList, fullText, n = 30):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[ : n]

def localWords(feed0, feed1):
    docList = []
    classList = []
    fullText = []
    
    minLen = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(minLen):
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
    vocabList = createVocabList(docList)
    topNWords = calcMostFreq(vocabList, fullText, n = 110)
    for pairW in topNWords:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    # 训练
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)

    # 测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('The error rate is: ', float(errorCount) / len(testSet))


# 测试
if __name__ == '__main__':
    mcR = feedparser.parse('minecraftRss.txt')
    tR = feedparser.parse('terrariaRss.txt')
    localWords(mcR, tR)