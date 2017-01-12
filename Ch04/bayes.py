'''
Created on Oct 19, 2010

@author: Peter
'''

'''
Add Jan 10, 2017
Convert from python 2 to python 3
Korean Comments
@author: Yedarm Seong <mybirth0407@gmail.com>
'''
from numpy import *
from commons import *


# 데이터 집합을 불러온다.
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1: 폭력적인, 0: 폭력적이지 않은
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 모든 문서에 있는 모든 유일한 단어 목록을 생성한다.
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 두 집합의 합집합을 계산한다.
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 주어진 문서 내 어휘 목록에 있는 단어의 존재 여부를 0-1 벡터로 반환한다.
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 나이브 베이즈 분류기를 훈련시킨다.
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # ones(10) = array([1., 1., ..., 1.])
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    # 2.0으로 변경한다.
    # p0Denom = 0.0
    # p1Denom = 0.0

    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # log로 변경한다.
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom

    # 언더플로우를 방지하기 위해 log를 사용한다.
    # 실제 값과 그래프상 거의 유사하기 때문에 값의 오차는 고려하지 않아도 된다.
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    # 각 분류 항목에 대한 조건부 확률을 반환한다.
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # element-wise mult
    # 벡터 연산을 반복문 없이 사용할 경우 퍼포먼스가 증가한다.
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 긴 문자열(파일 내용)을 입력받아 단어 리스트를 반환한다.
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 임의로 선택된 이메일로부터 오류율을 보여준다.
def spamTest():
    docList = []
    classList = []
    fullText = []

    unzip('email.zip')
    # email.zip 내용 중 인코딩 관련 문제가 있어 zip 파일을 수정하였다.
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    removeDirectory('email')
    # 문서 리스트로부터 단어 리스트를 생성한다.
    vocabList = createVocabList(docList)
    # python 2
    # python 3에서는 range 의 deletion이 되지 않는다.
    # trainingSet = range(50)
    trainingSet = list(range(50))
    testSet = []

    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    # trainNB0 분류기를 훈련시킨다.
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    # 남은 아이템을 분류한다.
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])

    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList,fullText


# 자주 등장하는 30개의 단어를 반환한다.
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),
                        key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


# feed1과 feed0에서 지역 특색에 따른 단어 사용을 도출한다.
def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 문서 리스트로부터 단어 리스트를 생성한다.
    vocabList = createVocabList(docList)

    # 중복과 구조적 접속사 제거를 위해
    # 가장 많이 등장하는 30개의 단어를 제거한다.
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    # python 2
    # trainingSet = range(2 * minLen)
    trainingSet = list(range(2 * minLen))
    testSet = []

    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    # trainNB0 분류기를 훈련시킨다.
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    # 남은 아이템을 분류한다.
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


# 등장한 대부분의 단어들을 표시한다.
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")

    for item in sortedSF:
        print(item[0])

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")

    for item in sortedNY:
        print(item[0])
