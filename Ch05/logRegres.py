'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''

'''
Add Jan 14, 2017
Convert from python 2 to python 3
Korean Comments
@author: Yedarm Seong <mybirth0407@gmail.com>
'''

from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')

    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 로지스틱 회귀에서 기울기 상승을 최적화한다.
def gradAscent(dataMatIn, classLabels):
    # NumPy 행렬로 변환
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    # 기울기 계산의 멈춤 조건을 반복 횟수로 주었다.
    maxCycles = 500
    weights = ones((n, 1))

    # heavy on matrix operations
    # 무거운 행렬 연산
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 로지스틱 회귀 최적선을 플롯한다.
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 확률적으로 기울기 상승을 최적화한다.
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 0버전보다 수렴이 더 빠르게 개선된 확률적 기울기 상승 최적화 알고리즘
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        # python 2
        # dataIndex = range(m)
        dataIndex = list(range(m))
        for i in range(m):
            # 연산을 반복함에 따라 alpha값을 감소시킨다.
            # 일정한 간격(0.0001)이 있기 때문에 0에는 도달하지 못한다.
            # alpha = 4 / (1.0 + j + i) + 0.01
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# 입력받은 벡터를 sigmoid 확률로 분류한다.
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []

        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []

        for i in range(21):
            lineArr.append(float(currLine[i]))

        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0

    for k in range(numTests):
        errorSum += colicTest()

    print("after %d iterations the average error rate is: %f"
        % (numTests, errorSum / float(numTests)))
