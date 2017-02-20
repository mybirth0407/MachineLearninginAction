"""
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      
    inX: vector to compare to existing dataset (1xN)
    dataSet: size m data set of known vectors (NxM)
    labels: data set labels (1xM vector)
    k: number of neighbors to use for comparison (should be an odd number)
            
Output:
    the most popular class label

@author: pbharrin
"""

"""
Add Jan 05, 2017
Convert from python 2 to python 3
Korean Comments
@author: Yedarm Seong <mybirth0407@gmail.com>
"""

from numpy import *
import operator
from os import listdir
from commons import *


"""
inX: 분류할 데이터
dataSet: 기존 데이터
labels: 기존 데이터의 라벨
k: 선택할 최근접 이웃의 수
"""
def classify0(inX, dataSet, labels, k):
    # dataSet의 행의 크기
    dataSetSize = dataSet.shape[0]
    # inX를 행렬로 변환한다.
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # numpy.argsort(): 정렬된 후의 인덱스를 반환한다.
    # 예) a = [1.48, 1.41, 0, 0.1], b = a.argsort()
    # b = [2, 3, 1, 0]
    # for x in b: a[x] = [0, 0.1, 1.41, 1.48]
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # python 2
    # sortedClassCount = sorted(
    #     classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    # itemgetter(1): 사전의 값을 기준으로 정렬한다.
    # itemgetter(0): 사전의 키를 기준으로 정렬한다.
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    return sortedClassCount[0][0]


# 훈련 집합과 해당 훈련 집합의 분류 항목(labels)을 생성하여 반환한다.
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 파일을 읽어 행렬과 분류 항목을 반환한다.
def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    # 파일 전체를 읽는다.
    arrayOLines = fr.readlines()
    # 파일의 줄 수를 가져온다.
    numberOfLines = len(arrayOLines)
    # 반환할 행렬을 생성하고 0으로 초기화한다.
    returnMat = zeros((numberOfLines, 3))
    # 반환할 분류 항목을 생성한다.
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        # 해당 라인의 맨 앞과 맨 뒤 문자가 공백이라면 제거한다.
        line = line.strip()
        # 탭을 기준으로 문자열을 분리한다.
        listFromLine = line.split('\t')
        # index행, 모든 열에 대입한다.
        # l[:, i]: 모든 행, i열을 뜻한다.
        returnMat[index, :] = listFromLine[0:3]

        # 리스트의 음수 인덱스는 마지막 원소부터 시작한다.
        # -1 의 경우 마지막 원소를 가리킨다.
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))

        index += 1
    return returnMat, classLabelVector


# 데이터셋을 정규화한다.
def autoNorm(dataSet):
    # 정규화: (x - min) / (max - min)
    # min, max(0) 각 열 마다 가장 작은(큰) 원소를 반환한다.
    # min, max(1) 각 행 마다 가장 작은(큰) 원소를 반환한다.
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # tile(v, (m, n)): m행 n열 행렬의 모든 원소를 v로 초기화하여 생성한다. 
    # x - min
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 구성 요소 현명하게 나누기
    # x - min / max - min
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 데이트 사이트의 분류기를 검사한다.
def datingClassTest():
    # 10%의 데이터셋으로 검사한다.
    hoRatio = 0.10
    # 파일로부터 데이터셋을 읽어 로드한다.
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 검사할 데이터 수
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        # k-NN 알고리즘을 수행하여 분류한다.
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :],
            datingLabels[numTestVecs:m], 3
        )

        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, datingLabels[i]))

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


# 사람의 정보를 입력하면 분류 항목을 보여준다.
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    # python 2
    # percentTats = float(raw_input(
    # ffMiles = float(raw_input("frequent flier miles earned per year?"))
    # iceCream = float(raw_input("liters of ice cream consumed per year?"))

    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, 3
    )

    print("You will probably like this person: %s" %
          resultList[classifierResult - 1])


# 이미지를 하나의 벡터로 변환한다.
# 여러 줄의 데이터를 한 줄의 벡터로 나열한다.
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 필기체의 분류기를 검사한다.
def handwritingClassTest():
    hwLabels = []
    # digits 압축을 해제한다.
    unzip('digits.zip')
    # 훈련 집합 데이터를 로드한다.
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, classNumStr))

        if (classifierResult != classNumStr):
            errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

    # 디렉토리를 제거한다.
    removeDirectory('trainingDigits')
    removeDirectory('testDigits')
