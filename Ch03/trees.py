"""
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
"""

"""
Add Jan 10, 2017
Convert from python 2 to python 3
Korean Comments
@author: Yedarm Seong <mybirth0407@gmail.com>
"""
from math import log
import operator

# 간단한 데이터 생성
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # 이산 값으로 변경
    return dataSet, labels

# 섀넌 엔트로피를 계산한다.
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    # 가능한 모든 분류 항목에 대한 사전 생성
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0

        labelCounts[currentLabel] += 1
    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # 밑이 2인 로그
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
             # 분할한 속성 잘라내기
            reducedFeatVec = featVec[:axis]
            # a = [1, 2, 3], b = [4, 5, 6]
            # a.extend(b) = [1, 2, 3, 4, 5, 6]
            # a.append(b) = [1, 2, 3, [4, 5, 6]]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 마지막 열은 분류 항목 표시에 사용된다.
    numFeatures = len(dataSet[0]) - 1  
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 모든 속성에 대해 계산한다.
    for i in range(numFeatures):
        # 이 속성에 대한 모든 예시 리스트를 생성한다.
        featList = [example[i] for example in dataSet]
        # 유일한 값들의 집합을 얻는다.
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 정보 이득을 계산한다.
        infoGain = baseEntropy - newEntropy
        # 최대 정보 이득과 현재 정보 이득을 비교하여
        # 현재 정보 이득이 최대 정보 이득보다 크다면, 최대로 지정한다.
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 최대 속성을 정수로 반환
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        # classCount 사전에 vote가 없다면 사전에 추가한다.
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    # itemgetter(1): 사전의 값을 기준으로 정렬한다.
    sortedClassCount = sorted(
        classCount.iteritems(), key=operator.itemgetter(1), reverse=True
    )
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # 예) dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no']]
    # classList = ['yes, 'yes', 'no']
    classList = [example[-1] for example in dataSet]

    # 모든 분류 항목이 같은 경우 분할을 중단한다.
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 데이터 집합에 더 이상 속성이 없는 경우 분할을 중단한다.
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 모든 분류 항목을 복사하여
        # 트리가 이미 존재하는 분류 항목을 망치지 않는다.
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels
        )
    return myTree


# 의사결정 트리를 위한 분류 함수
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 분류 항목 표시를 문자열로 반환한다.
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# pickle을 가지고 의사결정 트리를 유지시킬 수 있다.
# pickle을 이용하여 트리를 저장한다.
def storeTree(inputTree, filename):
    import pickle
    # 바이트가 아니라 파일에 dump 하지 못한다.
    # fw = open(filename, 'w')
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

# pickle로 저장한 트리를 읽어온다.
def grabTree(filename):
    import pickle
    # 바이트로 저장했기 때문에, 바이트로 읽어와야 한다.
    # fr = open(filename)
    fr = open(filename, 'rb')
    return pickle.load(fr)
