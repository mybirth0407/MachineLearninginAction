'''
Created on Oct 14, 2010

@author: Peter Harrington
'''

'''
Add Jan 10, 2017
Convert from python 2 to python 3
Korean Comments
@author: Yedarm Seong <mybirth0407@gmail.com>
'''
import matplotlib.pyplot as plt

# 플롯할 상자와 화살표 형태를 정의한다.
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 트리에 있는 노드의 개수를 확인한다.
def getNumLeafs(myTree):
    numLeafs = 0
    # python 2
    # firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        # 해당 노드가 사전 형태인지 검사한다.
        # 사전이 아니라면 단말 노드(leaf nodes)이다.
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 트리의 깊이를 확인한다.
def getTreeDepth(myTree):
    maxDepth = 0
    # python 2
    # firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        # 해당 노드가 사전 형태인지 검사한다.
        # 사전이 아니라면 단말 노드(leaf nodes)이다.
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1

        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 텍스트 주석을 가진 트리 노드를 플롯한다.
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt,
        textcoords='axes fraction', va="center", ha="center",
        bbox=nodeType, arrowprops=arrow_args
    )


# 자식 노드와 부모 노드 사이에 텍스트를 플롯한다.
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


# 첫번째 key가 어떤 속성으로 분할되었는지 말해준다.
def plotTree(myTree, parentPt, nodeTxt):
    # 트리의 너비 구하기
    numLeafs = getNumLeafs(myTree)
    # 트리의 높이 구하기
    depth = getTreeDepth(myTree)

    # python 2
    # firstStr = myTree.keys()[0]
    # 텍스트 라벨이 반드시 이 노드여야 한다.
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) 
             / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 해당 노드가 사전 형태인지 검사한다.
        # 사전이 아니라면 단말 노드(leaf nodes)이다.
        if type(secondDict[key]).__name__ == 'dict':
            # 재귀
            plotTree(secondDict[key], cntrPt, str(key))
        # 단말 노드를 출력한다.
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(
                secondDict[key], (plotTree.xOff, plotTree.yOff),
                cntrPt, leafNode
            )
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
# if you do get a dictonary you know it's a tree, and the first element
# will be another dict


# 리스팅 3.7, 69페이지
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # no ticks
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # ticks for demo puropses
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = - 0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()
# 리스팅 3.5, 65페이지


# 미리 정의한 트리를 도출한다.
def retrieveTree(i):
    listOfTrees = [
    {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
    {'no surfacing': {
        0: 'no',
        1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]

# createPlot(thisTree)
