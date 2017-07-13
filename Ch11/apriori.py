"""
Created on Mar 24, 2011
Ch 11 code
@author: Peter
"""

"""
Add Mar 08, 2017
Convert from python 2 to python 3
Korean Comments
@author: Yedarm Seong <mybirth0407@gmail.com>
"""

from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset은 변하지 않는 자료구조이다.
    # python 2
    # return map(frozenset, C1)
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can] = 1
                else: ssCnt[can] += 1
    # numItems보단 numTransactions가 더 적절하다.
    # numItems = float(len(D))
    numTransactions = float(len(D))

    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# Ck를 생성한다.
# 후보 집합을 생성한다.
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk): 
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            # if first k - 2 elements are equal
            if L1 == L2:
                # set union
                retList.append(Lk[i] | Lk[j])
    return retList

# Apriori 알고리즘을 통해 빈발 집합과 지지도 사전을 반환
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    # python 2
    # D = map(set, dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        # scan DB to get Lk
        Lk, supK = scanD(D, Ck, minSupport)
        # update: set 혹은 dict에서 add와 같은 기능을 한다.
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# 신뢰도 값을 가지고 규칙들의 리스트를 생성한다.
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # only get the sets with two or more items
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 신뢰도 값을 계산한다.
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # create new list to return
    prunedH = []
    for conseq in H:
        # 신뢰도를 계산한다.
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf: 
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)

            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 결과로부터 룰을 생성한다.
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    # try further merging
    if len(freqSet) > (m + 1):
        # create Hm + 1 new candidates
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# 룰을 출력한다.
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()


# votesmart의 apikey를 활용해야 한다.
# 현재는 유료 시스템이라 이용하지 않았다.
# from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'get your api key first'
# def getActionIds():
#     actionIdList = []; billTitleList = []
#     fr = open('recent20bills.txt') 
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum) #api call
#             for action in billDetail.actions:
#                 if action.level == 'House' and \
#                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print('bill: %d has actionId: %d' % (billNum, actionId))
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print("problem getting bill %d" % billNum)
#         sleep(1)                                      #delay to be polite
#     return actionIdList, billTitleList
        
# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician) 
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print('getting votes for actionId: %d' % actionId)
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not transDict.has_key(vote.candidateName): 
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except: 
#             print("problem getting actionId: %d" % actionId)
#         voteCount += 2
#     return transDict, itemMeaning
