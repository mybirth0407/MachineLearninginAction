##__Introduction to Machine Learning in Action__

이 책은 4부에 걸쳐 다양한 기계 학습 알고리즘들을 소개한다.  
1부에서는 '분류', 2부에서는 '회귀로 수치형 값 예측하기', 3부에서는 '비지도 학습', 4부에서는 '부가적인 도구들'에 대하여 다룬다.  

1부에서는 '기계 학습 기초', 'k-최근접 이웃 알고리즘', '의사결정 트리', '나이브 베이즈', '로지스틱 회귀', '서포트 벡터 머신', '에이다부스트 메타 알고리즘으로 분류 개선하기',  
2부에서는 '회귀: 수치형 값 예측하기', '트리 기반 회귀',  
3부에서는 'k-평균 클러스터링', '어프라이어리 알고리즘으로 연관 분석하기', 'FP-성장 알고리즘으로 빈발 아이템 집합 찾기',  
4부에서는 '데이터 간소화를 위한 PCA 사용하기', 'SVD로 데이터 간소화하기', '빅 데이터와 맵 리듀스'  
를 다루고 있다.

1, 2부는 '지도 학습'을, 3부에서는 '비지도 학습'을 다루게 된다.

이 책에서는 Top 10 Algorithms in Data Mining 논문에서 다룬 10가지 알고리즘 '<C4.5(trees)', 'k-평균(k-means)', '서포트 벡터 머신(suppor vector machines)', '어프라이어리(Apriori)', '기댓값 최대화(Expectation Maximization)', '페이지랭크(PageRank)', '에이다부스트(AdaBoost)', 'k-최근접(k-Nearest Neighbors)', '나이브 베이즈(Naive Bayes)', '카트(CART)'> 중 기댓값 최대화, 페이지랭크 알고리즘을 제외한 8가지 알고리즘을 소개한다.


##__1부 분류__

1, 2부는 지도 학습 방법에 대해 다룬다.

지도 학습에서는 데이터를 학습할 때 '목적 변수'를 명시해야 한다.  
목적 변수는 기계 학습 알고리즘을 가지고 예측을 하고자 하는 것이다.  
이 책에서는 목적 변수가 다음 두 가지의 경우일 때에서만 다룬다.  
1. 명목형 값: 파충류, 어류, 포유류, 양서류, 식물, 균류 등  
2. 수치형 값: 0.100, 42.001, 1000.743 등


###__1장 기계 학습 기초__


####1.1 기계 학습이란 무엇인가?

기계 학습은 데이터를 정보로 변환하는 것이다.  
'자판기는 돈이 들어오거나 버튼을 누르는 것에 상관없이 항상 더 좋은 작업을 할 수 있어야 한다.' 와 같이 해결해야 할 문제가 무엇인지 알 수 없는 경우 혹은 문제를 제대로 이해하지 못한 경우에는 문제를 해결하기 위해 통계학이 필요하다.


####1.2 주요 전문용어

표 1.1

|| 무게 | 날개 길이 | 물갈퀴 | 등의 색상 | 종 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1001.1 | 125.0 | X | 갈색 | Buteo jamaicensis |
| 2 | 3000.7 | 200.0 | X | 회색 | Sagittarius serpentarius |
| 3 | 3300.0 | 220.3 | X | 회색 | Sagittarius serpentarius |
| 4 | 4100.0 | 136.0 | O | 검정색 | Gavia immer |
| 5 | 3.0 | 11.0 | X | 녹색 | Calothorax lucifer |
| 6 | 570.0 | 75.0 | X | 검정색 | Campephilus principalis |

무게, 날개 길이, 물갈퀴, 등의 색상은 같은 것들을 '속성(features)' 이라고 한다.  
무게와 날개 길이는 수치형 값을 갖고 물갈퀴와 등의 색상은 명목형 값을 갖는다.  
물갈퀴는 참/거짓 이진값을 갖고, 등의 색상은 색상의 개수 만큼의 정수값을 갖는다.  

분류 항목은 새의 종류이지만 더 명확하게 아이보리색 부리를 가진 딱따구리 등으로 분류 항목을 줄일 수도 있다.  

기계 학습 알고리즘에서는 분류를 사용하여 결정하게 되는데, 이를 위해서는 알고리즘을 훈련하거나 학습시켜야 한다.  
알고리즘을 훈련하기 위해서 훈련 집합(Training Set)이라는 양질의 데이터가 주어진다.  
위의 알고리즘에서는 표 1.1의 6개의 데이터가 훈련 집합이다.  
훈련 집합으로 훈련된 알고리즘에는 검사 집합(Test Set)을 프로그램에 넣어 검사한다.  
검사 집합에는 목적 변수가 주어지지 않으며, 훈련된 알고리즘이 목적 변수를 어떻게 예측하는지 알 수 있다.  
예측된 값과 검사 집합의 본래 목적 변수를 비교하여 알고리즘이 얼마나 정확한지 짐작할 수 있다.  
훈련 집합, 검사 집합 모두 사용하는것이 좋다.


####1.3 기계 학습의 주요 기술

표 1.2

| 지도 학습 방법 ||
| :---: | :---: |
| 분류 | 회귀 |
| k-최근접 이웃 | 선형 회귀 |
| 나이브 베이즈 | 지역적 가중치가 부여된 선형 회귀 |
| 서포트 벡터 머신 | 리지 |
| 의사결정 트리 | 라쏘 |

| 비지도 학습 방법 ||
| :---: | :---: |
| 군집화 | 밀도 추정 |
| k-평균 | 기댓값 최대화 |
| 디비스캔 | 파젠 윈도우 |

지도 학습의 경우는 알고리즘에 무엇을 예측할 것인지 제공한다.  
비지도 학습의 경우는 주어진 데이터에 분류 항목 표시나 목적 변수가 없다.  
비지도 학습 방법에는 유사한 아이템들을 모으는 군집화 방법과 통계적인 방법을 사용하여 값을 찾는 밀도 추정 방법이 있다  
속성이 많은 데이터의 경우는 적은 수의 속성으로 줄이거나, 2차원 또는 3차원으로 데이터를 그려보는 방법을 사용하기도 한다.  

####1.4 올바른 알고리즘 선정 방법

올바른 알고리즘을 선정하기 위해서는  
1. 목적을 고려한다.  
2. 보유하고 있는 데이터를 고려한다.  

1\. 목적을 고려한다.  

목적 값을 예측할 때에는 지도 학습 방법을, 그렇지 않다면 비지도 학습 방법을 살펴보는 것이 좋다.  
목적 변수의 값이 '예/아니오', 'A/B/C', '빨강/노랑/검정'과 같이 이산적인 값이라면 분류를,  
0.00 ~ 100.00, -999 ~ 999, +∞ ~ -∞ 같은 수치형 값이라면 회귀를 살펴봐야 한다.  

목적 값의 예측이 아니라면 비지도 학습을 살펴봐야 한다.  
가지고 있는 데이터가 어떤 이산적인 무리에 알맞은지를 알아보려면 군집화를,  
각각의 무리에 알맞은 정도를 수치적으로 평가하기 위해서는 밀도 추정 알고리즘을 살펴봐야 한다.

2\. 보유하고 있는 데이터를 고려한다.

데이터에 대해 더 많이 알아야만 성공적인 시스템을 구축할 수 있다.  
속성이 '명목형인가', '연속형인가', '속성 내에 누락된 값들은 없는가', '누락된 값이 있다면 데이터가 누락된 상황은 왜 존재하는가', '데이터 내에 오류가 있는가', '매우 드물게 발생하는 어떤 것이 존재하는가' 와 같은 데이터 속성에 대한 모든 것은 알고리즘 선택 과정의 폭을 좁히는 데 도움을 준다.


####1.5 기계 학습 응용 프로그램 개발 단계

1. 데이터 수집
2. 입력 데이터 준비
3. 입력 데이터 분석
4. 알고리즘 훈련
5. 알고리즘 테스트
6. 사용하기

---















