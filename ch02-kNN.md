## __1부 분류__


### __2장 k-최근접 이웃 알고리즘(k-Nearest Neighbors)__

액션 영화를 액션 영화라고, 로맨스 영화를 로맨스 영화라고 부를 수 있는 이유는 많은 영화들이 해당 범주안의 어떤 특징을 가지고 있기 때문이다.  
액션 영화에서도 키스하는 장면이 나오고, 로맨스 영화에서도 발차기를 하는 장면이 나온다.  
하지만 로맨스 영화에 키스 장면이 더 많이 나오고, 액션 영화에 발차기 장면이 더 많이 나온다.  
키스나 발차기 또는 영화마다 다른 어떤 것들로 판단 기준을 삼는다면, 어떤 영화가 어떤 장르에 속하는지 알아낼 수 있을 것이다.  
이번 장에서는 영화 데이터를 사용해 k-최근접 이웃 알고리즘의 개념을 설명한다.  


#### 2.1 거리 측정을 이용하여 분류하기

|| k-최근접 이웃 알고리즘 |
| :---: | :---: |
| 장점 | 높은 정확도, 오류 데이터(outlier)에 둔감, 데이터에 대한 가정이 없음 |
| 단점 | 계산 비용이 높음, 많은 메모리 요구 |
| 적용 | 수치형 값, 명목형 값 |

훈련 집합의 모든 데이터에는 '분류 항목 표시(labels)'가 붙어 있고, 각 데이터가 어떤 분류 항목으로 구분되는지 알 수 있다.  
분류 항목 표시가 붙어 있지 않은 데이터를 추가했을 때,  
기존 데이터와 새로 추가된 데이터를 비교하여 가장 유사한 상위 k개의 데이터들 중 '다수결(majority vote)'을 통해 추가된 데이터의 분류 항목을 결정한다.  
다수결이라는 것은 분류 항목 표시의 개수를 뜻한다.  
예를 들어 분류 항목을 알지 못하는 X 와 가장 유사한 데이터가 A, B, C, D, E 이고 각각 분류 항목이 a, a, a, b, b 일 때 X는 다수결에 의해 a 분류 항목으로 예측된다.(a: 3, b: 2)  

다음은 로맨스 영화와 액션 영화를 분리하는 간단한 예제이다.  
표 2.1을 보면 영화와 해당 영화에 등장하는 발차기 장면, 키스 장면의 회수가 있다.  
?의 영화 유형을 알아내기 위해 k-최근접 이웃 알고리즘을 사용할 것이다.  

표 2.1

| 영화 제목 | 발차기 장면 횟수 | 키스 장면 횟수 | 유형 |
| :---: | :---: | :---: | :---: |
| California Man | 3 | 104 | 로맨스 |
| He's Not Really into Dudes | 2 | 100 | 로맨스 |
| Beautiful Woman | 1 | 81 | 로맨스 |
| Kevin Longblade | 101 | 10 | 액션 |
| Robo Slayer 3000 | 99 | 5 | 액션 |
| Amped 2 | 98 | 2 | 액션 |
| ? | 18 | 90 | 알 수 없음 |

물음표에 해당하는 영화의 유형이 무엇인지 알기 위해서 다른 모든 영화들과의 거리를 계산한다.  
(발차기 장면 횟수, 키스 장면 횟수) 2차원 xy 좌표계를 만들 수 있고, 점과 점 사이의 거리를 계산할 수 있다.  
?와 California Man을 예시로 들면,  

![equation](https://latex.codecogs.com/gif.latex?distance%20%3D%20%5Csqrt%7B%2818%20-%203%29%5E%202%20&plus;%20%2890%20-%20104%29%5E2%7D%20%5Capprox%2020.5)  
이고, 해당 과정을 모든 영화들에게 반복하면 표 2.2가 된다.

표 2.2

| 영화 제목 | 영화 '?' 와의 거리 |
| :---: | :---: |
| California Man | 20.5 |
| He's Not Really into Dudes | 18.7 |
| Beautiful Woman | 19.2 |
| Kevin Longblade | 115.3 |
| Robo Slayer 3000 | 117.4 |
| Amped 2 | 118.9 |

k는 보통 20 미만의 정수를 사용하고, 이 예제에서는 k = 3 이다.  
가장 유사한 상위 3개의 영화는 California Man, He's Not Really into Dudes, Beautiful Woman 이고 세 영화 모두 로맨스이므로 다수결에 의해 ?의 영화는 로맨스로 예측할 수 있다.  

k-최근접 이웃 알고리즘은 거리 측정이라는 매우 간단한 방법으로 데이터를 분류하는 데 효과적인 방법이다.  
사례 기반 학습(instance-based learning)의 한 예시이고, 다루기 쉬운 데이터 사례가 있어야만 한다.  
또한 모든 데이터를 계산하고 처리해야 하기 때문에 많은 컴퓨터 자원을 필요로 하고 데이터들의 구조에 대한 정보를 알 수 없다는 단점이 있다.  
즉, 유사한 데이터들에 대해 '평균' 혹은 '좋은 사례'를 알 수 없다.

---