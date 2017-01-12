##__1부 분류__


###__4장 나이브 베이즈(Naive Bayes)__


k-NN이나 의사결정 트리와 같은 알고리즘은 분류기로써 가끔 잘못된 예측을 하기도 했다.  
이러한 잘못된 예측 대신, 분류기에게 확률을 요청하도록 만들 수도 있다.  

많은 기계 학습 알고리즘들은 확률 이론에 기반을 두고 있다.  
이번 장에서는 의사결정 트리의 참-거짓에 대한 부분을 확률로 다룬다.  
같은 속성이 등장한 횟수를 데이터 집합의 전체 사례의 수로 나눠 확률을 계산한다.  

이번 장에서는 확률 이론으로 데이터를 분류하는 몇 가지 방법을 알아볼 것이다.  
간단한 확률 분류기부터 약간의 가정을 통한 나이브 베이즈(Naive Bayes) 분류기를 학습한다.  
'나이브(Naive)' 라는 단어를 사용하는 이유는 몇 가지 간단한(Naive)한 가정을 통해 만들어진 것이기 때문이다.  
이 장에서는 조건부 확률에 대해서도 다룰 것이다.  

####4.1 베이지안 의사결정 이론으로 분류하기

|| 나이브 베이즈 |
| :---: | :---: |
| 장점 | 소량의 데이터를 가지고 작업이 이루어지며, 여러 개의 분류 항목을 다룰 수 있다. |
| 단점 | 입력 데이터에 따라 민감하게 작용한다. |
| 적용 | 명목형 값 |

나이브 베이즈는 베이즈 정리의 일부분이다.  
따라서 나이브 베이즈 이전에 베이즈 정리 이론을 짚고 넘어갈 것이다.  

데이터 내에 두 개의 분류 항목이 있는 데이터 집합을 가정한다.  
분류 항목 1에 속할 확률 방정식 ![equation](https://latex.codecogs.com/gif.latex?p1%28x%2C%20y%29), 분류 항목 2에 속할 확률 방정식 ![equation](https://latex.codecogs.com/gif.latex?p2%28x%2C%20y%29)가 있다.  
속성 (x, y)를 가지고 분류 확률을 선택할 때 다음과 같은 규칙을 따른다.  

![equation](https://latex.codecogs.com/gif.latex?p1%28x%2C%20y%29%20%3E%20p2%28x%2C%20y%29)이면 분류항목 1에 속한다.  
![equation](https://latex.codecogs.com/gif.latex?p2%28x%2C%20y%29%20%3E%20p1%28x%2C%20y%29)이면 분류항목 1에 속한다.  

요약하면, 더 높은 확률을 갖는 분류 항목을 선택한다.  
베이즈 정리 이론은 결국 높은 확률을 갖는 의사결정을 선택하는 것이다.  

1000개의 데이터를 갖는 집합에 대해 의사결정을 선택해야 할 때, 다음 세 가지 방법이 있다.  
1. k-NN을 사용하여 1000개의 거리 계산을 수행한다.  
2. 의사결정 트리를 사용하여 데이터를 분할한다.  
3. 각 분류 항목의 확률은 계산하고 비교한다.  

의사결정 트리는 매우 성공적이지는 않았고, k-NN은 간단한 확률을 비교하는 많은 계산이 요구된다.  
이러한 점들을 감안한다면 확률적인 비교가 가장 효과적일 수 있다.  

####4.2 조건부 확률

7개의 돌이 담긴 병이 있다.  
그림 4.1와 같이 3개는 회색, 4개는 검은색의 돌이다.  
그림 4.1

![image](https://s24.postimg.org/y6vhu6zr9/image.png)

이 중 임의로 하나의 돌을 꺼낸다면, 뽑은 돌이 회색일 확률은 3/7 이다.  
이것을 ![equation](https://latex.codecogs.com/gif.latex?p%28gray%29)라고 하고, 이 값은 회색 돌의 개수를 세고, 전체 돌의 개수를 나눔으로써 계산된다.  

그렇다면 돌들이 두 병에 나눠져 있다면 어떻게 될 지 생각해볼 수 있다.

그림 4.2, 병 A

![image](https://s23.postimg.org/wfd8hafqj/image.png)

그림 4.3, 병 B

![image](https://s27.postimg.org/q3u5kdy77/image.png)

그림 4.2와 그림 4.3에서 ![equation](https://latex.codecogs.com/gif.latex?p%28gray%29) 나 ![equation](https://latex.codecogs.com/gif.latex?p%28black%29) 를 계산한다면 어떤 병을 선택하냐에 따라 확률이 달라질 것이다.  
병 A에서 회색 돌을 꺼낼 확률을 ![equation](https://latex.codecogs.com/gif.latex?p%28gray%7CA%29)라고 하고 이것을 조건부 확률이라 한다.  
![equation](https://latex.codecogs.com/gif.latex?p%28gray%7CA%29)는 2/4 이고, ![equation](https://latex.codecogs.com/gif.latex?p%28gray%7CB%29)는 2/3 이다.  
조건부 확률의 계산 방법은 다음과 같다.  
![equation](https://latex.codecogs.com/gif.latex?p%28gray%7CB%29%20%3D%20p%28gray%20%5Ccap%20B%29%20/%20p%28B%29)

![equation](https://latex.codecogs.com/gif.latex?p%28gray%20%5Ccap%20B%29)가 2/7이 되는지 확인해보면, 병 B에 있는 회색 돌을 전체 돌의 개수로 나누는 것으로 계산할 수 있다.  
병 B에는 7개의 돌 중 3개가 들어 있으므로 ![equation](https://latex.codecogs.com/gif.latex?p%28B%29)는 3/7이다.  
다음으로 ![equation](https://latex.codecogs.com/gif.latex?p%28gray%7CB%29)는 앞에서 2/3이었기 때문에 우리는 원하는 값을 얻을 수 있다.  

조건부 확률을 다루는 유용한 방법 중 베이즈 규칙(Bayes' rule)이 있다.  
조건부 확률에서 베이즈 규칙은 기호 교환에 대해 설명한다.  
만약 ![equation](https://latex.codecogs.com/gif.latex?p%28x%7Cc%29)을 알고 있는 상태에서 ![equation](https://latex.codecogs.com/gif.latex?p%28c%7Cx%29)를 알고 싶다면, 다음과 같이 찾을 수 있다.  

![equation](https://latex.codecogs.com/gif.latex?p%28c%7Cx%29%20%3D%20%5Cfrac%7Bp%28x%7Cc%29%20p%28c%29%7D%7B%7D%20%7Bp%28x%29%7D)

####4.3 조건부 확률로 분류하기
조건부 확률을 알았으니 앞에서 언급한 다음 두 가지를 확장해볼 수 있다.
![equation](https://latex.codecogs.com/gif.latex?p1%28x%2C%20y%29%20%3E%20p2%28x%2C%20y%29)이면 분류항목 1에 속한다.  
![equation](https://latex.codecogs.com/gif.latex?p2%28x%2C%20y%29%20%3E%20p1%28x%2C%20y%29)이면 분류항목 1에 속한다.  

이러한 간단한 규칙으로는 전체 내용을 설명하지는 못한다.  
우리에게 필요한 ![equation](https://latex.codecogs.com/gif.latex?p%28c_1%7Cx%2Cy%29)와 ![equation](https://latex.codecogs.com/gif.latex?p%28c_2%7Cx%2Cy%29)을 비교해보도록 한다.  
x, y로 정해진 한 지점이 있다면, 이 지점이 분류 항목 ![equation](https://latex.codecogs.com/gif.latex?c_1)에 속할 확률이 높은지, 분류항목 ![equation](https://latex.codecogs.com/gif.latex?c_2)에 속할 확률이 높은지에 대한 문제를 해결하기 위해 베이즈 규칙을 사용한다.  

![equation](https://latex.codecogs.com/gif.latex?p%28c_i%7Cx%2Cy%29%20%3D%20%5Cfrac%7Bp%28x%2Cy%7Cc_i%29%20p%28c_i%29%7D%7Bp%28x%2Cy%29%7D)

이 정의를 가지고 베이지안 분류 규칙을 정의할 수 있다.  

![equation](https://latex.codecogs.com/gif.latex?p%28c_1%7Cx%2Cy%29%20%3E%20p%28c_2%7Cx%2Cy%29)라면 분류 항목 ![equation](https://latex.codecogs.com/gif.latex?c_1)에 속한다.  
![equation](https://latex.codecogs.com/gif.latex?p%28c_2%7Cx%2Cy%29%20%3E%20p%28c_1%7Cx%2Cy%29)라면 분류 항목 ![equation](https://latex.codecogs.com/gif.latex?c_2)에 속한다.  

이렇게 베이즈 규칙을 사용함으로써 알려진 확률로부터 알려지지 않은 것을 계산하고, 아이템들을 분류할 수 있다.

####4.4 나이브 베이즈로 문서 분류하기

기계 학습의 한 가지 중요한 응용 프로그램은 자동 문서 분류 프로그램이 있다.  
문서 분류에 있어 전체 문서, 각각의 이메일은 하나의 사례가 되고, 이메일을 구성하고 있는 요소(단어)들은 속성이 된다.  
속성들이 서로 독립적이라고 가정하면 데이터의 수는 하나의 단어가 다른 단어 옆에도 나타날 확률이 같다고 볼 수 있다.  
즉, '베이컨(bacon)'이 '맛있는(delicious)'이라는 단어와 '건강에 해로운(unhealthy)'이라는 단어 옆에 나타날 확률을 같다고 가정한다.  
또한 모든 속성에 대해 중요도도 같다고 가정한다.  
이러한 가정들 때문에 나이브라는 단어가 붙었지만, 나이브 베이즈는 이런 결함들을 가지고도 일을 잘 수행한다.  

문서의 단어 벡터 ![w](https://latex.codecogs.com/gif.latex?w)를 가지고 해당 문서가 어떤 분류 항목(![c_i](https://latex.codecogs.com/gif.latex?c_i))에 속하는지 검사하기 위해 다음과 같은 수식을 이용한다.  

![equation](https://latex.codecogs.com/gif.latex?p%28c_i%7Cw%29%20%3D%20%5Cfrac%7Bp%28w%7Cc_i%29p%28c_i%29%7D%7Bp%28w%29%7D)

왼쪽의 값을 구하기 위해, 오른쪽의 값을 계산한다.  
오른쪽의 벡터 ![w](https://latex.codecogs.com/gif.latex?w)는 단어들의 벡터인데, 이를 개별적인 속성으로 펼치면 다음과 같다.

![c_i](https://latex.codecogs.com/gif.latex?p%28w_0%2Cw_1%2Cw_2%2C...%2Cw_n%7Cc_i%29)

위 식에서 모든 단어들이 서로 독립적이라는 가정을 통해 이에 대한 확률은 ![](https://latex.codecogs.com/gif.latex?p%28w_0%7Cc_i%29p%28w_1%7Cc_i%29p%28w_2%7Cc_i%29...p%28w_n%7Cc_i%29)로 쉽게 계산할 수 있다.  

이렇듯 나이브한 가정들을 통해 문서에 대한 속성들로, 문서들을 쉽게 분류할 수 있고 이는 조금 무리한 가정이 섞여있더라도 충분히 잘 작동한다.

---
