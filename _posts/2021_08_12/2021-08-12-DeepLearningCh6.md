---
title: "밑바닥부터 시작하는 딥러닝 Chapter 6"
excerpt: 학습 관련 기술들
categories: [Deep Learning]
tags: []
last_modified_at: 2021-08-12 12:18:00 +0900
---

> *이 포스트는 책 [<u>밑바닥부터 시작하는 딥러닝</u>](https://books.google.co.kr/books/about/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0_%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94_%EB%94%A5%EB%9F%AC%EB%8B%9D.html?id=SM9KDwAAQBAJ&source=kp_book_description&redir_esc=y)을 정리한 내용입니다.*


<br>

> # 6.1 매개변수 갱신
---

**최적화 (optimization)** : 손실 함수의 값을 가능한 낮추는 매개변수의 최적값을 찾는 문제를 푸는 것

**확률적 경사 하강법 (SGD)** : 매개변수의 기울기를 구해 기울어진 방향으로 매개변수 값을 갱신하는 것을 반복하는 최적화 방법

<br>

>> ## 6.1.2 확률적 경사 하강법 (SGD)
---

$$ W \leftarrow W - \eta\frac{\partial L}{\partial W}$$

$ W $ : 가중치 매개변수

$ \eta $ : 학습률

$ \frac{\partial L}{\partial W} $ : $ W $에 대한 손실 함수의 기울기

<br>

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

params : 가중치 매개변수들을 담은 딕셔너리

lr : 학습률

grads : 가중치 매개변수들에 대한 손실 함수의 기울기를 담은 딕셔너리

<br>

```python
network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...)
    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grads)
    ...
```

get_mini_batch()로 불러온 random한 데이터로 최적화하기 때문에 **'확률적'** 경사 하강법임

<br>

>> ## 6.1.3 SGD의 단점
---

$$ f(x, y) = \frac{1}{20}x^2 + y^2 $$

<br>

![그래프와 등고선](/assets/images/2021_08_12/6_1_3_1.PNG)

y축 방향으로 훨씬 가파른 모양의 함수

<br>

![기울기](/assets/images/2021_08_12/6_1_3_2.PNG)

(0, 0)보다는 y = 0을 향해 있음

<br>

![SGD 경로](/assets/images/2021_08_12/6_1_3_3.PNG)

매우 비효율적인 경로로 최적화가 진행됨

<br>

SGD는 비등방성 (anisotropy, 방향에 따라 성질 (기울기)이 달라지는) 함수에 대해서는 비효율적임

<br>

>> ## 6.1.4 모멘텀
---

**모멘텀 (momentum)** : 본래 물리학 용어로 운동량, 추진력 등을 의미하며 기하학에서는 곡선 위의 한 점의 기울기를 뜻함

$$ v \leftarrow \alpha v - \eta\frac{\partial L}{\partial W} $$

$$ W \leftarrow W + v $$

$ v $ : 속도

$ \alpha $ : 0.9 등의 값으로 아무 외력이 없을 때 서서히 하강시키는 역할

첫번째 수식은 기울기 방향으로 힘을 받아 물체가 가속된다는 물리 법칙을 나타냄

<br>

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```

<br>

![모멘텀 경로](/assets/images/2021_08_12/6_1_4.PNG)

SGD보다 효율적인 경로로 최적화가 진행됨

<br>

>> ## 6.1.5 AdaGrad
---

**학습률 감소 (learning rate decay)** : 학습을 진행하면서 학습률을 점차 줄여가는 방법

학습률을 낮추는 가장 간단한 방법은 매개변수의 전체 학습률 값을 일괄적으로 낮추는 것

**AdaGrad** : 각각의 매개변수에 따라 학습률을 낮추는 비율을 다르게 적용

$$ h \leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W} $$

$$ W \leftarrow W - \eta\frac{1}{\sqrt h}\frac{\partial L}{\partial W}$$

$ \odot $ : 행렬의 원소별 곱셈

$ h $ : 기울기 값의 제곱들의 합

$ \frac{1}{\sqrt h} $를 곱한다는 것은 매개변수의 원소 중에서 크게 갱신되었을수록 학습률을 낮추겠다는 의미

AdaGrad는 학습을 진행할수록 갱신 강도가 약해져 오랜 기간 학습 후엔 갱신량이 0에 수렴해서 갱신이 되지 않음

**RMSProp** : 이 단점을 개선한 기법으로 과거의 모든 기울기를 균일하게 더해가는 것이 아니라, 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영 (지수이동평균, EMA)

<br>

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

<br>

![AdaGrad 경로](/assets/images/2021_08_12/6_1_5.PNG)

모멘텀보다도 효율적인 경로로 최적화가 진행됨

<br>

>> ## 6.1.6 Adam
---

**Adam** : 2015년에 제안된, 모멘텀과 AdaGrad를 융합한 기법

<br>

![Adam 경로](/assets/images/2021_08_12/6_1_6.PNG)

모멘텀과 AdaGrad의 중간처럼 보임

<br>

>> ## 6.1.7 어느 갱신 방법을 이용할 것인가?
---

위 예시에선 AdaGrad가 제일 효율적이나 문제에 따라 결과가 달라짐

학습률 등의 하이퍼 파라미터도 고려해야함

<br>

>> ## 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교
---

<br>

![Adam 경로](/assets/images/2021_08_12/6_1_8.PNG)

100개의 뉴런으로 구성된 5층 신경망에서 활성화 함수로는 ReLU 사용

<br>

> # 6.2 가중치의 초깃값
---

>> ## 6.2.1 초깃값을 0으로 하면?
---

**가중치 감소 (weight decay) 기법** : 가중치 매개변수의 값이 작아지도록 학습하여 오버피팅을 억제하는 방법

가중치의 초깃값을 모두 0으로 설정하면 (정확히는 가중치를 모두 같은 값으로 설정하면) 학습이 이뤄지지 않음

왜냐하면 손실 함수의 값을 편미분한 값이 모두 동일하기 때문에 갱신량도 동일해서 가중치들끼리 다른 값을 가질 수 없음

그러므로 초깃값은 무작위로 설정해야함

<br>

>> ## 6.2.2 은닉층의 활성화값 분포
---

가중치의 초깃값에 따른 은닉층 활성화값들의 변화는 어떨까?

활성화 함수로 시그모이드 함수를 사용하는 5층 신경망에 무작위 데이터를 흘리며 각 층의 활성화값 분포의 히스토그램 관찰 실험

```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i:
        x = activations[i - 1]
    
    w = np.random.randn(node_num, node_num) * 1
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z
```

<br>

![표준편차 1](/assets/images/2021_08_12/6_2_2_1.PNG)

<br>

각 층의 활성화값들이 0과 1에 치우쳐 분포됨

출력이 0 또는 1에 가까워질수록 미분값은 0에 가까워져 역전파의 기울기 값이 점점 작아지다가 사라짐

이런 문제를 **기울기 소실** (gradient vanishing)이라고 함

<br>

가중치의 표준편차를 0.01로 바꿔 같은 실험을 반복

```python
    ...
    w = np.random.randn(node_num, node_num) * 0.01
    ...
```

<br>

![표준편차 0.01](/assets/images/2021_08_12/6_2_2_2.PNG)

<br>

이번에는 0.5 부근에 집중됨

기울기 소실 문제는 일어나지 않았지만 다수의 뉴런이 같은 값을 출력하고 있으니 **표현력을 제한**한다는 관점에서 문제가 됨

**Xavier 초깃값** : Xavier Glorot, Yoshua Bengio의 논문에서 권장하는 가중치 초깃값

앞 계층의 노드가 $ n $개라면 표준편차가 $ \frac{1}{\sqrt n} $인 분포를 사용

```python
    ...
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    ...
```

<br>

![Xavier 초깃값](/assets/images/2021_08_12/6_2_2_3.PNG)

<br>

앞의 두 방식보다 확실히 넓게 분포됨

그런데 층이 깊어질수록 오른쪽으로 쏠리는 경향을 보임

이는 sigmoid 함수가 (0, 0.5)에서 점대칭이기 때문

이를 보완하려면 원점 대칭인 tanh 함수를 활성화함수로 사용하면 됨

<br>

>> ## 6.2.3 ReLU를 사용할 때의 가중치 초깃값
---

**He 초깃값** : Kaiming He의 이름을 딴, ReLU에 특화된 초깃값

앞 계층의 노드가 $ n $개라면 표준편차가 $ \sqrt {\frac{2}{n}} $인 분포를 사용

ReLU의 치역이 음이 아닌 실수이므로 더 넓게 분포시키기 위해 Xavier 초깃값에 비해 $ \sqrt 2 $배 늘어났다고 해석하면 됨

<br>

![ReLU](/assets/images/2021_08_12/6_2_3.PNG)

<br>

위 두 방식은 기울기 소실 문제를 일으키지만 He 초깃값은 균일하게 분포함

<br>

## *결론*

|활성화 함수|초깃값|
|:--------:|:----:|
|sigmoid   |Xavier|
|tanh      |Xavier|
|ReLU      |He|

<br>

>> ## 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교
---























