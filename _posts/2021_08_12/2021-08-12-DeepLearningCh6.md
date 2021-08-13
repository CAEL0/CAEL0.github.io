---
title: "밑바닥부터 시작하는 딥러닝 Chapter 6"
excerpt: 학습 관련 기술들
categories: [Deep Learning]
tags: [밑바닥부터 시작하는 딥러닝]
last_modified_at: 2021-08-13 13:48:00 +0900
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

첫 번째 수식은 기울기 방향으로 힘을 받아 물체가 가속된다는 물리 법칙을 나타냄

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

학습률 등의 하이퍼 파라미터도 고려해야 함

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

그러므로 초깃값은 무작위로 설정해야 함

<br>

>> ## 6.2.2 은닉층의 활성화값 분포
---

가중치의 초깃값에 따른 은닉층 활성화 값들의 변화는 어떨까?

활성화 함수로 시그모이드 함수를 사용하는 5층 신경망에 무작위 데이터를 흘리며 각 층의 활성화 값 분포의 히스토그램 관찰 실험

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

각 층의 활성화 값들이 0과 1에 치우쳐 분포됨

출력이 0 또는 1에 가까워질수록 미분 값은 0에 가까워져 역전파의 기울기 값이 점점 작아지다가 사라짐

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

이를 보완하려면 원점 대칭인 tanh 함수를 활성화 함수로 사용하면 됨

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

*결론*

|활성화 함수|초깃값|
|:--------:|:----:|
|sigmoid   |Xavier|
|tanh      |Xavier|
|ReLU      |He|

<br>

>> ## 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교
---

<br>

![비교](/assets/images/2021_08_12/6_2_4.PNG)

<br>

층별 뉴런 수가 100개인 5층 신경망에서 활성화 함수로 ReLU를 사용

std = 0.01인 경우 학습이 전혀 되지 않음

나머지 둘은 학습이 잘 되며, He가 조금 더 빠름

<br>

> # 6.3 배치 정규화
---

각 층이 활성화를 적당히 퍼뜨리도록 강제할 수 있을까?

<br>

>> ## 6.3.1 배치 정규화 알고리즘
---

**배치 정규화** (Batch Normalization)의 장점

* 학습을 빨리 진행할 수 있음 (학습 속도 개선)

* 초깃값에 크게 의존하지 않음

* 오버피팅 억제 (드롭아웃 등의 필요성 감소)

<br>

![배치 정규화 계층](/assets/images/2021_08_12/6_3_1_1.PNG)

<br>


배치 정규화 계층을 신경망에 삽입

학습 시 미니배치 단위로 평균이 0, 분산이 1이 되도록 정규화

$$ B = \{ x_1, x_2, ..., x_m \} $$

$$ \mu_B \leftarrow \frac{1}{m} \sum_{i = 1}^m x_i $$

$$ \sigma_B^2 \leftarrow \frac{1}{m} \sum_{i = 1}^m (x_i - \mu_B)^2 $$

$$ \hat x_i \leftarrow \frac{x_i - \mu_B}{\sqrt {\sigma_B^2 + \epsilon}} $$

이 처리를 활성화 함수의 앞 혹은 뒤에 삽입함으로써 데이터 분포가 덜 치우치게 할 수 있음

또한 배치 정규화 계층마다 이 정규화된 데이터에 고유한 확대와 이동 변환 수행

$$ y_i \leftarrow \gamma \hat x_i + \beta $$

$ \gamma $ : 확대

$ \beta $ : 이동

처음에는 $ \gamma = 1, \beta = 0 $부터 시작하고 학습하면서 적합한 값으로 조정

<br>

![계산 그래프](/assets/images/2021_08_12/6_3_1_2.PNG)

<br>

>> ## 6.3.2 배치 정규화의 효과
---

<br>

![효과](/assets/images/2021_08_12/6_3_2_1.PNG)

<br>

배치 정규화가 학습을 빨리 진전시킴

<br>

![비교](/assets/images/2021_08_12/6_3_2_2.PNG)

<br>

배치 정규화를 사용한 경우 초깃값에 상관없이 대체로 학습 진도가 빠름

배치 정규화를 사용하지 않은 경우에는 초깃값에 따라 학습이 진행되지 않기도 함

<br>

> # 6.4 바른 학습을 위해
---

>> ## 6.4.1 오버피팅
---

오버피팅은 주로 다음의 두 경우에 일어남

* 매개변수가 많고 표현력이 높은 모델

* 훈련 데이터가 적음

두 요건을 충족시켜 일부러 오버피팅을 일으켜봄

60000개인 MNIST 데이터셋의 훈련 데이터 중 300개만 사용

각 층의 뉴런은 100개, 활성화 함수는 ReLU, 총 7층 네트워크 사용

<br>

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
```

<br>

![정확도](/assets/images/2021_08_12/6_4_1.PNG)

<br>

훈련 데이터의 정확도는 1에 가까워지나 시험 데이터는 저조함

<br>

>> ## 6.4.2 가중치 감소
---

**가중치 감소** (weight decay) : 오버피팅 억제 방법 중 하나로 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 페널티를 부과

가중치를 $ W $라고 할 때 $ \frac{1}{2} \lambda W^2$을 손실 함수에 더함

$ \lambda $는 정규화의 세기를 조절하는 하이퍼파라미터

이를 적용하려면 오차역전파법에 따른 결과에 정규화 항을 미분한 $ \lambda W $를 더하면 됨

<br>

![가중치 감소](/assets/images/2021_08_12/6_4_2.PNG)

<br>

가중치 감소를 적용하자 훈련 데이터와 시험 데이터의 정확도 차이가 줄어듦

즉, 오버피팅이 억제됨

<br>

>> ## 6.4.3 드롭아웃
---

**드롭아웃** (Dropout) : 은닉층의 뉴런을 임의로 삭제하면서 학습하는 방법

훈련 때는 데이터를 흘릴 때마다 삭제할 뉴런을 무작위로 선택하고, 시험 때는 모든 뉴런에 신호를 전달

시험 때 각 뉴런의 출력에 훈련 때 삭제 안 한 비율을 곱하여 출력

<br>

![드롭아웃](/assets/images/2021_08_12/6_4_3_1.PNG)

<br>

```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        
        return x * (1 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
```

<br>

![드롭아웃 비교](/assets/images/2021_08_12/6_4_3_2.PNG)

<br>

드롭아웃을 이용하면 표현력을 높이면서도 오버피팅을 억제할 수 있음

<br>

> # 6.5 적절한 하이퍼파라미터 값 찾기
---

>> ## 6.5.1 검증 데이터
---

하이퍼파라미터의 성능을 평가할 때는 시험 데이터를 사용해서는 안 됨

그렇지 않으면 하이퍼파라미터 값이 시험 데이터에 오버피팅 됨

그러므로 하이퍼파라미터 전용 확인 데이터가 필요 $ \rightarrow $ **검증 데이터** (validation data)

* 훈련 데이터 : 매개변수 학습
* 검증 데이터 : 하이퍼파라미터 성능 평가
* 시험 데이터 : 신경망의 범용 성능 평가

검증 데이터를 얻는 가장 간단한 방법은 훈련 데이터 중 20% 정도를 검증 데이터로 먼저 분리하는 것

```python
(x_train, t_train), (x_test, t_test) = load_mnist()

x_train, t_train = shuffle_dataset(x_train, t_train)

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
```

<br>

>> ## 6.5.2 하이퍼파라미터 최적화
---

하이퍼파라미터의 최적값이 존재하는 범위를 조금씩 줄여나감

우선 대략적인 범위를 설정 그 범위에서 무작위로 하이퍼파라미터 값을 골라낸 후, 그 값으로 정확도를 평가

범위는 주로 로그 스케일로 지정

학습 시간이 오래 걸릴 수 있으므로 에폭을 작게 하여 1회 평가에 걸리는 시간을 단축하는 것이 효과적

0. 하이퍼파라미터 값의 범위를 설정

1. 설정된 범위에서 하이퍼파라미터의 값을 무작위로 추출

2. 1단계에서 샘플링한 하이퍼파라미터 값을 사용하여 학습 후 검증 데이터로 정확도를 평가 (에폭을 작게 설정)

3. 1단계와 2단계를 특정 횟수 반복하며, 그 정확도의 결과를 보고 하이퍼파라미터의 범위를 좁힘

<br>

>> ## 6.5.3 하이퍼파라미터 최적화 구현하기
---

MNIST 데이터셋을 사용하여 학습률과 가중치 감소 계수를 탐색

로그 스케일 범위에서의 무작위 추출은 다음과 같은 코드를 이용

```python
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
```

<br>

![하이퍼파라미터 최적화](/assets/images/2021_08_12/6_5_3_1.PNG)

<br>

![하이퍼파라미터 값](/assets/images/2021_08_12/6_5_3_2.PNG)

<br>

학습이 잘 될 때의 학습률은 0.001~0.01, 가중치 감소 계수는 $ 10^{-8} \sim 10^{-6} $ 정도임

이렇게 적절한 값이 위치한 범위를 좁혀가다가 특정 단계에서 최종 하이퍼파라미터 값을 하나 선택

<br>

> # 6.6 정리
---

* 매개변수 갱신 방법에는 SGD 외에도 모멘텀, AdaGrad, Adam 등이 있음

* 가중치 초깃값을 정하는 방법은 올바른 학습을 하는 데 매우 중요함

* 가중치의 초깃값으로는 Xavier 초깃값과 He 초깃값이 효과적

* 배치 정규화를 이용하면 학습을 빠르게 진행할 수 있으며, 초깃값에 영향을 덜 받음

* 오버피팅을 억제하는 정규화 기술로는 가중치 감소와 드롭아웃이 있음

* 하이퍼파라미터 값 탐색은 최적값이 존재할 법한 범위를 점차 좁히면서 하는 것이 효과적