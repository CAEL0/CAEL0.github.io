---
title: "밑바닥부터 시작하는 딥러닝 Chapter 7"
excerpt: 합성곱 신경망 (CNN)
categories: [Deep Learning]
tags: [밑바닥부터 시작하는 딥러닝]
last_modified_at: 2021-08-17 11:29:00 +0900
---

> *이 포스트는 책 [<u>밑바닥부터 시작하는 딥러닝</u>](https://books.google.co.kr/books/about/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0_%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94_%EB%94%A5%EB%9F%AC%EB%8B%9D.html?id=SM9KDwAAQBAJ&source=kp_book_description&redir_esc=y)을 정리한 내용입니다.*

<br>

**합성곱 신경망** (Convolutional Neural Network) : 이미지 인식과 음성 인식 등 다양한 곳에서 사용되는 인공 신경망

**합성곱** (convolution) : 두 함수 중 하나를 반전, 이동시켜가며 나머지 함수와의 곱을 연이어 적분

$$ (f * g)(t) = \int_{-\infty}^\infty f(\tau)g(t - \tau)d\tau $$

<br>

> # 7.1 전체 구조
---

지금까지 본 신경망처럼 계층을 조합하여 만들 수 있음

**합성곱 계층** (convolutional layer)과 **풀링 계층** (pooling layer)이 새롭게 등장

**완전연결** (fully-connected) : 인접하는 계층의 모든 뉴런과 결합된 형태의 신경망, 지금까지 Affine 계층이라고 부름

<br>

![완전연결 네트워크](/assets/images/2021_08_16/7_1_1.PNG)

Affine 계층과 ReLU 계층으로 이뤄진 네트워크

<br>

![CNN 네트워크](/assets/images/2021_08_16/7_1_2.PNG)

합성곱 계층 (Conv)과 풀링 계층 (Pooling)이 추가됨

3번째 층처럼 풀링 계층은 생략하기도 함

Affine-ReLU $ \rightarrow $ Conv-ReLU-(Pooling)

출력에 가까운 층에서는 Affine-ReLU 구성을 사용할 수 있음

마지막 출력 층은 Affine-Softmax 조합을 그대로 사용

<br>

> # 7.2 합성곱 계층
---

>> ## 7.2.1 완전연결 계층의 문제점
---

**데이터의 형상이 무시됨**

데이터가 이미지일 경우, 보통 이미지는 가로, 세로, 채널 (색상)으로 구성된 3차원 데이터이지만 완전연결 계층에 입력시킬 땐 1차원 데이터로 평탄화해줘야 함

MNIST 데이터셋을 예로 들면, 형상이 (1, 28, 28)이었던 이미지를 (784, )로 바꿔 입력시킴

이렇게 하면 데이터의 공간적 정보를 살릴 수 없음

$ \rightarrow $ CNN은 데이터의 형상을 유지함

**특징 맵** (feature map) : CNN에서 합성곱 계층의 입출력 데이터

<br>

>> ## 7.2.2 합성곱 연산
---

![합성곱 연산](/assets/images/2021_08_16/7_2_2_1.PNG)

데이터의 형상 = (높이, 너비) = (행 개수, 열 개수)

입력 : (4, 4) / 필터 (커널) : (3, 3) / 출력 : (2, 2)

**윈도우** (window) : 필터가 입력 데이터와 겹치는 부분

**단일 곱셈-누산** (fused multiply-add, FMA) : 대응하는 원소끼리 곱한 후 총합을 구하는 계산

<br>

![합성곱 연산 과정](/assets/images/2021_08_16/7_2_2_2.PNG)

윈도우를 일정 간격 이동해가며 FMA를 시행함

CNN에서의 필터의 매개변수 = 완전연결 신경망에서의 가중치 매개변수

<br>

![합성곱 연산 편향](/assets/images/2021_08_16/7_2_2_3.PNG)

FMA 결과의 각 원소에 편향을 더해주면 출력 데이터가 됨

<br>

>> ## 7.2.3 패딩
---

**패딩** (padding) : 입력 데이터 주변을 특정 값으로 채움

<br>

![패딩 예시](/assets/images/2021_08_16/7_2_3.PNG)

(4, 4) 크기의 입력 데이터에 폭이 1인 패딩을 적용

패딩이 추가되어 입력 데이터의 크기가 (6, 6)이 됐고, (4, 4) 크기의 출력 데이터가 생성됨

패딩은 주로 출력 크기를 조정할 목적으로 사용

<br>

>> ## 7.2.4 스트라이드
---

**스트라이드** (stride) : 필터를 적용하는 위치의 간격

<br>

![스트라이드 2 예시](/assets/images/2021_08_16/7_2_4.PNG)

스트라이드를 2로 하면 필터를 적용하는 윈도우가 두 칸씩 이동

입력 크기 : (H, W) / 필터 크기 : (FH, FW) / 출력 크기 : (OH, OW) / 패딩 : P / 스트라이드 : S

$$ OH = \frac{H + 2P - FH}{S} + 1 $$

$$ OW = \frac{W + 2P - FW}{S} + 1 $$

<br>

>> ## 7.2.5 3차원 데이터의 합성곱 연산
---

![3차원 합성곱 예시](/assets/images/2021_08_16/7_2_5_1.PNG)

![3차원 합성곱 과정](/assets/images/2021_08_16/7_2_5_2.PNG)

입력 데이터의 채널 수와 필터의 채널 수가 같아야 함

모든 필터의 크기가 같아야 함

<br>

>> ## 7.2.6 블록으로 생각하기
---

3차원의 합성곱 연산은 데이터와 필터를 직육면체 블록이라고 생각

입력 데이터 : 채널 수 C / 높이 H / 너비 W $ \rightarrow $ (C, H, W)

필터 : 채널 수 C / 높이 FH / 너비 FW $ \rightarrow $ (C, FH, FW)

<br>

![3차원 합성곱 블록](/assets/images/2021_08_16/7_2_6_1.PNG)

필터를 1개만 사용하면 출력 데이터의 채널은 1개임

<br>

![3차원 합성곱 다수의 필터](/assets/images/2021_08_16/7_2_6_2.PNG)

필터를 FN개 적용 $ \rightarrow $ 출력 맵의 채널도 FN개

그러므로 필터의 가중치 데이터는 4차원 : (출력 채널 수, 입력 채널 수, 높이, 너비)

<br>

![3차원 합성곱 다수의 필터 + 편향](/assets/images/2021_08_16/7_2_6_3.PNG)

편향은 채널 하나에 값 하나씩으로 구성됨 : (FN, 1, 1)

<br>

>> ## 7.2.7 배치 처리
---

각 계층에 흐르는 데이터의 차원을 하나 늘려 4차원으로 저장

(데이터 수, 채널 수, 높이, 너비)

<br>

![배치 처리](/assets/images/2021_08_16/7_2_7.PNG)

신경망에 4차원 데이터가 하나 흐를 때마다 데이터 N개에 대한 합성곱 연산이 이뤄짐

<br>

> # 7.3 풀링 계층
---

세로, 가로 방향의 공간을 줄이는 연산

<br>

![풀링 예시](/assets/images/2021_08_16/7_3.PNG)

2x2 **최대 풀링** (max pooling)을 스트라이드 2로 처리하는 예시

보통 풀링의 윈도우 크기와 스트라이드는 같은 값으로 설정

최대 풀링 외에도 평균 풀링 등이 있지만 주로 최대 풀링을 사용

<br>

>> ## 7.3.1 풀링 계층의 특징
---

명확한 처리이므로 학습해야 할 매개변수가 없음

<br>

![채널 수 변화 없음](/assets/images/2021_08_16/7_3_1_1.PNG)

채널마다 적용하는 연산이므로 채널 수가 변하지 않음

<br>

![강건함](/assets/images/2021_08_16/7_3_1_2.PNG)

입력의 변화에 영향을 적게 받음

<br>

> # 7.4 합성곱/풀링 계층 구현하기
---

>> ## 7.4.1 4차원 배열
---

```python
x = np.random.rand(10, 1, 28, 28)
x.shape  # (10, 1, 28, 28)
```

높이 28, 너비 28, 채널 1개인 데이터 10개 무작위 생성

<br>

```python
x[0].shape  # (1, 28, 28)
x[1].shape  # (1, 28, 28)
```

x\[i\]로 i번째 데이터에 접근

<br>

```python
x[0, 0].shape  # (28, 28)
x[0][0].shape  # (28, 28)
```

x\[i\]\[j\] (혹은 x\[i, j\])로 i번째 데이터의 j번째 채널의 공간 데이터에 접근

<br>

>> ## 7.4.2 im2col로 데이터 전개하기
---

합성곱 연산을 for문으로 구현하려면 적어도 4중 for문을 써야함

Numpy에 for문을 사용하면 성능 저하가 일어나므로 **im2col** (image to column)이라는 편의 함수를 사용해 구현

<br>

![im2col 예시1](/assets/images/2021_08_16/7_4_2_1.PNG)

배치 안의 데이터 수까지 포함한 4차원 입력 데이터를 2차원 행렬로 변환

<br>

![im2col 예시2](/assets/images/2021_08_16/7_4_2_2.PNG)

입력 데이터를 필터링 (가중치 계산)하기 좋게 전개

입력 데이터에서 필터를 적용하는 영역 (3차원 블록)을 한 줄로 늘어놓음

스트라이드가 작아 필터 적용 영역이 겹치게 되면 im2col로 전개한 후의 원소 수가 원래 블록의 원소 수보다 많으므로 메모리를 더 많이 소비함

<br>

![im2col 예시3](/assets/images/2021_08_16/7_4_2_3.PNG)

합성곱 계층의 필터 (가중치) 또한 1열로 전개 후 행렬곱을 계산 (Affine 계층에서 했던 계산과 거의 유사)

마지막으로 출력 결과인 2차원 행렬을 4차원으로 변형

<br>

>> ## 7.4.3 합성곱 계층 구현하기
---

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    다수의 이미지를 입력받아 2차원 배열로 변환 (평탄화)
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col
```

```python
x1 = np.random.rand(1, 3, 7, 7)  # (데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # (9, 75)
```

col1의 높이는 (7 - 5 + 1) * (7 - 5 + 1) = 9, 너비는 3 * 5 * 5 = 75

즉 각 행은 윈도우의 위치에 따른, 필터와 합성곱을 수행하게 되는 입력 데이터의 원소들임

<br>

```python
x2 = np.random.rand(10, 3, 7, 7)  # (데이터 수, 채널 수, 높이, 너비)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)  # (90, 75)
```

입력 데이터가 여러 개일 땐 행을 추가함

<br>

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # 필터 (개수, 채널, 높이, 너비)
        self.b = b  # 편향 (개수, )
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # col.shape = (N * out_h * out_w, C * FH * FW)
        col = im2col(x, FH, FW, self.stride, self.pad)

        # col_W.shape = (C * FH * FW, FN)
        col_W = self.W.reshape(FN, -1).T

        # np.dot(col, col_W).shape = (N * out_h * out_w, FN)
        # self.b.shape = (FN, )
        # broadcasting
        # out.shape = (N * out_h * out_w, FN)
        out = np.dot(col, col_W) + self.b

        # out.shape = (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
```

4차원 입력 데이터 x를 im2col 함수를 통해 2차원으로 만들어주고, 이 행렬과 곱할 수 있도록 필터 W의 형상을 바꿔줌

행렬 곱 후 편향을 더한 결과를 다시 4차원으로 만들어주고, 축의 순서를 원래대로 변경함

<br>

![transpose](/assets/images/2021_08_16/7_4_3.PNG)

합성곱 계층의 역전파를 계산할 때에는 im2col을 역으로 처리해야 함 (col2im)

<br>

>> ## 7.4.4 풀링 계층 구현하기
---

![풀링 전개](/assets/images/2021_08_16/7_4_4_1.PNG)

풀링 계층 구현도 합성곱 계층처럼 im2col을 이용해 입력 데이터를 전개하지만 풀링 적용 영역을 채널마다 독립적으로 전개함

<br>

![풀링 forward](/assets/images/2021_08_16/7_4_4_2.PNG)

풀링을 적용한 후 형상을 변환함

<br>

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # col.shape = (N * out_h * out_w, C * pool_h * pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)

        # col.shape = (N * out_h * out_w * C, pool_h * pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # out.shape = (N * out_h * out_w * C, 1)
        out = np.max(col, axis=1)

        # out.shape = (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out
```

im2col 함수를 통해 4차원 입력 데이터 x를 2차원으로 만들어 준 뒤, 각 행 별로 풀링을 적용시키기 위해 형상을 바꿔줌

합성곱의 forward와 마찬가지로 out의 축의 순서를 x와 동일하게 바꿔줌

<br>

> # 7.5 CNN 구현하기
---

![CNN 구현](/assets/images/2021_08_16/7_5.PNG)

손글씨 숫자 인식 CNN

<br>

```python
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['strid']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) ** 2)

        # 가중치 매개변수 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0],
                                                             filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t):
        # forward propagation
        self.loss(x, t)

        # back propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        layes = reversed(self.layers.values())
        for layer in layers:
            dout = layer.backward(dout)
        
        # 기울기 저장
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
```

이 SimpleConvNet을 MNIST 데이터셋으로 학습하면 훈련 데이터에 대한 정확도는 99.82%, 시험 데이터에 대한 정확도는 98.96%가 나옴

<br>

> # 7.6 CNN 시각화하기
---

>> ## 7.6.1 1번째 층의 가중치 시각화하기
---

1번째 층 합성곱 계층의 가중치 형상 : (30, 1, 5, 5)

$ \rightarrow $ 30개의 5 x 5 회색조 이미지로 시각화할 수 있음

<br>

![1층 시각화](/assets/images/2021_08_16/7_6_1_1.PNG)

무작위 회색조 이미지에서 학습을 거치며 규칙성을 가짐

에지 (색상이 바뀐 경계선)와 블롭 (국소적으로 덩어리진 영역) 등에 영향을 받음

<br>

![에지에 반응하는 필터](/assets/images/2021_08_16/7_6_1_2.PNG)

필터 1은 세로 에지에 반응해, 세로 방향으로 색상 경계가 생긴 모자 끝부분과 어깨 부분에 더욱 민감하게 반응함

|||||
|:-:|:-:|:-:|:-:|
|10|10|10|10|
|10| 1| 1| 1|
|10| 1| 1| 1|
|10| 1| 1| 1|

|||
|:-:|:-:|
|10|10|
| 1| 1|

예로 들어, 위와 같은 4 x 4 입력 데이터와 2 x 2 필터의 합성곱을 계산해보면 (stride = 1)

||||
|:-:|:-:|:-:|
|211|202|202|
|121| 22| 22|
|121| 22| 22|

가로 에지에 반응하는 필터였기 때문에 가로 방향의 경향성에 더욱 민감한 것을 알 수 있음

<br>

>> ## 7.6.2 층 깊이에 따른 추출 정보 변화
---

계층이 깊어질수록 추출되는 정보 (강하게 반응하는 뉴런)는 더욱 추상화됨

<br>

![8층 CNN](/assets/images/2021_08_16/7_6_2.PNG)

일반 사물 인식 CNN (AlexNet)

층이 깊어지면서 뉴런이 반응하는 대상이 단순한 모양에서 고급 정보로 변화함

<br>

> # 7.7 대표적인 CNN
---

>> ## 7.7.1 leNet
---

1998년 제안된 손글씨 숫자 인식 네트워크

<br>

![LeNet](/assets/images/2021_08_16/7_7_1.PNG)

* LeNet은 시그모이드 함수를 사용 / 현재 CNN은 주로 ReLU를 사용

* LeNet은 서브샘플링을 하여 중간 데이터의 크기를 줄임 / 현재 CNN은 최대 풀링이 주류

<br>

>> ## 7.7.2 AlexNet
---

2012년 발표된 모델로 LRN이라는 국소적 정규화를 실시하는 계층을 이용하고 드롭아웃을 사용함

네트워크 구성 면에서는 큰 차이가 없지만, 병렬 계산에 특화된 GPU의 보급과 빅데이터의 접근성 완화가 딥러닝의 발전을 가져옴

<br>

> # 7.8 정리
---

* CNN은 지금까지의 완전연결 계층 네트워크에 합성곱 계층과 풀링 계층을 새로 추가함

* 합성곱 계층과 풀링 계층은 im2col 함수를 이용하면 간단하고 효율적으로 구현할 수 있음

* CNN을 시각화해보면 계층이 깊어질수록 고급 정보가 추출됨

* 대표적인 CNN에는 LeNet과 AlexNet이 있음

* 딥러닝의 발전에는 빅데이터와 GPU가 크게 기여함