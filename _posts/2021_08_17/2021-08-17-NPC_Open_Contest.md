---
title: "중앙대학교 NPC Open Contest"
excerpt: "2021.08.16 22:30 ~ 23:59"
categories: [Problem Solving]
tags: [BOJ, Contest]
last_modified_at: 2021-08-17 14:43:00 +0900
---

간만에 약간의 여유도 생기고, contest 자체도 90분짜리이기도 해서 참여해보았다.

<br>

![스코어보드](/assets/images/2021_08_17/scoreboard.PNG)

결과는 살짝 아쉬운 6등

사실 위에 분들을 이길 수 있을 거라 생각하진 않기 때문에 별로 아쉽진 않다.

맨날 3~4시간짜리 contest만 참여하다가 90분 만에 다 풀려니 은근히 빡셌다.

<br>

![통계](/assets/images/2021_08_17/statistics.PNG)

![맞았습니다](/assets/images/2021_08_17/correct.PNG)

A번을 풀고 바로 E번으로 넘어가서 E, F를 제일 먼저 풀었다 ㅎㅎ

<br>

> # A. [<u>이진 딸기</u>](https://www.acmicpc.net/problem/22935)
---

티어 : Silver V

술게임 이진딸기를 알고 있었기 때문에 바로 문제를 풀 수 있었다.

1, 2, ..., 14, 15, 14, ..., 2가 한 사이클을 이루므로 mod 28에 대해 결과를 매핑해주면 된다.

결과 도출에는 bin(), zfill(), replace()를 이용했다.

```python
import sys

for _ in range(int(sys.stdin.readline())):
    n = (int(sys.stdin.readline()) - 1) % 28
    if n <= 14:
        n += 1
    else:
        n = 29 - n
    print(bin(n)[2:].zfill(4).replace('0', 'V').replace('1', '딸기'))
```

<br>

> # B. [<u>주간 달력</u>](https://www.acmicpc.net/problem/22936)
---

티어 : Gold IV

A를 풀고 B로 넘어왔는데 문제가 즉각 이해되지 않아 E번으로 넘어갔었다.

결국 풀지 못했는데, 차근차근 읽어보니 브루트포스로 풀 수 있었다.

>> 시작 날짜를 정하면 N개의 주력을 이어붙인다.  
>> M개의 일정 중에 만들어 놓은 주력과 기간이 겹치는 게 있으면 테이프를 잘라 붙인다.  
>> 주력에 붙어 있는 테이프 면적이 제일 클 때, 테이프를 자르는 횟수는?

범위가 다 작아서 모든 경우를 계산해보면 된다.

처음에는 max_area가 갱신될 때마다 테이프를 끊은 횟수를 계산했더니 TLE가 떴다.

그도 그럴 것이 그렇게 구현하면 시간 복잡도가 대략 O(50000NM)이 나온다.

그래서 첫 번째 for문을 전부 돌리고 테이프를 끊은 횟수를 한 번만 계산해 줬다.

```python
import sys

n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
schedule = [tuple(map(int, sys.stdin.readline().split())) for _ in range(m)]
max_area = 0
max_count = 0
max_start = 0

for start in range(1, 50002 - 7 * n):
    end = start + 7 * n - 1
    cur_area = 0
    count = 0

    for x, y in schedule:
        if start <= x:
            if min(y, end) - x + 1 > 0:
                cur_area += min(y, end) - x + 1
                count += 1
        elif y <= end:
            if y - max(x, start) + 1 > 0:
                cur_area += y - max(x, start) + 1
                count += 1
        else:
            cur_area += end - start + 1
            count += 1

    if cur_area > max_area:
        max_area = cur_area
        max_count = count
        max_start = start

for k in range(1, n):
    cut = max_start + 7 * k - 1
    for x, y in schedule:
        if x <= cut < y:
            max_count += 1

print(max_count)
```

<br>

> # C. [<u>교수님 계산기가 고장났어요!</u>](https://www.acmicpc.net/problem/22937)
---

티어 : Silver II

contest 도중에는 없었는데 끝나고 나니 언어 제한이 생겼다.

사실 파이썬으로 푼 풀이가 왜 틀린지 모르겠어서 파이썬에 대한 불신만 늘어갔다.

파이썬으로 풀 때 아이디어는 ~~조금 양아치 같지만~~ float로 읽어온 후 10 ** 9를 곱해 정수 범위에서 곱셈을 해주고, 적당히 다시 소수로 만드는 것이었다.

Decimal 라이브러리를 떠올렸으면 아주 쉽게 풀 수 있었을 텐데 아쉽다.

C++로는 큰 수 곱셈을 구현해서 풀었다.

```cpp
#include <iostream>

using namespace std;

void solve() {
    char A[10], B[10];
    char z;

    int i = 0, flag = 0, loop = 10;
    while (loop--) {
        cin >> z;
        if (z == '-') {
            flag++;
            loop++;
            continue;
        }
        if (z == '.') {
            loop++;
            continue;
        }
        A[i] = z;
        i++;
    }

    i = 0;
    loop = 10;
    while (loop--) {
        cin >> z;
        if (z == '-') {
            flag++;
            loop++;
            continue;
        }
        if (z == '.') {
            loop++;
            continue;
        }
        B[i] = z;
        i++;
    }

    int ans[20] = {0, };
    for (i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            ans[i + j + 1] += ((int)A[i] - 48) * ((int)B[j] - 48);
        }
    }
    int q = 0;
    for (i = 19; i >= 0; i--) {
        ans[i] += q;
        q = ans[i] / 10;
        ans[i] %= 10;
    }
    if (flag == 1) cout << '-';
    if (ans[0]) cout << ans[0];
    cout << ans[1] << '.';

    for (i = 2; i < 20; i++) {
        cout << ans[i];
    }
    cout << '\n';
}

int main() {
    cin.tie(NULL); cin.tie(NULL);
    ios::sync_with_stdio(false);
    
    int n;
    cin >> n;
    while (n--) solve();

    return 0;
}
```

<br>

> # D. [<u>백발백준하는 명사수</u>](https://www.acmicpc.net/problem/22938)
---

티어 : Bronze II

백발백중이이 아니라 백발백**준**인거 지금 봤다...

문제를 슥 스캔하니 두 원이 겹치냐 안 겹치냐 묻는 문제였다.

우변에 제곱 안 씌웠다가 한 번 틀렸다...

```python
import sys

a, b, c = map(int, sys.stdin.readline().split())
d, e, f = map(int, sys.stdin.readline().split())

print(['NO', 'YES'][(a - d) ** 2 + (b - e) ** 2 < (c + f) ** 2])
```

<br>

> # E. [<u>쿠키크루</u>](https://www.acmicpc.net/problem/22939)
---

티어 : Gold V

A번을 풀고 B번을 건너 뛴 뒤 무엇을 풀까 고민하다가 문제 제목이 마음에 들어서 골랐다.

문제가 살짝 길지만 요약하자면, H에서 출발해 #에 도착할 때 W, C, B, J 중에서 적어도 한 종류를 모두 (3군데) 방문해야 한다.

경우의 수가 4 x 3!에 N도 100 이하라 브루트포스로 풀었다.

itertools.permutations()를 이용할 수도 있었지만, 그냥 복붙해서 다 쓰는 게 더 빠를 것 같았다.

```python
import sys


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


n = int(sys.stdin.readline())
board = [list(sys.stdin.readline().strip()) for _ in range(n)]

assassin = []
healer = []
mage = []
tanker = []

for i in range(n):
    for j in range(n):
        if board[i][j] == 'J':
            assassin.append((i, j))
        elif board[i][j] == 'C':
            healer.append((i, j))
        elif board[i][j] == 'B':
            mage.append((i, j))
        elif board[i][j] == 'W':
            tanker.append((i, j))
        elif board[i][j] == 'H':
            x, y = i, j
        elif board[i][j] == '#':
            w, z = i, j

res = float('inf')

for name, job in zip(['Assassin', 'Healer', 'Mage', 'Tanker'], [assassin, healer, mage, tanker]):
    d = float('inf')
    d = min(d, dist((x, y), job[0]) + dist(job[0], job[1]) + dist(job[1], job[2]) + dist(job[2], (w, z)))
    d = min(d, dist((x, y), job[0]) + dist(job[0], job[2]) + dist(job[2], job[1]) + dist(job[1], (w, z)))
    d = min(d, dist((x, y), job[1]) + dist(job[1], job[0]) + dist(job[0], job[2]) + dist(job[2], (w, z)))
    d = min(d, dist((x, y), job[1]) + dist(job[1], job[2]) + dist(job[2], job[0]) + dist(job[0], (w, z)))
    d = min(d, dist((x, y), job[2]) + dist(job[2], job[0]) + dist(job[0], job[1]) + dist(job[1], (w, z)))
    d = min(d, dist((x, y), job[2]) + dist(job[2], job[1]) + dist(job[1], job[0]) + dist(job[0], (w, z)))
    if res > d:
        res = d
        ans = name

print(ans)
```

<br>

> # F. [<u>선형 연립 방정식</u>](https://www.acmicpc.net/problem/22940)
---

티어 : Gold II

브루트포스를 하기엔 10 ** 12가지 경우라서 TLE가 뜰 것이다.

그럼 뭐 별 수 있겠는가, Gaussian Elimination을 사용해야 한다.

처음에 pypy3로 제출했다가 RE가 떠서 뭐지 싶었는데, 예전에 pypy3를 사용할 때 math.lcm 때문에 ModuleNotFoundError가 떴던 기억이 떠올라 python으로 제출했고, 통과됐다.

구글링을 해보니 math.lcm이 python 3.9 버전에 새로 추가된 함수라고 한다.

pypy3엔 아직 추가되지 않았나 보다.

```python
import sys
from math import lcm

n = int(sys.stdin.readline())
equation = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]

for z in range(n - 1):
    l = lcm(*[equation[i][z] for i in range(n - z)])

    for i in range(n - z):
        k = l // equation[i][z]
        for j in range(z, n + 1):
            equation[i][j] *= k

    for i in range(n - z - 1):
        for j in range(z, n + 1):
            equation[i][j] -= equation[- z - 1][j]

res = []
for z in range(n):
    x = equation[z][-1] // equation[z][- z - 2]
    res.append(x)

    for i in range(z + 1, n):
        equation[i][-1] -= x * equation[i][- z - 2]

print(*reversed(res))
```

<br>

> # G. [<u>RPG 마스터 오명진</u>](https://www.acmicpc.net/problem/22941)
---

티어 : Silver I

숫자 범위도 크고 시간도 짧으므로 O(1) 풀이를 고안해야 한다.

전투 시작 후 몇 턴까지 용사가 죽지 않고, 마왕도 스킬 발동을 하지 않을까를 생각해 보면 쉽게 풀 수 있다.

```python
import sys

x, y, z, w = map(int, sys.stdin.readline().split())
p, s = map(int, sys.stdin.readline().split())

t = min((x - 1) // w, (z - p - 1) // y)

x -= t * w
z -= t * y

z -= y
if z <= 0:
    print('Victory!')
    exit()

x -= w
if x <= 0:
    print('gg')
    exit()

if 1 <= z <= p:
    z += s

t = min((x - 1) // w, (z - 1) // y)

x -= t * w
z -= t * y

z -= y
if z <= 0:
    print('Victory!')
else:
    print('gg')
```