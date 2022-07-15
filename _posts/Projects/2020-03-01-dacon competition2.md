---
layout: post
title: (Dacon) 천체 분류 알고리즘 실험 공유
subtitle: 천체 분류 알고리즘
# bigimg: /img/path.jpg
category: Projects
tags: [deeplearning, classification]
---

"Multi perceptron layers 으로 문제를 해결해보고 싶었다."

## 요약

- 내가 시도한 내용 

0. 데이터 학습시, normalization 실험 
1. 클래스 불균형 제거를 위한 오버 샘플링 
2. 아키텍처의 변형 
3. 카테고리 범주의 임베딩

dataframe 형태의 데이터에 대해서 어떤 아키텍처를 사용해야할까 고민 => 기본 neural network 사용 
=> 목적함수는 cross entropy 사용 (어떤 클래스라고 표현되는 인덱스와의 차이를 학습) 
=> torch.nn.CrossEntropy() : logsoftmax가 자동으로 구현되어 있어서 고려하여 사용함. 

### Data

천체의 특성을 정의하는 특성값들이 21가지가 존재하고, 특성에 따라서 19가지의 유형으로 (개념) 구분을 한다.
특성들은 어떤 밝기 등에 대해서 연속적인 수치로 표현이 되어있다.
천체라는 것은 사람이 만든 범주이므로 multi classification으로 접근을 했고 19가지의 유형으로 분류하기 위한 모델을 구성해보기로 했다.
보통 이런 multi classification 문제의 경우 Cross Entropy loss를 사용하는데, pytorch document에 가보면 LogSoftmx + NLLLoss 라고 설명되어 있다. 즉, 최종적으로 C 클래스로 분류한 모델의 ouput을 Logsoftmax를 취하고 negative log likelihood를 구한다.

> NLLLoss의 경우 C 클래스에 대한 target의 인덱스를 가지고 결과 값과 log likeli hood를 구한다. 인덱스는 분명 [0,0,1] (class 3개인 경우) 과 같은 인코딩 방식으로 변환되어 표현될 것이라고 생각한다. 자세한 것은 torch의 구현 과정을 살펴볼 필요가 있어보인다. 

다만, torch에는 기본적으로 log softmax가 내장되어 있기 때문에 따로 비교하지 않지만 softmax를 사용하지 않고 굳이 logsoftmax를 사용하는지 궁금증이 생겼다.

일반적으로 분류 문제를 학습시킬 때, softmax를 사용하는 경우가 많다. 모델을 통과한 값이 k개의 클래스 중 하나로 분류 되어야 한다면, 나온 수치값들을  p로 사용한다. 일반 소프트 맥스를 사용하는 경우와 log softmax를 사용하는 경우 실제 클래스와 다른 결과가 나올 때, 로그를 씌운 경우 실제 클래스와 확률이 더 차이가 날 수록 오차를 크게 만들어 구분을 용이하게 한다고 해석할 수 있다.

0. 

- 실험 중 든, 또 다른 궁금증. feature 의 범위는 1번 분류 feature를 제외하면 15 ~ 25 범위 안에 존재하는데 이를 standardization 하는 것이 나은가?
모델의 아키텍처 구성에 따라 달라지긴 하는데, 실험을 개인적으로 진행했을 때 standardization을 진행했을 시 처음부터 loss가 작은 값에서 시작하는 것을 확인할 수 있었다. 하지만 학습을 진행할 수록 더 낮은 loss 값이 나오는지는 확인이 어려웠다.

- 오히려 input 전에 feature normalization을 따로 진행하고 Batchnorm 및 Layernorm 을 진행했더니 loss가 0.6 이하로 떨어지지 않는 현상이 발생했다. (평소 0.39까지는 수렴됨)

* 막간을 이용한... standardization vs normalization 
매번 헷갈리는 개념이 있는데 위의 두 가지이다.
선형대수에서 Normalization 의 개념은 백터를 자기의 length로 나누는 것이라고 정의하고 있다.
여기는 data값을 [0,1] 사이로 만드는 것이다. 모든 매개변수가 동일한 양의 스케일을 가지는 경우 유용할 수 있다. 그렇지만 데이터 특이치는 손실됨.
Standardization는 data를 rescale하는 것이다. mean은 0으로, standard deviation은 1로 (unit variation) 만드는 것이다.

1.

```python

# class 별 분포, class 불균형
train.groupby('type', axis=0).count()['id']

type
GALAXY                 37347
QSO                    49680
REDDEN_STD             14618
ROSAT_D                 6580
SERENDIPITY_BLUE       21760
SERENDIPITY_DISTANT     4654
SERENDIPITY_FIRST       7132
SERENDIPITY_MANUAL        61
SERENDIPITY_RED         2562
SKY                      127
SPECTROPHOTO_STD       14630
STAR_BHB               13500
STAR_BROWN_DWARF         500
STAR_CARBON             3257
STAR_CATY_VAR           6506
STAR_PN                   13
STAR_RED_DWARF         13750
STAR_SUB_DWARF          1154
STAR_WHITE_DWARF        2160

==============================
======= over sampling ========
==============================
type
GALAXY                 37347
QSO                    49680
REDDEN_STD             14618
ROSAT_D                 6580
SERENDIPITY_BLUE       21760
SERENDIPITY_DISTANT     4654
SERENDIPITY_FIRST       7132
SERENDIPITY_MANUAL      3965
SERENDIPITY_RED         2562
SKY                     2159
SPECTROPHOTO_STD       14630
STAR_BHB               13500
STAR_BROWN_DWARF        1500
STAR_CARBON             3257
STAR_CATY_VAR           6506
STAR_PN                 1677
STAR_RED_DWARF         13750
STAR_SUB_DWARF          1154
STAR_WHITE_DWARF        2160

```
=> 위의 클래스 수를 보면 특정 클래스가 상대적으로 매우 적음을 확인 => 학습 정확도에 영향을 미칠 것이라 생각 => 오버 샘플링 (같은 데이터를 복사하는 형태로 1000개 이상으로 만들어주었다.)

=> 결과 : 학습시 0.34 까지 0.02정도 val loss가 떨어지나 test 시에는 별다른 성능 향상을 보지 못했다. 

2. 

기본적인 mlp 아키텍처를 변형할 수 있을까? 
이 생각은 기존에 이미지로 여겨지는 2-D array 데이터에 대해서 Convolution  filter weights를 구성한 네트워크에서 차용해서 1-D array의 Ax+b뿐만 아니라 Convolution 1D를 사용하면 어떨까 시작되었다.
feature input : [a,b,c,d,e....]를 attention 블록 형태로 구성해서 계산하면 다음 블록은 그 가중치를 추가적으로 넘겨서 layer를 통과할 것이라 생각
기본적인 아키텍처는 spatialNL 네트워크를 참고하여 변형([https://github.com/facebookresearch/video-nonlocal-net])

=> val loss가 1에서 떨어지지 않음. train loss는 떨어지나 학습이 잘 되는지 확인할 수가 없음.

``` python
class SpatialNL(nn.Module):
    """Spatial NL block for classification
       revised to 1d
    """

    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale
        super(SpatialNL, self).__init__()
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1,
                           stride=1, bias=False)
        self.bn = nn.BatchNorm1d(inplanes)
        self.l1 = nn.Conv1d(inplanes, 7000, kernel_size=1,
                           stride=1, bias=False)
        self.l2 = nn.Conv1d(7000, 4000, kernel_size=1,
                           stride=1, bias=False)
        self.l3 = nn.Conv1d(4000, 1000, kernel_size=1,
                           stride=1, bias=False)
        self.l4 = nn.Conv1d(1000, 19, kernel_size=1,
                           stride=1, bias=False)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, d = t.size()
        
        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)
        att = torch.bmm(t, p)
        
        if self.use_scale:
            att = att.div(c**0.5)
        att = self.softmax(att)
        
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        
        x = x.view(b, c, d)
        x = self.z(x)
        x = self.bn(x) + residual
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        return x

```

3.

- 카테고리 임베딩
딥러닝으로 학습을 하려다보니 범주 값을 수치로 생각하고 그냥 넣었을 때 별다른 차이가 없었다.
범주 값이 수치적으로 표현되어 입력값으로 들어가면 결과가 달라지지 않을까 고민했다.
다만 수치적으로 표현될 때, 천체를 분류한다는 목적에 어느정도 부합하려면 그에 맞게 목적함수를 두고 학습해야하는데 
원래 학습 중인 cross entropy를 두고 임베딩 값도 같이 학습시키면 되려나 하는 부분을 시도했으나 코드로는 
배치마다 불러올 때, 임베딩 값으로만 바꿔서 수치형으로 들어가 학습시키게 하는 형태로만 코드를 구성했다.
결과적으로 큰 차이는 없었다...

input 중에 fiberID 라는 범주형 값을(0-1000) 10 dimension의 벡터로 바꿔줌.


``` python
class Embedding(nn.Module):
    def __init__(self, n):
        super().__init__()
        """ Embedding model for entity 
            Args : n(int) output shape
        """
        self.n = n
        self.embed = nn.Embedding(1001,n)
        
    def forward(self, x):
        outputs = self.embed(x)
        #outputs = outputs.reshape(self.n,)
        return outputs
    
class PandasDataset(Dataset):
    def __init__(self, data_path):
        super(PandasDataset, self).__init__()
        self.values = pd.read_csv(data_path)
        self.X, self.Y = self.values.iloc[:,3:], self.values.iloc[:,1] # type
        self.tmp_x , self.tmp_y = self.X.values, self.Y.values
        
        self.fiber = self.get_fiber(torch.LongTensor((self.values.iloc[:,2].values, ))) 
        self.tmp_x = np.append(self.fiber.squeeze(0).detach().numpy(), self.tmp_x, axis=1) # float
    
        #self.tmp_x = self.normalize(self.tmp_x)
    
    def get_fiber(self, x):
        embed_model = Embedding(10)
        fiber = embed_model(x)
        return fiber
    
    def normalize(self,tmp_data):
        normalized_data = (tmp_data - tmp_data.mean(axis=1,keepdims=True)) / tmp_data.std(axis=1, keepdims=True)
        return normalized_data
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return {
            'X':torch.from_numpy(self.tmp_x)[idx],  # torch.FloatTensor() : 연산시 gpu를 제대로 사용 못함.
            'Y':torch.from_numpy(self.tmp_y)[idx]
        }

```


## feature enginerring 

- 이 대회는 주어진 데이터에서 컬럼을 늘리는 등 상위 참가자들이 피처 가공하는 과정을 거치는 것을 다수 확인할 수 있었다. 이와 관련하여 보고 내용을 정리해봤다. 되돌아보면 사실 처음 대회를 개최한 데이터도 SDSS에서 가공하여 만든 것이므로 SDSS를 참고하여 관계가 있는 데이터를 분석하고 늘리는 것도 좋은 방법이라고 생각한다.

SDSS data 
이하 데이터는 https://www.sdss.org/dr16/가 제공한 자료를 https://dacon.io/에서 가공했으며,
CC BY 4.0 라이센스를 적용 받습니다. 
(위의 설명을 보면 SDSS에서 천체 데이터를 데이콘에서 분류문제로 알고리즘을 만들 수 있도록 가공한 것으로 보인다.)

- psfMag : Point spread function magnitudes : 먼 천체를 한 점으로 가정하여 측정한 빛의 밝기입니다.
- fiberMag : Fiber magnitudes : 3인치 지름의 광섬유를 사용하여 광스펙트럼을 측정합니다. 광섬유를 통과하는 빛의 밝기입니다.
- petroMag : Petrosian Magnitudes : 은하처럼 뚜렷한 표면이 없는 천체에서는 빛의 밝기를 측정하기 어렵습니다. 천체의 위치와 거리에 상관없이 빛의 밝기를 비교하기 위한 수치입니다.
- modelMag : Model magnitudes : 천체 중심으로부터 특정 거리의 밝기입니다.
- fiberID : 관측에 사용된 광섬유의 구분자

train : 73 MB (199991 ,24) 
test : 3 MB (10009, 22)

공유 후기를 확인하면서 새로 배울 수 있었던 부분은 관련이 있어 보이는 feaure를 모두 만들고 관련이 없는 것들을 제거해나가는 형태로 피처를 다시 구성한다는 점이었다. (아래 자료들은 dacon의 수상자 공유 자료를 참고했음을 밝힙니다.)

차후 생각해보니.. SDSS에서 천체를 분류할 때, 특히 psfMag과 같은 빛의 밝기와 밝기의 측정에 영향을 미칠 수 있는 petroMag, modelMag 는 서로의 연산을 통해서 추가적인 정보를 만들 수 있을 것 같다. 

어쨋든, 추가적으로 클래스를 분류하는데 도움되는 정보를 만들 수 있는 것에 초점을 맞춰서 '처음해봐요'님의 후기를 보면 diff도 max별 diff, sum / min별 diff, sum을 각각 다 구분해서 구했음을 볼 수 있다.


Row별, Magnitude별 max, min, max-min, std, sum을 구한다.

https://www.sdss.org/dr12/algorithms/magnitudes/#asinh
위의 사이트를 참고해서 기본 column으로 추가 column을 만들 수 있다.
```python
color_list = ['u', 'g', 'r', 'i', 'z']
b_list = [1.4*10e-10, 0.9*10e-10, 1.2*10e-10, 1.8*10e-10, 7.4*10e-10]
f0_list = [24.63, 25.11, 24.80, 24.36, 22.83]
for c, b, f0 in zip(color_list, b_list, f0_list):
    all_data[f'psfMag_{c}_asinh'] = -2.5*np.log(10)*(np.arcsinh((all_data[f'psfMag_{c}']/f0)/(2*b))+np.log(b))
```
=> 찾아보니 이런 자료들이 있는거보면.. 역시 공들여서 조사하고 실험하고 생각하는게 중요해보인다..


- 계산 전에 Null or NaN 처리를 진행했음을 확인할 수 있다. 
<pre>
train = all_data.loc[all_data['type'].notnull()]
test = all_data.loc[all_data['type'].isnull()].reset_index(drop=True)
</pre>

사실 추가로 정리하게된 목적이 'permutation importance'라는 것때문이었는데, 이렇게 변수를 구상해서 많이 추가하다보면 학습을 방해하는 변수를 추가할 수 도 있는 셈이다. 이런 부분을 어느정도 확인하고 통제할 부분이 필요해 보인다. 

permutation importance는 feature importance를 확인을 목적으로 model이 data에 fit된 후 측정한다.
모델을 학습시키고 데이터X, y, seed, iteration 한 값을 넣으면 변수가 중요한 순으로 반환해준다.
기본 아이디어는 결과를 기준으로 변수를 random shuffle하게 되는 경우 모델 성능에 영향을 많이 미치는 순을 확인할 수 있다는 것이다.
weight의 경우 감소된 정확도를 의미한다. +- 값의 경우는 one-reshuffling to the next 다시 reshuffle 시 변화되는 변동 폭을 의미하는 듯 보인다.

<pre>
Weight	Feature
0.1750 ± 0.0848	Goal Scored
0.0500 ± 0.0637	Distance Covered (Kms)
0.0437 ± 0.0637	Yellow Card
0.0187 ± 0.0500	Off-Target
0.0187 ± 0.0637	Free Kicks
0.0187 ± 0.0637	Fouls Committed
0.0125 ± 0.0637	Pass Accuracy %
0.0125 ± 0.0306	Blocked
0.0063 ± 0.0612	Saves
0.0063 ± 0.0250	Ball Possession %
0 ± 0.0000	Red
0 ± 0.0000	Yellow & Red
0.0000 ± 0.0559	On-Target
-0.0063 ± 0.0729	Offsides
-0.0063 ± 0.0919	Corners
-0.0063 ± 0.0250	Goals in PSO
-0.0187 ± 0.0306	Attempts
-0.0500 ± 0.0637	Passes
</pre>

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

```
참고 : https://www.kaggle.com/dansbecker/permutation-importance

=> PermutationImportance는 예측이나 분류 모델에 대해서 변수 해석 자료정도로 참고할 수 있을 것으로 보인다.
더불어 실제 실험시 설계를 디테일하게 할 필요가 보인다.

결론적으로 추가된 모든 변수들에 대해 학습하고(학습시 하나씩 확인하는 것이 정확하나.. 그렇다면 Permutation Importance를 쓰는 의미가 희석될 것으로 보인다.) validation dataset에서 점검할 때, 실험해보고 싶은 변수들을 섞어보고 score가 많이 깎이지 않으면 일정 cutoff를 주어 제거하는 식으로 적용해볼 수 있을 것으로 보인다.

- 참고 : https://dacon.io/competitions/official/235573/codeshare/693 



