---
layout: post
title: [논문 리뷰] What Make Good In-Context Examples for GPT-3?
subtitle: In-context learning 이해하기.
categories: Posts
---

<!-- ![_config.yml]({{ site.baseurl }}./images/logo.png) -->

# What Make Good In-Context Examples for GPT-3?

"In-context learning series : 논문 리뷰 "

### 1. 서론

GPT-3은 NLP 연구에서 새로운 획을 그었다.
이전에는, NLP모델들이 pre-trained 그리고 나서 특정 태스크에 대해 fine-tuned를 하였는데,
GPT-3은 다른 모델들과 다르게 "in-context" learning ability 로 차별화된다.

약간의 in-context example 들만 있으면 GPT-3는 파인 튜닝 없이도 학습하지 않은 (unseen) 경우들에 대해서 일반화 (generalize) 를 할 수 있다.
이와 같은 능력은 사람이 가지는 독특함애 대한 가능성으로 많은 기술적 가능성을 열어주었다. (This opens up many new technological possibilities that ~) 미래의 NLP 시스템들은 이메일로 이를 확장하고 코드를 만들어내고 텍스트에서 엔터티를 추출해 낼 것이다.

이러한 강력하고 (powerful) 다재다능한 (versatile*) 능력에도 GPT-3에게는 실용적인 문제점들을(practical challenges) 가지고 있다.
논문에서는 training set에서 task-relevant한 examples을 샘플링하여 이용하지만, GPT-3의 경우 in-context examples의 선택마다 그 성능이 요동치는 (fluctuate) 경향이 있다.

&rarr; 저자들은 SST-2 데이터세트를 이용해서,  Training set에서 뽑은 in-context example들이 Test accuracy 성능의 분산 차이가 있는 것을 Table 1에서 같이 제시한다.

그리고나서,
Our work aims to carefully examine this issues to gain a deeper understanding on how to better select in-context examples to improve GPT-3's performance.
라고 명확하게 목적을 밝힌다.

GPT-3 모델에 training set 전체를 이용하여 파인튜닝하는 방법이 있겠지만, 현재의 GPT-3는 그 용량이 175B에 달하여 로드 및 계산 비용이 일반 리서치 랩에서 진행하기엔 한계가 있다고 주장한다.

&rarr; 저자들은 위와 같은 이야기들을 함께 적는 것이, GPT-3를 파인튜닝하여 접근하지 않는 이유에 대한 근거로 보인다.


### 2. 방법론

![]({{site.url}}/images/WMGICEG/fig1.png)
