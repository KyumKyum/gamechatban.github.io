---
layout: home
author_profile: true
title: "Title Here"
excerpt_separator: "<!--more-->"
categories:
  - Post Formats
tags:
  - Post Formats
  - readability
  - standard
---

## Deep Learning Project - Chatting Toxicity Analysis and Possible Report Reason Regression

#### 비지도학습/지도학습 기반 게임 채팅 toxic level 측정 및 가능한 신고 사유 예측

> 생명과학과 2018023427 이승현
> 정보시스템학과 2019014266 임규민
> 정치외교학과 2022094366 장정원

**I. Proposal**

1. Motivation
2. What do you want to see at the end

**II. Datasets**

**III. Methodology**

1. Unspervised Learning
2. spervised Learning

**IV. Evaluation & Analysis**

**V. Related Work (e.g., existing studies)**

**VI. Conclusion: Discussion**

# I. Proposal

**1. Motivation**
팀 게임을 하면 채팅 기능이 존재한다. 팀원들끼리 채팅을 통해 협력하고 소통하기 위함이다. 적들과 대화하는 것 또한 가능하다. 그러나 상대방을 실제로 마주하고 있지 않은 온라인의 특성 상 상대방을 불쾌하게 만드는 대화가 자주 등장한다.
[플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계]를 보면 트롤링이 일어나는 원인 중 '익명성', '시스템적 규제의 허술', '실시간', '다중성'으로 본다는 것을 알 수 있다. 또한 트롤링의 원인을 개인의 특성에서 둔 연구를 보면 트롤링이 재미, 지루함, 복수와 같은 심리적인 요인에서 나오는 것이라고 분석하였다.
우리 조는 리그오브레전드(League of Legends)에서 report라 불리는 채팅 신고 내역을 분석하며 이러한 채팅을 단속하고 향후 보다 깨끗한 인터넷 채팅문화를 만들고자 이 프로젝트를 진행하였다.

**2. What do you want to see at the end**
우리 조는 리그오브레전드 채팅 신고 datasets에 TF-IDF를 사용하여 어떤 단어 또는 문장이 신고를 많이 받았는지를 분석하여 Toxic Level이라는 새로운 변수를 만들것이다. 이후 이를 데이터에 넣어 이를 기계학습시켜볼 것이다. 학습된 장치에 어떠한 채팅을 입력하고 이 채팅의 Toxic Level이 몇인지를 맞추게 한 다음, 이 수준이면 정지를 받을지 아닐지도 파악하게 할 것이다. 또한 정지의 이유가 무엇인지까지 나오게 하는 것이 우리의 목표이다.

# II. Datasets

# III. Methodology

**1. Unspervised Learning**

**2. spervised Learning**

# IV. Evaluation & Analysis

# V. Related Work (e.g., existing studies)

이준명, 나정환, 도영임. (2016). 플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계. 한국게임학회 논문지, 16(1), 63-71, 10.7583/JKGS.2016.16.1.63

#VI. Conclusion: Discussion
