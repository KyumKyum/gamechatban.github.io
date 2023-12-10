# 2023-02 Ai-X Deep Learning Project

###### ㅇ

---

# Contents

**I. Proposal**

1. Motivation
2. Objective

**II. Team Members**

**III. Dataset**

**IV. Project Strategy**

1. Unsupervised Learning

   i) Learning Strategy

   ii) Code Explanation

   iii) Result Evaluation and Analysis

   iv) Full Code

2. Supervised Learning (Regression)

   i) Learning Strategy

   ii) Code Explanation

   iii) Result Evaluation and Analysis

   iv) Full Code

3. Supervised Learning (Classification)

   i) Learning Strategy

   ii) Code Explanation

   iii) Result Evaluation and Analysis

   iv) Full Code

**V. Video Link**

**VI. Conclusion**

**VII. Related Works**

---

# I. Proposal

### **1. Motivation**

팀 게임을 하면 채팅 기능이 존재한다. 팀원들끼리 채팅을 통해 협력하고 소통하기 위함이다. 적들과 대화하는 것 또한 가능하다. 그러나 상대방을 실제로 마주하고 있지 않은 온라인의 특성상 상대방을 불쾌하게 만드는 대화가 자주 등장한다.

![argue](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/2b37b8d5-8ccc-4196-8383-a9d39433ea0e){: width="50%", .align-center}

해당 근거를 뒷받침하는 연구 결과 역시 존재한다. 논문 <<플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계>>[^1] 에서 밝히기를, 트롤링이 일어나는 원인을 '익명성', '시스템적 규제의 허술', '실시간', '다중성'으로 분석하고 있다. 또한 트롤링의 원인을 개인의 특성에서 둔 연구를 보면 트롤링이 재미, 지루함, 복수와 같은 심리적인 요인에서 나오는 것이라고 분석하였다.

[^1]: 이준명, 나정환, 도영임, 플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계, 한국게임학회 논문지,2016,

그렇다면 이러한 게임 속 대화의 문제점을 제재하기 위한 방안은 어떤 것이 있었을까. 먼저 유명 게임 리그오브레전드(League of Legends)에서는 "지속적으로 부정적인 행동을 보이는 계정이 가장 먼저 받는 불이익은 주로 채팅 제한이다. 해당 기능이 활성화되면 게임 내에서 채팅을 할 수 없게 되고, 해당 제한은 특정 일수가 지날 때까지 지속된다.[^2]라는 규정을 가지고 있다. 이외에도 다양한 게임에서 부적절한 채팅을 주고받을 때 처벌을 가하는 조항을 가지고 있다.

[^2]: League of Legends 고객지원 신고 - 제재 및 이용 제한 - 게임 내 이용 제한 알아보기

하지만 정작 어떤 기준에 의해서, 어떤 말을 했을 때 처벌이 이루어지는지는 알려진 바가 거의 없다. 이로 인해 억울하게 처벌 받는 유저나 명백히 부적절한 언행을 한 유저를 신고했음에도 해당 유저에 대한 처벌이 이루어지지 않거나 이루어져도 알 방법이 없는 상황이다. [^3]

[^3]: 조현덕·이정서 기자, "모니터 속 무법지대를 처벌하려면", 중대신문, 2022.12.05, https://news.cauon.net/news/articleView.html?idxno=37564

따라서, 이러한 게임 내 부적절한 채팅을 확인하고 유저 스스로 채팅의 부적절성을 판단하는 것을 도움으로써 향후 보다 건전한 게임 채팅 문화에 기여하고자 이 프로젝트를 진행하였다. 우리 조는 리그오브레전드(League of Legends)라는 전 세계적으로 유행하는 AOS 장르 게임의 채팅 신고 내역을 활용하여 해당 채팅이 타 유저로 하여금 얼마나 불쾌하게 만드는지 비지도학습과 지도학습을 활용하여 정량화하고 이를 면밀히 분석하고자 한다.

### **2. Objective**

해당 딥러닝 프로젝트는 비지도학습과 지도학습을 거쳐 채팅에 대한 전체적인 분석과 기계학습을 통한 toxic score 예측 모델과 신고 사유 분류 모델을 설계할 것이다. 개괄적인 프로세스는 다음과 같다.

**1. 비지도학습 단계 (Unsupervised Learning)**

- 1-1) TF-IDF (Term Frequency - Inverse Document Frequency) 가중치 값을 계산 및 활용하여, 각 채팅 단어의 빈도수를 정규화하여 많이 등장하고 영향력이 높은 단어의 가중치를 계산한다.
- 1-2) 각 계산된 가중치를 바탕으로 각 채팅에 대한 toxic score를 계산한다.
- 1-3) 이후 지도학습에 활용될 column만을 남겨놓는다.

**2. 지도학습 단계 (Supervised Learning)**

A. Toxic Score 예측 모델

- 2-1) LSTM (Long Short-Term Memory) Model을 활용하여 예측 모델을 학습시킨 후, 오차율과 모델의 결과값을 시각화 한다.
- 2-2) LightGBM (Gradient Boosting) Model을 활용하여 예측 모델을 학습시킨 후, 오차율과 모델의 결과값을 시각화 한다.

B. 신고 사유 분류 모델

- 2-3) Random Forest Classifier Model을 활용하여 분류 모델을 학습시킨 후, 오차율과 모델의 결과값을 시각화 한다.

**각 비지도/지도 학습이 끝난 이후, 결과값과 오차율, 시긱화 및 그래프에 대한 분석을 실시할 것이다.**

---

# II. Team Memebers

- 생명과학과 2018023427 이승현: 데이터 전처리 전략, 그래프 및 딥러닝 결과 분석
- 정보시스템학과 2019014266 임규민: 딥러닝 및 특징 공학 전략 수립, 코드 작성
- 정치외교학과 2022094366 장정원: 자료 수집 및 글 작성, 영상 촬영

---

# III. Dataset

리그오브레전드(League of Legends)에서 report당한 채팅들에 관한 데이터셋이다. 약 160만개의 영문 채팅 기록을 가지고 있는 dataset이며, 다음과 같은 항목을 가지고 있다. (출처: Kaggle[^4])

[^4]: https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs

![raw](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/09ec0c53-1ca5-4190-a23f-8b9c0539085b)

- message: 신고당한 메세지 내용

- association_to_offender: 해당 채팅의 소속.

  - Emeny: 적군의 채팅
  - Ally: 아군의 채팅
  - Offender: 신고 당한 유저의 채팅

- time: 신고당한 시간

- case_total_reports: Tribunal이라는 게임 내 시스템으로 넘어가기까지의 신고 횟수

  - Tribunal 시스템은 게임 내에서 플레이어들이 부적절한 행동을 신고하고 심사하는 시스템으로서, 일정 수 이상의 신고를 빋은 채팅은 해당 시스템에 인해서 적절한 조치를 받게 됩니다

- allied_report_count: 규정위반자에게 상대팀의 적이 신고한 횟수

- enemy_report_count: 규정위반자에게 같은팀의 아군이 신고한 횟수

- most_common_report_reason: 5가지의 기본적인 신고 사유.

  - Negative Attitude: 부정적 태도
  - Offensive Language: 공격적 언행
  - Assisting Enemy Team: 적군을 도와주는 행위 (트롤링)
  - Verbal Abuse: 욕설 및 인신 공격
  - Spamming: 광고형 채팅 (스팸)

- chatlog_id: 고유한 chatlog 식별 번호

- champion_name: 채팅을 친 사용자가 플레이한 챔피언 이름

---

# IV. Project Strategy

### **1. Unsupervised Learning**

비지도 학습(Unsupervised Learning)은 기계학습의 주요 분야 중 하나로, 데이터의 내재된 구조와 패턴을 파악하는 데 중점을 둔다. 이러한 학습 방식은 특별한 목표값이나 라벨이 주어지지 않은 상태에서 입력 데이터의 특성을 탐색하고 해석하는데 사용된다. 지도학습과 대조되는 비지도 학습에서는 사전에 정의된 라벨이나 목표값이 필요하지 않으며, 대신 시스템은 데이터셋 내부의 숨겨진 구조나 관계를 발견하는 것에 중점을 둔다. 이러한 특징은 비지도 학습을 탐험적이고 데이터 중심적인 기술로 만들어, 기존의 패턴이 명확하지 않거나 인간이 라벨을 만드는 데 제한적이거나 비용이 많이 드는 상황에서 특히 효과적으로 사용된다.

비지도 학습의 핵심은 라벨이 지정되지 않은 데이터에서 의미 있는 통찰력을 추출하는 것으로, 다양한 도메인에서의 응용 가능성을 열어놓고 있다. Clustering은 두드러진 비지도 학습 기술 중 하나로, 비슷한 데이터 포인트를 기반으로 함께 그룹화하여 데이터 내부의 구조나 자연스러운 그룹을 드러내는 것을 목표로 한다.[^5] 차원 축소 기술은 또 다른 비지도 학습의 측면으로, 데이터의 본질적인 특성을 추출하면서 중복되거나 노이즈가 많은 정보를 제거하는 것을 목표로 한다. 비지도 학습 알고리즘은 계층적 클러스터링부터 오토인코더 및 생성적 적대 신경망까지 다양하며, 머신러닝 전문가들에게는 라벨이 없는 데이터로부터 가치 있는 지식을 얻어내어 주어진 정보의 본질적인 복잡성에 대한 깊은 이해를 촉진한다.

[^5]: Roman, Victor, Unsupervised Machine Learning: Clustering Analysis, Medium, 2019.03.07,https://towardsdatascience.com/unsupervised-machine-learning-clustering-analysis-d40f2b34ae7e

#### i) Learning Strategy

##### Unsupervised Learning Methodology: TF-IDF

TF-IDF는 문서 내 단어마다 중요도를 고려하여 가중치를 주는 통계적인 단어 표현 방법이다. TF는 단어의 빈도를 고려하는 것이고, IDF는 역 문서 빈도를 고려하는 것으로 이 둘의 곱으로 TF-IDF를 구한다.

TF의 계산 방법은 카운트 기반 단어표현 방법인 DTM(Document Term Matrix)과 동일하다. 먼저 각 문서에서 나타난 전체 단어를 알파벳 순으로 배열한다. 이후 각 문서별로 사용된 단어들의 개수를 세서 벡터로 표현한다. 그러나 DTM은 단순하게 문서 데이터에서의 단어의 빈도수만을 고려한다. 때문에 중요한어와 불필요한 단어를 구분하기 어렵다는 한계점을 가지고 있다.

그렇기에 중요한 단어에는 높은 가중치를 부여하고, 덜 중요한 단어에는 낮은 가중치를 부여할 필요가 있다. 이를 해결하기 위한 것이 DF(Document Frequency)이다.

IDF는 Inverse Document Frequency이다. DF 값의 역수라는 뜻이다. DF는 전체 문서에서의 특정 단어가 등장한 문서 개수이다. 많은 문서에 등장할수록 그 단어의 가치는 줄어들게 된다. 그렇기에 역수를 취하면 적은 문서에 등장할수록 IDF의 값이 높아지게 된다. 이 둘의 값을 곱한 TF\*IDF 값이 TF-IDF 값이다.

이렇게 단어 빈도와 문서 빈도를 같이 고려하여 계산하기에 TF-IDF는 뛰어난 성능으로 높은 정확도를 보인다. 또한 용어마다 점수를 메기기에 직관적으로 확인이 가능하며 이후 진행할 toxic score를 계산하기 용이하다는 장점도 가지고 있다. 이러한 이유로 우리 조는 비지도 학습의 방법으로 TF-IDF를 활용하여 부정적인 영향을 끼치는 채팅을 분석해 볼 것이다.

TF-IDF을 활용하여, 다음과 같은 분석 단계를 거쳐 비지도학습의 결과를 얻을 것이다.

- 1-1) TF-IDF (Term Frequency - Inverse Document Frequency) 가중치 값을 계산 및 활용하여, 각 채팅 단어의 빈도수를 정규화하여 많이 등장하고 영향력이 높은 단어의 가중치를 계산한다.
- 1-2) 각 계산된 가중치를 바탕으로 각 채팅에 대한 toxic score를 계산한다.
- 1-3) 이후 지도학습에 활용될 column만을 남겨놓는다.

#### ii) Code Explanation

##### 0. Initialize

**0-1) Package Import**

```R
library(dplyr)
library(tm)
library(wordcloud)
library(ggplot2)
```

먼저 필요한 R package를 불러온다. 필요한 패키지는 다음과 같다.

- dplyr: 데이터 프레임 처리하는 함수군
- tm: tect mining을 줄인 말. 텍스트 전처리를 위한 패키지
- wordcloud: Word Cloud을 활용한 시각화를 위한 패키지
- ggplot2: 비지도학습 결과를 plotting 하기 위한 패키지.

**0-2) Dir Setting and Read Dataset**

```R
setwd("~/your_folder")
chatlogs <- read.csv("./chatlogs.csv")
```

Working Directory를 현재 폴더로 설정하고, 데이터 셋을 불러온다. 이후. 전처리 과정을 시행한다.

##### 1. Feature Engineering

**1-1) Datasets 가지치기**

```R
chatlogs <- chatlogs %>% filter(association_to_offender == 'offender')
chatlogs <- subset(chatlogs, select = -association_to_offender)
```

Datasets의 ‘association_to_offender’ 칼럼 중, “offender”에 해당하는 튜플만 추출한다. 이는 신고당한 사람의 채팅만을 선별하는 과정이다.
추출한 뒤에는 ‘association_to_offender’ 칼럼은 “offender”라는 튜플 외 다른 튜플은 삭제된 의미없는 칼럼이기에 삭제한다.

**1-2) 일부 문법적 표현 및 게임특수성 관련 표현 제거**

```R
champion_names <- read.csv("./champion_names.csv")

pattern <- paste0("\\b(?:is|are|&gt|&lt|was|were|", paste(unique(c(champion_names$Champion, champion_names$Abbreviation)), collapse = "|"), ")\\b")

chatlogs$message <- gsub(pattern, "", chatlogs$message, ignore.case = TRUE)

write.csv(chatlogs, "processed.csv")
```

리그오브레전드라는 게임 특성상 채팅에서 챔피언의 이름을 자주 말하게 된다. 이는 채팅 신고와 상관없는 단어이기에 champion_names.csv 파일을 이용해서 제거한다.
이후 문법적으로 사용되는 영단어인 is, are, &lt, &gt 등도 불필요하기에 제거한다.
처리한 Datasets을 새로 csv 파일로 저장한다.

**1-3) 심각도(severity)에 대한 feature engineering**

```R
chatlogs$severity <- cut(
  chatlogs$case_total_reports,
  breaks = c(-Inf, 3, 6, Inf),
  labels = c("Severe", "Normal", "Low"),
  include.lowest = TRUE
)
```

Datasets의 'case total reports' 값을 기반으로 "severity"를 만든다. severity는 3단계로 나누는데, case total reports가 3 이하인 경우 Severe(심각), 4 이상 6 이하인 경우 Normal(보통), 7 이상인 경우 Low(낮음)이다. 이는 case total reports가 낮은 경우 적은 사용으로도 신고될 만큼 심각성이 크다고 생각했기 때문이다.

**1-4) 문자열 합치기**

```R
concatenated <- chatlogs %>%
  group_by(most_common_report_reason, severity) %>%
  summarise(concatenated_text = paste(message, collapse = " ")) %>%
  ungroup()

write.csv(concatenated, "concat.csv")
```

신고 사유를 바탕으로 chatlog들을 그룹화한다. 그룹화한 이후 문자열 합치기(concatenate)한다. 이렇게 신고 사유별로 분류된 하나의 문자열을 새로운 feature인 ‘심각도(severity)’를 고려하여 작성된다. 이러한 새로운 csv 파일을 저장한다.

이러한 전처리 과정 이후 TF-IDF matrix 분석을 한다. 이는 앞에서 합쳐진 문자열에 대해 TF-IDF를 처리하여 각 단어의 'toxic level'을 얻는 과정이다.

##### 2. TF-IDF

**2-1) TF-IDF를 위한 말뭉치(corpus)를 생성하고 추가 전처리를 진행**

```R
corpus <- Corpus(VectorSource(concatenated$concatenated_text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
```

말뭉치(corpus)는 자연어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합을 뜻한다. 우리 조는 TF-IDF를 처리하기 위하여 말뭉치를 생성한다. 전처리 과정을 거친 텍스트에서 구두점, 여백, 관사 등을 제거하고 모든 문자열을 소문자로 변환한다.

**2-2) DTM 생성 & TF-IDF matrix 생성**

```R
dtm <- DocumentTermMatrix(corpus)

tf_idf <- weightTfIdf(dtm)
tf_idf <- t(as.matrix(tf_idf))
```

TF-IDF를 위해 DTM(Document Term Matrix)를 생성한다. DTM은 Unsupervised Learning 처음에 소개했던 대로 문서 단어 행렬이다. DTM을 바탕으로 TF-IDF matrix를 생성한다.

**2-3) matrix 순서 바꾸기**

```R
tf_idf_col_name <- paste(concatenated$most_common_report_reason, concatenated$severity, sep = "_")
colnames(tf_idf) <- tf_idf_col_name
```

현재 matrix 순서대로면 메세지들이 한 행에 연달아 나오는 문제가 발생한다. 이를 해결하기 위하여 새로운 열(column)을 만들고 각 행마다 메세지들을 넣게 matrix의 순서를 바꾼다. 이제 신고 사유와 심각도를 분석하기 쉬워졌다.

**2-4) 값 보정**

```R
tf_idf <- round((tf_idf * 1000), 2)
```

toxic level을 얻기 위해 값을 보정하고 반올림한다.

**2-5) 새로운 데이터 프레임으로 변환**

```R
tf_idf_df <- as.data.frame(tf_idf)

write.csv(tf_idf_df, "toxic_lev.csv")
```

완료된 TF-IDF matrix를 새로운 데이터 프레임으로 변환하고 'toxicity_lev.csv'라는 CSV 파일로 내보낸다.

**2-6) toxic score 정의 & 계산**

```R
chatlogs$toxic_score <- 0

for(i in 1:nrow(chatlogs)) {
  tlv <- 0 # Toxic Level for current chatlog
  message <- chatlogs$message[i]
  terms <- unlist(strsplit(message, " ")) # Split the message into terms.
  terms <- trimws(terms) # Trim term
  for (term in terms) {
    found <- term_scores$total_tfidf[term_scores$term == term] # Find such terms in term scores
    head(found)
    if(!identical(found,numeric(0))){ # if such term exists,
      tlv <- tlv + found # Add the term score.
    }
  }
  # Apply Weight based on severity.
  weight <- ifelse(chatlogs$severity[i] == "Normal", 0.6,
                   ifelse(chatlogs$severity[i] == "Low", 0.3, 1))

  chatlogs$toxic_score[i] <- round((tlv * weight), 2)
}

res_log <- chatlogs[, c("X", "message", "most_common_report_reason", "toxic_score")]
write.csv(res_log, "offender_chatlog_with_toxic_score.csv")
```

Offender의 채팅 toxic level을 toxic score라는 새로운 칼럼을 정의하여 재구성한다. 이후 채팅 로그의 메시지를 추출한 뒤 각 단어의 앞뒤 공백을 제거하고 해당 단어의 TF-IDF 가중치를 가져와 채팅 로그의 toxic score를 계산하고 해당 열에 값을 저장한다.

##### 3. Visualization

```R
# 1. Word Cloud -> Shows visualized output of tf-idf
wordcloud(words = term_scores$term, freq = term_scores$total_tfidf, min.freq = 1, scale=c(3,0.5), colors=brewer.pal(8, "Dark2"))

# 2. Scatter plot -> Shows the relationship between severity and number of reports, and resulting toxic score
ggplot(chatlogs, aes(x = case_total_reports, y = toxic_score, color = severity)) +
  geom_point() +
  labs(title = "Scatter Plot of Case Reports vs Toxic Scores",
       x = "Number of Case Reports",
       y = "Toxic Score",
       color = "Severity") +
  theme_minimal()
```

- Term의 가중치가 높을수록 더욱 글씨가 커지는 word cloud을 통해 시각화를 진행했다.
- '신고 횟수는 심각도에 반비례 한다'는 전처리 과정 중의 가정을 시각화를 통해 입증한다.

#### iii) Result Evaluation and Analysis

##### 1. Resulting Dataframe

1. 각각의 열은 각각 순번, 데이터셋에 적혀있던 순번, 채팅 내용, 신고 사유, toxic score 순서대로 정렬되어 있다.
   <img width="658" alt="tf_idf_df" src="https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/920ab174-c1bb-46ab-a1bd-305df5fef27e">

2. 최저점은 0으로 실제 신고 사유와 무관한 채팅 속 단어들은 대부분 0점이다.
   <img width="557" alt="tf_idf_low" src="https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/3aba0e8c-0ba9-42af-a1c3-0113efed3dfb">

3. 최고점은 98.52이고, 채팅의 길이가 길수록 toxic score가 높게 분포하는 경향을 가지고 있다.
   <img width="658" alt="tf_idf_high" src="https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/87111630-63d9-4585-8570-6e51219918fc">

4. 비슷한 toxic score 임에도 채팅 길이의 차이가 있는 경우는 주로 적나라한 욕설이 있을수록 채팅이 짧아도 toxic score가 높게 측정되었다.

##### 2. Result Analysis

**A. Scatter Plot: Case of Report and Severity**
![TF_IDF  Scatter Plot](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/4ba98d24-b777-44d7-9cec-1fa05b4a8c06)

- 위 그림은 TF-IDF 모델의 toxic score와 해당 유저의 신고 당한 횟수를 나타낸 것이다. 그래프는 toxic score이 높은 채팅일수록 적은 신고 횟수를 나타낸다.

- 이는 수위가 높은 채팅일수록 더 적은 횟수의 신고만으로도 처벌이 이루어졌음을 의미하고, 동시에 toxic score을 도출하는 과정이 정확히 이루어졌음을 시사한다.

**B. Word Cloud**
![TF_IDF  Word Cloud](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/0225d1c5-cd98-4dad-9d2b-42a34fc33b4b)

- 위 그림은 TF-IDF 결과를 토대로 word cloud를 생성한 결과이다. 해당 word cloud는 특정 단어의 가중치가 높아질수록 단어의 크기가 커지는 형태인데, 상대적으로 일상에서 많이 쓰이는 단어가 상대적으로 큰 크기를 나타내는 것을 볼 수 있다.

- 해당 시각화 결과물에서 주목해야 할 부분은 눈에 잘 보이는 단어들이 아닌 오히려 작아서 잘 보이지 않는 단어들이다. 앞선 scatter plot의 결과와 마찬가지로 전체 말뭉치(corpus) 중 특정 단어가 많이 언급된다는 의미는 반복된 표현임에도 toxic level이 낮아 실제 신고로 이어진 빈도가 낮다는 것이다.

#### iv) Full Code

```R
# AI-X Final Project
# Unsupervised Learning based on TF-IDF

# Dataset: League of Legends Tribunal Chatlogs (Kaggle)
# https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs

library(dplyr) # R Package: dplyr - advanced filtering and selection
library(tm) # R Package: tm - Text Mining/Merging for preprocess of TF-IDF, and TF-IDF itself
library(wordcloud) # R Package: Used for visulaization of TF-IDF Result
library(ggplot2) # R Package: Used for visualization of relationship between #case reports and toxic scores.

setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

chatlogs <- read.csv("./chatlogs.csv")

# Pre-Processing Steps; Feature Engineering Pipeline for the chatlogs.
# 1. Pruning
  # Select only tuples where association_to_offender = offender.
# 2. Grammatical & Game-specific Expression Removal:
  # Remove common grammatical expressions like "is" and "are" to enhance the validity of the analysis.
# 3. Feature Engineering - Severity:
  # Introduce a new feature called 'severity' based on the total number of case reports.
  # Total case report <= 3: Severe
  # Total Case Report >= 4 && <= 6: Normal
  # Total Case Report >= 7: Low
# 4. Concatenation of Chatlogs:
  # Group chatlogs based on the common reported reason.
  # Concatenate chatlogs within each group into a single text.
  # Chatlogs will be merged into a single column for each most common reported reason, considering the newly defined 'severity' feature.

# Step 1: Pruning: Select only offender's chatlog.
chatlogs <- chatlogs %>% filter(association_to_offender == 'offender')
# Remove not-required column (association_to_offender: All columns will be offender)
chatlogs <- subset(chatlogs, select = -association_to_offender)

# Step 2: Gramatical Game-specific Expression Removal: Used gsub and REGEX to do such task.
# Read champion names
champion_names <- read.csv("./champion_names.csv")
# Create a regex pattern for both grammatical expressions and champion names/abbreviations
pattern <- paste0("\\b(?:is|are|&gt|&lt|was|were|", paste(unique(c(champion_names$Champion, champion_names$Abbreviation)), collapse = "|"), ")\\b")

# Remove both grammatical expressions and champion names/abbreviations from chatlogs$message
chatlogs$message <- gsub(pattern, "", chatlogs$message, ignore.case = TRUE)

# Export into csv for later use. (Pre-processed.csv)
write.csv(chatlogs, "processed.csv")

# Step 3:  Feature Engineering - Severity
chatlogs$severity <- cut( # Categorize numbers into factors.
  chatlogs$case_total_reports,
  breaks = c(-Inf, 3, 6, Inf),
  labels = c("Severe", "Normal", "Low"),
  include.lowest = TRUE
)

# Step 4: Concatenation of Chatlogs
concatenated <- chatlogs %>%
  group_by(most_common_report_reason, severity) %>% #Group by following two category
  summarise(concatenated_text = paste(message, collapse = " ")) %>%
  ungroup()

write.csv(concatenated, "concat.csv")

# TF-IDF (Term-Frequency Inverse Document Frequency) Matrix Anaylsis; Process TF-IDF for each concatenated text to get 'toxiticy level of each words'.
# 1. Create a corpus for TF-IDF, pre-process it.
# 2. Create DTM for TF-IDF, and generate TF-IDF matrix
# 3. Transpose it and apply new column name to analyse the reported reason and severity.
# 4. Scale Up and round the value to get toxic level.
# 5. Export into csv. 'toxicity_lev.csv'

# Create a Corpus from the column concatenated_text.
corpus <- Corpus(VectorSource(concatenated$concatenated_text))

# Additional pre-process of the text in the corpus. (e.g. removing punctuation, stripping whitespaces, etc.)
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert each contents into lower case
corpus <- tm_map(corpus, removePunctuation) # Remove Punctuations
corpus <- tm_map(corpus, removeWords, stopwords("english")) # Remove Additional English stopwords (a, the, etc) that hadn't been filtered.
corpus <- tm_map(corpus, stripWhitespace) # Strip Whitespace

# Create DTM (Document-Term Matrix) based on the corpus, which is used for TF-IDF
dtm <- DocumentTermMatrix(corpus)

# Create TF-IDF Matrix based on the DTM.
tf_idf <- weightTfIdf(dtm)
tf_idf <- t(as.matrix(tf_idf)) # Transpose

# Generate Column name
tf_idf_col_name <- paste(concatenated$most_common_report_reason, concatenated$severity, sep = "_")

# Set the column name of the transposed TF_IDF
colnames(tf_idf) <- tf_idf_col_name

# Scale up and Round the values
tf_idf <- round((tf_idf * 1000), 2)

# Convert TF-IDF matrix into a new data frame for further analysis.
tf_idf_df <- as.data.frame(tf_idf)

# Make it into csv for further analysis & supervised learning.
write.csv(tf_idf_df, "toxic_lev.csv")

# Extract terms and their total TF-IDF scores
term_scores <- data.frame(term = rownames(tf_idf_df), total_tfidf = rowSums(tf_idf_df))

# Assess Toxic level of each offender's chatlog
# Append a new column 'toxic_level' is chatlog.
chatlogs$toxic_score <- 0

for(i in 1:nrow(chatlogs)) {
  tlv <- 0 # Toxic Level for current chatlog
  message <- chatlogs$message[i]
  terms <- unlist(strsplit(message, " ")) # Split the message into terms.
  terms <- trimws(terms) # Trim term
  for (term in terms) {
    found <- term_scores$total_tfidf[term_scores$term == term] # Find such terms in term scores
    head(found)
    if(!identical(found,numeric(0))){ # if such term exists,
      tlv <- tlv + found # Add the term score.
    }
  }
  # Apply Weight based on severity.
  weight <- ifelse(chatlogs$severity[i] == "Normal", 0.6,
                   ifelse(chatlogs$severity[i] == "Low", 0.3, 1))

  chatlogs$toxic_score[i] <- round((tlv * weight), 2)
}


res_log <- chatlogs[, c("message", "most_common_report_reason", "toxic_score")]
write.csv(res_log, "offender_chatlog_with_toxic_score.csv")


#Visualization

# 1. Word Cloud -> Shows visualized output of tf-idf
wordcloud(words = term_scores$term, freq = term_scores$total_tfidf, min.freq = 1, scale=c(3,0.5), colors=brewer.pal(8, "Dark2"))

# 2. Scatter plot -> Shows the relationship between severity and number of reports, and resulting toxic score
ggplot(chatlogs, aes(x = case_total_reports, y = toxic_score, color = severity)) +
  geom_point() +
  labs(title = "Scatter Plot of Case Reports vs Toxic Scores",
       x = "Number of Case Reports",
       y = "Toxic Score",
       color = "Severity") +
  theme_minimal()
```

---

### **2. Supervised Learning (Regression)**

지도 학습(Supervised Learning)은 기계학습의 중요한 카테고리 중 하나로, 예시를 통해 학습하는 구조를 가지고 있다. 이 방식은 비지도 학습과는 대조적으로 학습을 위한 입력과 그에 대응하는 목표값(라벨)이 함께 제공된다. 이것은 관여자 또는 인간이 이미 문제에 대한 정답을 알고 있는 상황에서 사용된다. 인공지능 시스템은 주어진 입력 데이터를 기반으로 예측을 수행하고, 실제 목표값과 비교하여 오차를 최소화하도록 훈련된다. 이는 시스템이 주어진 작업에서 최적의 예측을 수행할 수 있도록 학습하는 데에 중점을 둔다. [^6]

[^6]: Jorge Leonel, Supervised Learning, Medium, 2018.06.03, https://medium.com/@jorgesleonel/supervised-learning-c16823b00c13

지도 학습은 다양한 응용 분야에서 활용되며, 특히 분류(Classification)와 회귀(Regression) 작업에서 효과적으로 활용된다. 분류 작업에서는 입력 데이터를 미리 정의된 클래스 중 하나로 할당하고, 회귀 작업에서는 연속적인 값을 예측하는 데 사용된다. 예를 들어, 이메일이 스팸인지 아닌지를 분류하거나 주택 가격을 예측하는 등 다양한 예측 작업에 지도 학습이 적용된다. [^7]

[^7]: Curtis Savage, What is Supervised Learning?, Medium, 2022.12.17, https://medium.com/ai-for-product-people/what-is-supervised-learning-fa8e2276893e

먼저, 분류 작업과 회귀 작업 중 회귀 작업을 우선으로 진행할 것이다. 지도학습 중 회귀를 통해 우리 조의 목적이었던 toxic level을 예측할 수 있도록 할 것이다. 모델은 LSTM과 light GBM을 사용한다. 이 두 가지 모델을 이용해서 기존의 데이터셋을 학습시키고, 메시지의 toxic level을 예측할 수 있도록 할 것이다.

#### i) Learning Strategy

##### Supervised Learning Methodology: LSTM

LSTM이란 Long Short-Term Memory Network의 약자로, RNN(Recurrent Neural Network)의 일종이다.[^8] 또한 RNN은 전통적인 전통적인 neural network에서 일어나는 문제를 해결하고자 만든 모델이다. 그렇기에 LSTM을 설명하기 위해선 우선 neural network와 RNN에 대한 설명이 선행되어야 할 것이다.

[^8]: 타리그 라시드(2017). 신경망 첫걸음. 한빛미디어.

우선 neural network에 대한 설명이다. 전통적 컴퓨터는 동물의 뇌와 다른 구조적 차이를 가지고 있다. 그렇기에 동물의 뇌가 아주 단순히 처리할 수 있는 내용도 쉽게 처리 하기 어려웠다. 이를 해결하기 위한 방법이 생물의 신경망과 비슷하게 인공신경망을 만들어 낸 것이다. 생물학적 뉴런의 동작원리와 같이 여러 입력을 받고 이 입력값들이 일정 수준을 넘어서면 시그모이드 함수라는 활성화 함수를 활용하여 출력값을 내보내게 되는 것이다.

그러나 전통적인 neural network에서는 생각을 지속적으로 하지 못한다는 단점이 있다. 이전에 일어나는 사건을 바탕으로 나중에 일어나는 사건을 생각할 수 없다는 것이다. 이러한 문제를 해결하기 위하여 만든 모델이 RNN이다. RNN또한 neural network이기에 입력값을 받아 출력값을 내놓는다. 그러나 스스로 반복하는 과정을 추가하여 이전 단계에서 얻은 정보가 지속되도록 하는것이다. 이렇게 RNN의 체인처럼 이어지는 성질은 음성 인식, 언어 모델링, 번역, 이미지 주석 생성 등 다양한 분야에서 데이터를 다루기 최적화된 구조의 neural network인 것이다.

LSTM은 이러한 RNN의 특별한 한 종류이다. 기존의 RNN은 현재 시점의 무언가를 얻기 위하여 최근의 정보만을 필요로 하는, 즉 필요한 정보를 얻기 휘한 시간 격차가 크지 않을 때는 문제가 일어나지 않는다. 그러나 더 많은 문맥을 필요로 하여 시간 격차가 커지는 경우에 RNN은 학습하는 정보를 계속 이어나가기 어려워한다는 문제가 있다. 이에 LSTM은 명시적으로 설계되어 긴 의존 기간의 문제를 피할 수 있도록 하였다. [^9]

[^9]: 한땀컴비, "딥러닝 기본 네트워크 - LSTM", 한땀한땀 딥려닝 컴퓨터 비전 백과사전, https://wikidocs.net/152773

이처럼 neural network와 RNN의 문제들을 보완하기 위해 만들어진 LSTM이기에 우리 조는 이러한 LSTM을 사용해 예측 모델을 학습시킨 후, 오차율과 모델의 결과값을 시각화 할 것이다.

##### Supervised Learning Metodology: light GBM

Gradient Boosting은 “Gradient”라는 개념을 이용하여 이전 모델의 약점을 보완하는 새로운 모델을 순차적으로 만든 뒤 이를 선형 결합하여 얻어낸 모델을 생성하는 지도 학습 알고리즘이다. [^10]

[^10]: kkiyou, "[머신러닝] LightGBM", Velog, https://velog.io/@kkiyou/lgbm

이때 Gradient”는 residual fitting과 일맥상통한 개념으로 실제로 residual fitting은 예측값의 residual(잔차)를 줄여나가며 정확도를 높여가는 방식인데 gradient 역시 일종의 residual로 Gradient Boosting에서는 negative gradient를 이용하여 다음 모델을 순차적으로 만들어 나간다.

만약이전 모델이 실제값을 정확하게 예측하지 못하는 약점을 가지게 되는 상황이라면 Gradient boosting은 실제값과 예측값의 차이를 줄여주는 함수를 찾는다. 이러한 함수들을 기존 함수에 선형적으로 더하여 예측값의 오차를 줄이는 방식이 바로 Gradient Boosting이다.

이 중 우리 모델에 사용된 Light GBM은 XGBoost에 비해 훈련 시간이 짧고 성능도 좋아 부스팅 알고리즘에서 가장 많은 주목을 받고 있는 알고리즘으로 Gradient Boosting을 발전시킨 것이 XGBoost, 여기서 속도를 더 높인 것이 LightGBM이다.

LightGBM은 트리 기준 분할이 아닌 리프 기준 분할 방식을 사용한다. 트리의 균형을 맞추지 않고 최대 손실 값을 갖는 리프 노드를 지속적으로 분할하면서 깊고 비대칭적인 트리를 생성한다. 이렇게 하면 트리 기준 분할 방식에 비해 예측 오류 손실을 최소화할 수 있다. [^11]

[^11]: LightGBM. (n.d.). LightGBM Documentation. Retrieved from https://lightgbm.readthedocs.io/en/stable/

##### Strategy

이번 지도학습의 목표는 <Regression 작업> 메시지의 toxic level을 예측할 수 있는 모델을 구축하는 것이다. 전략은 다음과 같다.

1. Feature engineering: 메시지와 toxic score를 추출하여 학습에 필요한 feature로 사용한다. 이후 데이터를 3:1의 비율로 학습 데이터와 테스트 데이터로 분할한다. 즉 75%와 25%인 것이다.

2. LSTM 모델을 사용한 regression 모델 구축: regression 수행하고 모델을 평가한다. 이후 모델의 결과를 시각화한다.

3. LightGBM을 사용한 regression 모델 구축: LSTM과 마찬가지로 regression 수행하고 모델을 평가한다. 이후 모델의 결과를 시각화한다.

4. 두 모델의 결과를 분석 및 비교한다.

#### ii) Code Explanation

##### 0. Initialize

**0-1) Package Import**

```R
library(keras)
library(tensorflow)
library(reticulate)
library(lightgbm)
```

먼저 필요한 R package를 불러온다. 필요한 패키지는 다음과 같다.

- keras: LSTM Model을 위한 패키지
- tensorflow: LSTM의 학습을 위한 패키지 (Python Package)
- reticulate: Python과 interfacing을 위한 패키지
- lightgbm: LightGBM Model을 위한 패키지

**0-2) Dir Setting and Read Dataset**

```R
setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

reticulate::py_config() # Show python interpreter currently configured to use.
# Read a dataset for supervised learning.
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')
```

Working Directory를 현재 폴더로 설정하고, 데이터 셋을 불러온다. 이후. 전처리 과정을 시행한다.

- (해당 regression task는 Python과의 interfacing이 필요하기에, `reticulate`을 활용하여 interface가 잘 되었는지 python configuration을 출력함으로서 확인한다.)

##### 1. Feature Engineering

**1-1) 값 정리**

```R
processed_df <- na.omit(processed_df)
processed_df <- processed_df[processed_df$toxic_score != 0, ]
```

빈 값이 있거나 Toxic score가 0인 행을 제거한다. 이로 인해 약 30만개였던 행의 수가 약 20만개로 줄어 데이터 처리가 수월해진다.

**1-2) LSTM을 위한 텍스트 전처리**

```R
tokenizer <- text_tokenizer()

fit_text_tokenizer(tokenizer, processed_df$message)
```

LSTM을 위한 텍스트 전처리가 선행되어야한다.
우선 텍스트 tokenizer를 생성한다. 이것이 텍스트 데이터를 신경망에 공급할 수 있는 형식으로 변환해줄 것이다.
그 이후 'processed_df' 데이터프레임의 메시지에 대해 텍스트 tokenizer를 학습한다.

**1-3) 정수 시퀀스 변환**

```R
sequences <- texts_to_sequences(tokenizer, processed_df$message)

X <- pad_sequences(sequences, maxlen = 50L)

set.seed(sample(100:1000,1,replace=F))
sample_index <- sample(1:nrow(processed_df), 0.8 * nrow(processed_df))
train_data <- X[sample_index, ]
test_data <- X[-sample_index, ]
train_labels <- processed_df$toxic_score[sample_index]
test_labels <- processed_df$toxic_score[-sample_index]
```

피팅된 tokenizer를 사용하여 텍스트 메시지를 정수 시퀀스로 변환합니다. 이후 시퀀스를 균일한 길이로 패딩한다. 이때 최대 길이는 50 token으로 정하였는데, 이는 텍스트 메세지의 최대 길이가 48이었기 때문이다.
전략에서 말했듯이 데이터를 학습데이터와 테스트 데이터로 랜덤하게 분할한다. 비율은 학습데이터가 75%, 테스트 데이터가 25%이다.

##### 2.LSTM 모델을 사용한 regression 모델 구축

**2-1) LSTM 모델 만들기**

```R
model <- keras_model_sequential()
```

시퀀셜 모델을 만든다.
이후 정수 시퀀스를 밀집 벡터로 변환하는 임베딩 레이어를 추가한다. 각 용어들의 설명은 다음과 같다.

- 1. 'input_dim'은 어휘 크기 (토크나이저의 출력)이다.
- 2. 'output_dim'은 밀집 임베딩의 차원이다.
- 3. 'input_length'는 입력 시퀀스의 길이로, 50 토큰으로 패딩됐다.

**2-2) 메세지 고유 단어 추출**

```R
unique_words <- unique(unlist(strsplit(tolower(processed_df$message), " ")))

vocabulary_size <- length(unique_words)
embedding_dim <- round(sqrt(vocabulary_size))
```

각 메시지에서 고유한 단어를 추출한다. 어휘 크기는 input dim 값이고, outd의 크기는 어휘 크기의 제곱근으로 설정하였다.

**2-3) 임베딩 레이어 추가**

```R
model %>%
  layer_embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = 50L) %>%
  layer_lstm(units = 100) %>%
  layer_dense(units = 1)
```

계산된 어휘 크기와 임베딩 크기를 기반으로 임베딩 레이어 추가한다.

**2-4) 모델 컴파일**

```R
model %>% compile(
  optimizer = 'adam', # Adam Optimizer
  loss = 'mean_squared_error',  # Mean Squared Error loss for regression
  metrics = c('mean_absolute_error') # Mean Absolute Error as an additional metric
)
```

Adam optimizer, regression을 위한 평균 제곱 오차 손실, 추가적인 메트릭으로 평균 절대 오차 값을 활용하여 모델을 컴파일한다.

**2-5) 모델학습**

```R
lstm_start_time <- Sys.time()

history <- model %>% fit(
  train_data, train_labels,
  epochs = 10, batch_size = 32,
  validation_split = 0.2
)

lstm_end_time <- Sys.time()

lstm_elapsed_time <- lstm_end_time - lstm_start_time
cat("Training Time (LSTM): ", lstm_elapsed_time, "\n")
```

모델을 학습시키고 이 과정에서 학습 시작 시간과 종료 시간을 체크한다. 이후 경과 시간을 평가한다.

**2-6) 모델 평가 및 시각화**

```R
model %>% evaluate(test_data, test_labels)

plot(history$metrics$loss, type = "l", col = "blue", xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1:1)
```

모델을 평가하고 training loss와 validation loss를 시각화한다.

##### 3. LightGBM을 사용한 regression 모델 구축

**3-1) 학습 준비**

```R
lgb_data <- lgb.Dataset(train_data, label = train_labels)

lgb_params <- list(
  objective = "regression",  # Use "regression" for regression tasks
  metric = "rmse",  # Root Mean Squared Error as the evaluation metric
  num_iterations = 100 # Number of iteration are going to apply.
)
```

Rgression을 위한 LightGBM 모델을 만든다. 학습 데이터에 LightGBM 데이터셋을 넣고, regression 목표 및 평가 메트릭을 포함하여 LightGBM 매개변수를 설정한다.

**3-2) 모델학습**

```R
lgb_start_time <- Sys.time()

lgb_model <- lgb.train(params = lgb_params, data = lgb_data, verbose = 1)

lgb_end_time <- Sys.time()

lgb_elapsed_time <- lgb_end_time - lgb_start_time
cat("Training Time (LSTM): ", lgb_elapsed_time, "\n")
```

LSTM 때와 마찬가지로 모델을 학습시키고 이 과정에서 학습 시작 시간과 종료 시간을 체크한다. 이후 경과 시간을 평가한다.

**3-3) 테스트 셋 예측과 모델 평가**

```R
predictions <- predict(lgb_model, test_data)

mse <- mean((predictions - test_labels)^2)
rmse <- sqrt(mse)
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
```

테스트 세트를 예상한다. 그리고 RMSE (Root Mean Squared Error)를 계산한다.

##### 4. Visualization

```R
residuals <- predictions - test_labels

qqnorm(residuals)

qqline(residuals, col = "red")
```

오차가 계산되고 Q-Q plot을 통해 시각화된다.

#### iii) Result Evaluation and Analysis

**A. LSTM: Training and Validation Loss Graph**
![LSTM_Graphical Result](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/977e0eb3-00ed-40a7-a3ea-c6cc0d2cd839)

- Epoch별 training의 결과를 나타낸 그래프이다.
  - 상단의 그래프는 loss (훈련 손실값), val_loss (validation loss; 예측 손실값)을 각 Epoch 별로 나타낸 것이다.
  - 하단의 그래프는 MAE (훈련 평균 손실 절대값), validation MAE (예측 평균 손실 절대값)을 각 Epoch 별로 나타낸 것이다.
- Epoch가 지날수록, 훈련의 손실값은 줄어들고, 예측 손실값은 늘어나는 것이 확인이 된다.

![LSTM-Training and Validation Loss](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/3e4e990e-4511-4955-8d69-b49bb5d2c72a)

- 위 그림은 지도학습에 사용된 LSTM 모델의 training loss와 validation loss 값을 따로 나타낸 그래프이다. 우리 모델의 training loss는 epoch가 증가함에 따라 감소하지만, validation loss는 전반적으로 증가하는 양상을 보인다.

- 이러한 결과가 도출된 원인은 모델이 training dataset과 validation dataset의 차이가 크게 존재해 epoch 초기부터 overfitting 되었을 가능성이 있다. 이러한 문제를 해결하기 위해서는 추가로 전처리를 하거나, 낮은 epoch 값부터 집중적으로 확인하여 generalization을 추가로 진행함으로써 validation loss 값을 최소화할 수 있을 것으로 예상된다.

- 그러나 동시에, overfitting이 발생하였다는 것은 모델의 수용능력이 데이터의 복잡한 관계를 학습하기에 충분하다고도 해석할 수 있고, overfitting 여부를 확인할 수 있었기 때문에 이를 방지하는 방향으로 모델을 개선해 나갈 수 있다.

**B. LightGBM: Q-Q Plot**
![LightGBM: QQPlot](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/57d32da5-6ed6-4c59-ba9e-e3788e62bb4a)

- 위 그림은 Light GBM 모델의 Q-Q plot이다. 해당 plot을 보면 기준선을 기준으로 양쪽, 위 아래로 전반적으로 대칭을 이루고 있고, 중심부에 점들이 집중적으로 분표한다. 하지만 양끝 값으로 갈수록 기준선으로부터 멀어지고 꼬리 값이 더 두꺼운 분포를 보인다.

<p align="center">
<img width="516" alt="q-q" src="https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/6bef0e79-1ee2-49ff-9d0b-77ca88b5a7fb">
</p>

- 위 그림[^12] 속 plot 중 우리 모델에 가장 가까운 것은 **heavy-tailed**이다. 이는 Light GBM 모델을 통해 평가했을 때 Theoretical Quantile이 Sample Quantile보다 작다는 것을 의미한다.
- 이를 정리하면, 모델의 분포에 비정규성이 존재하며 이는 regression 모델에 오차가 존재할 수 있음을 나타낸다.

[^12]: yuns_u, QQ Plot 해석하기, Velog, https://velog.io/@yuns_u/QQ-plot-%ED%95%B4%EC%84%9D%ED%95%98%EA%B8%B0

#### iv) Full Code

```R
# AI-X Final Project
# Supervised Learning after doing unsupervised learning.
# Part 1: Regressing Toxic Score of Chatting

# Dataset: A chatlog with toxic score assessed by unsupervised learning.
# Column: X(id), Message, Most common report reason, Toxic Score

library(keras) # R Package: LSTM Model
library(tensorflow) # R Package: Tensorflow
library(reticulate) # R Package for interfacing with python
library(lightgbm) # R PAckage: LightGBM

setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

reticulate::py_config() # Show python interpreter currently configured to use.

# Read a dataset for supervised learning.
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')

# Objective: [Regression Task]: Build a model that can predict the toxic level of message.
# Strategy
# 1. Feature Engineering
  # Extract message and toxic score, which are the features required for learning.
  # Split data into training data and test data (75% : 25%)
# 2. Build a regression model using LSTM Model
  # Make regression and evaluate the model.
  # Plot the model result.
# 3. Build a regression model using LightGBM.
  # Make regression and evaluate the model.
  # Plot the model result.
# 4. Analyze & Compare the result of two models,

# 1-1 Feature Engineering
# Remove the missing values
processed_df <- na.omit(processed_df)
# Remove rows where toxic_score is equal to 0
processed_df <- processed_df[processed_df$toxic_score != 0, ]

# Preprocess the text data for LSTM
# Create a text tokenizer; convert text data into a format that can be fed into a neural network.
tokenizer <- text_tokenizer()

# Fit the text tokenizer on the messages in the 'processed_df' dataframe
# Learns the vocabulary of the corpus and assigns a unique integer index to each word.
fit_text_tokenizer(tokenizer, processed_df$message)

# Convert the text messages to sequences of integers using the fitted tokenizer
# Each word in the messages is replaced with its corresponding integer index.
sequences <- texts_to_sequences(tokenizer, processed_df$message)

# Pad the sequences to ensure uniform length (Maximum length of 50 tokens)
X <- pad_sequences(sequences, maxlen = 50L)  # The maximum length of message is 48.

# Split the data into training and testing sets
set.seed(sample(100:1000,1,replace=F)) # Random sampling
sample_index <- sample(1:nrow(processed_df), 0.8 * nrow(processed_df))
train_data <- X[sample_index, ]
test_data <- X[-sample_index, ]
train_labels <- processed_df$toxic_score[sample_index]
test_labels <- processed_df$toxic_score[-sample_index]

# Build the LSTM model
# Create a sequential model
model <- keras_model_sequential()

# Add an embedding layer to convert integer sequences to dense vectors
  # 'input_dim' is the size of the vocabulary (output of the tokenizer)
  # 'output_dim' is the dimension of the dense embedding
  # 'input_length' is the length of the input sequences (padded to 50 tokens)

unique_words <- unique(unlist(strsplit(tolower(processed_df$message), " "))) # Unique words of each message.

vocabulary_size <- length(unique_words) # Size of vocab. Will be a value of input_dim.
embedding_dim <- round(sqrt(vocabulary_size)) # Size of output, Will be a root(sqrt) of vocab size,

# Adding embedding layer based on the calcualted vocab size and embedding size.
model %>%
  layer_embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = 50L) %>%
  layer_lstm(units = 100) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  optimizer = 'adam', # Adam Optimizer
  loss = 'mean_squared_error',  # Mean Squared Error loss for regression
  metrics = c('mean_absolute_error') # Mean Absolute Error as an additional metric
)

# Train the model
# Check starting time
lstm_start_time <- Sys.time()

# Start Training
history <- model %>% fit(
  train_data, train_labels,
  epochs = 10, batch_size = 32,
  validation_split = 0.2
)

# Check ending time
lstm_end_time <- Sys.time()

# Calcualte Elapsed time;
lstm_elapsed_time <- lstm_end_time - lstm_start_time
cat("Training Time (LSTM): ", lstm_elapsed_time, "\n")

# Evaluate the model
model %>% evaluate(test_data, test_labels)

# Visualization: Line Plot of Training and Validation Loss
plot(history$metrics$loss, type = "l", col = "blue", xlab = "Epoch", ylab = "Loss", main = "Training and Validation Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1:1)

# 1-3: Build a regression model using LightGBM (Gradient Boosting Model)

# LightGBM Model for Regression
lgb_data <- lgb.Dataset(train_data, label = train_labels)

# Set LightGBM parameters
lgb_params <- list(
  objective = "regression",  # Use "regression" for regression tasks
  metric = "rmse",  # Root Mean Squared Error as the evaluation metric
  num_iterations = 100 # Number of iteration are going to apply.
)

# Train the model
# Check starting time
lgb_start_time <- Sys.time()

# Start Training
lgb_model <- lgb.train(params = lgb_params, data = lgb_data, verbose = 1)

# Check ending time
lgb_end_time <- Sys.time()

# Calcualte Elapsed time;
lgb_elapsed_time <- lgb_end_time - lgb_start_time
cat("Training Time (LSTM): ", lgb_elapsed_time, "\n")

# Make predictions on the test set
predictions <- predict(lgb_model, test_data)

# Evaluate the model (RMSE)
mse <- mean((predictions - test_labels)^2)
rmse <- sqrt(mse)
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

## Visualization
# Calculate residuals
residuals <- predictions - test_labels

# Q-Q (Quantile-Quantile) Plot of Residuals
# Q-Q plots are used to assess whether a set of data follows a particular theoretical distribution, such as a normal distribution.
# In this context, the plot compares the quantiles of the residuals against the quantiles of a standard normal distribution. (norm)
# If the points on the plot closely follow the reference line, it suggests that the residuals are approximately normally distributed.
# Deviations from the line may indicate non-normality.(Potential issues exists with the assumptions of the regression model)
qqnorm(residuals)
#Add a straight line, which passes first and third quatiles of the data, for a reference line to the Q-Q plot.
qqline(residuals, col = "red")
```

### **3. Supervised Learning (Classification)**

이제 남은 분류 작업을 진행할 차례이다. 이번에는 toxic level을 예측하는 것 말고도 이 메시지가 어떤 'most_common_report_reason'에 속하는지 예측하는 것도 우리 조의 목적이었다. 이를 위해 지도학습 중 분류를 진행할 것이다. 모델은 Random Forest를 사용한다. 이 모델을 이용하여 기존의 데이터셋을 학습시키고, 메시지가 'most_common_report_reason' 중 어디에 속하는지를 예측할 수 있도록 할 것이다.

#### i) Learning Strategy

##### Random Forest Classifier

Random Forest는 머신러닝 기법 중 하나고, 오버피팅을 방지하기 위해, 최적의 기준 변수를 랜덤 선택한다. Forest라는 말이 들어간 이유는 마치 숲을 이루듯이 여러 개의 의사결정트리들이 앙상블을 형성하여 Random Forest를 만들기 때문이다. [^13]

[^13]: scikit-learn. (n.d.). RandomForestClassifier Documentation. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

여기서 앙상블이란 강력한 하나의 모델을 사용하지 않고, 약한 여러 개의 모델을 사용하는 것이다. 의사결정트리들의 예측들을 조합하여 결론을 도출한다. 다양한 예측들을 이용하기에 더 정확한 예측이 가능해진다.

의사결정트리의 알고리즘을 먼저 살펴보면 Random Forest의 이해가 쉬울 것이다. 먼저 부트스트래핑(Bootstraping)이라는 데이터 셈플링 방법을 시행한다. 이는 데이터에서 샘플링을 할 때 복원추출을 한다. 학습데이터셋을 여러가지로 만들어 유니크하다는 특징을 가지고 있다. 부트스트래핑 한 데이터셋을 통해 의사결정트리들을 만든다. 작은 나무들이기에 노드가 적고 특성이 많지 않다는 특징이 있다. 모든 특성을 사용하지 않고, 몇 개의 특성만 골라서 사용하는 것이다.

이후 의사결정트리들을 합치는 과정(Aggregating)을 거친다. 이는 문제의 종류에 따라 방식이 달라진다. 회귀 문제의 경우 기본 모델들의 결과들을 평균 내고, 분류 문제일 경우 다수결로 투표해서 가장 많은 투표를 받은 것을 결과로 낸다.

Random Forest는 분류와 회귀 문제를 모두 다룰 수 있으며 편의성과 유연성이 뛰어나다. 결측치를 다루기 쉽고, 대용량 데이터 처리가 쉽다는 장점도 가지고 있다. 또한 과적합 문제를 해결해주고, 특성중요도를 구할 수 있다. [^14] 이러한 장점을 가지고 있기에 우리 조는 지도학습의 장치로 Random Forest를 사용할 것이다.

[^14]: Breiman L (2001). "Random Forests". Machine Learning.

#### ii) Code Explanation

##### 0. Initialize

**0-1) Package Import**

```R
library(randomForest)
library(caret)
library(pROC)
```

먼저 필요한 R package를 불러온다. 필요한 패키지는 다음과 같다.

- randomForest: Random Forest 패키지
- caret: 복잡한 회귀와 분류 문제에 대한 모형 훈련과 조절과정을 간소화하는 패키지
- pROC: pROC curve 패키지

**0-2) Dir Setting and Read Dataset**

```R
setwd("~/your_folder")
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')
sampled_df <- processed_df[complete.cases(processed_df[,c("message", "most_common_report_reason")]),]
```

Working Directory를 현재 폴더로 설정하고, 전처리 과정을 거친 데이터 셋을 불러온다. 데이터셋에서 결측치는 제거한다.

##### 1.Feature Engineering

**1-1) 범주형 변수로 변환**

```R
sampled_df$most_common_report_reason = as.factor(sampled_df$most_common_report_reason)
```

'most_common_report_reason' 을 범주형으로 factorize한다. 변수들은 범주형과 연속형으로 나눠지는데, 'most_common_report_reason'은 이산적으로 나누어져야 하는데 데이터기에 범주형으로 바꿔주는 작업을 해야한다.

범주형으로 바꾸는 함수는 as.factor()이고, 'most_common_report_reason'라는 기존에 있는 변수를 범주형으로 덮어쓰기 위해 위의 코드를 사용한다.

**1-2) 정수 시퀀스 변환**

```R
set.seed(sample(100:1000,1,replace=F)) # Random sampling
trainIndex <- createDataPartition(sampled_df$most_common_report_reason, p = 0.8, list = FALSE)
train_data <- sampled_df[trainIndex, ]
test_data <- sampled_df[-trainIndex, ]

train_features <- setdiff(names(train_data), "most_common_report_reason")
test_features <- setdiff(names(test_data), "most_common_report_reason")
```

Supervised Learning(Regression)에서 사용한 것과 동일하게 데이터를 학습데이터와 테스트 데이터로 랜덤하게 분할한다. 비율은 학습데이터가 75%, 테스트 데이터가 25%이다.

이후 train_data와 test_data 간의 column이 일치하는지 확인한다.

##### 2. Random Forest Classification

**2-1) 모델학습**

```R
rf_start_time <- Sys.time()

rf_model <- randomForest(most_common_report_reason ~ ., data = train_data, ntree = 100)

rf_end_time <- Sys.time()

rf_elapsed_time <- rf_end_time - rf_start_time
cat("Training Time (SVM): ", rf_elapsed_time, "\n")
```

모델을 학습시키고 이 과정에서 학습 시작 시간과 종료 시간을 체크한다. 이후 경과 시간을 평가한다.

**2-2) 모델 테스트**

```R
rf_predictions <- predict(rf_model, newdata = test_data)

confusion_matrix_rf <- table(Actual = test_data$most_common_report_reason, Predicted = rf_predictions)
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
```

위에서 학습시킨 모델을 토대로 'test_data'에 대한 예측을 수행하게 한다. 이후 혼돈행렬과 정확도로 나타낸다.

**2-3) 결과 출력**

```R
print("Confusion Matrix (Random Forest):")
print(confusion_matrix_rf)
cat("Accuracy (Random Forest):", accuracy_rf, "\n")
```

Random Forest 예측결과에 따라 Confusion Matrix와 정확도를 출력하는 작업을 한다.

##### 3. Visualization

**3-1) 시각화: Heatmap**

```R
conf_mat_rf <- confusionMatrix(rf_predictions, test_data$most_common_report_reason)

conf_matrix_values <- conf_mat_rf$table

heatmap(conf_matrix_values,
        col = c("white", "lightblue", "blue"),
        main = "Confusion Matrix (Random Forest)",
        xlab = "Predicted",
        ylab = "Actual")
```

Random Forest 모델의 예측 결과를 시각화하기 위해 Confusion Matrix를 히트맵(Heatmap)으로 표현하는 작업을 진행한다.

확률을 셋으로 나눠서 확률이 높은 순서부터 blue, lightblue, white이다.

**3-2) 시각화: ROC curve**

```R
rf_probs <- as.numeric(predict(rf_model, newdata = test_data, type = "response"))

actual_labels_binary <- as.numeric(test_data$most_common_report_reason) - 1

roc_curve <- roc(actual_labels_binary, rf_probs)

plot(roc_curve, col = "blue", main = "ROC Curve for Random Forest",
     col.main = "darkblue", col.lab = "black", lwd = 2)
```

Random Forest에 대한 ROC curve를 생성하고 시각화는 작업을 수행한다.

#### iii) Result Evaluation and Analysis

- Random forest 모델을 활용하여 도출된 confusion matrix를 각각 Heat map과 ROC curve로 시각화 하였다. 이때 heat map은 채팅 신고 사유의 예측 값과 실제 값을 비교하는 confusion matrix를 경우의 수에 따라 색상의 진한 정도로 시각화한 것이다.

  - **Confusion matrix**는 모델의 훈련을 통한 예측의 성능을 측정하기 위해 예측값과 실제 값을 비교하기 위한 표를 의미한다.
    <img width="480" alt="Cfm" src="https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/6c89d411-2bb3-40b7-8469-16ce21568406">

  - 위 표에서는 예측 값과 실제 값의 일치 여부에 따라 총 4개의 경우의 수로 나타낸 것이다. 각각 TP(True Positive)는 긍정예측을 성공, TN(True Negative)는 부정예측을 성공, FP(False Positive)는 긍정예측을 실패, FN(False Negative)는 부정예측을 실패한 경우를 의미한다. 위 경우의 수가 발생한 횟수를 계산하여 정확도, 민감도, 정밀도 등의 평가 척도를 계산할 수 있다.

**A. RF-Classifier: Heatmap**

![Heatmap (improved)](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/a5326c3d-fb4f-46dd-8f81-6fa1c5a5225f)

- Classifier Model의 Confusion Matrix를 Heatmap으로 시각화한 결과이다.
- Heat map을 살펴보면 예측 신고 사유와 실제 신고 사유가 각각 6가지 존재한다.
  - 신고 사유 Inappropriate Name(부적절한 이름), Spamming(스팸성 채팅)는 모두 가장 높은 비율로 예측 신고 사유와 실제 신고 사유가 일치했다.
  - 가장 정확도가 낮은 예측 값은 신고 사유 Negative Attitude였는데 이는 Inappropriate Name 사유만을 제외한 나머지 실제 신고 사유에 모두 높은 비율로 분포했다.

**B. RF-Classifier: ROC Curve**

- ROC curve는 False Positive Rate(FPR)와 True Positive Rate(TPR)를 각각 x축, y축에 표시한 그래프로 모델의 민감도(sensitivity)와 특이도(specificity)를 평가하기 위해 사용한다. Heatmap에 이어, ROC Curve 역시 random forest에서 도출된 confusion matrix를 시각화 하는 데에 사용되었다.

![ROC Curve](https://github.com/KyumKyum/gamechatban.github.io/assets/59195630/043588be-5ffe-4cc6-b1d2-84bf054e61d7)

- ROC curve는 일단 그래프 자체가 참조선 위에 그려졌다. 그러나 이론적으로 정확도가 높다고 판별하는 기준인 AUC 값 0.8 보다 높다고 보기는 어렵다.
- 이는 앞선 confusion matrix의 heat map을 통해서도 확인했듯이 True Positive Rate(TPR)가 높다고 보기 어려웠다. 2개의 신고 사유를 제외하곤 예측 값과 실제 값의 차이가 컸기 때문이다.
- 이로 인해 TPR을 나타내는 ROC curve의 sensitivity 값의 빠른 증가와 참조선을 상회하는 크기의 그래프의 기울기를 그림에서 찾아볼 수 없었다.

#### iv) Full Code

```R
# AI-X Final Project
# Supervised Learning after doing unsupervised learning.
# Part 2: Classifying Common Report Reason of Chatting

# Dataset: A chatlog with toxic score assessed by unsupervised learning.
# Column: X(id), Message, Most common report reason, Toxic Score

# Load required libraries
library(randomForest) # Random Forest Model
library(caret) # Confusion Matrix
library(pROC) # pROC Curve


setwd("~/Desktop/Dev/HYU/2023-02/AI-X/project/gamechatban") # Change this value to your working dir.

# Read a dataset for supervised learning.
processed_df <- read.csv('./offender_chatlog_with_toxic_score.csv')
# Remove missing value
sampled_df <- processed_df[complete.cases(processed_df[,c("message", "most_common_report_reason")]),]

# Factorize the target variable 'most common report reason'.
sampled_df$most_common_report_reason = as.factor(sampled_df$most_common_report_reason)

# Split the data into training and testing sets
set.seed(sample(100:1000,1,replace=F)) # Random sampling
trainIndex <- createDataPartition(sampled_df$most_common_report_reason, p = 0.8, list = FALSE)
train_data <- sampled_df[trainIndex, ]
test_data <- sampled_df[-trainIndex, ]

# Make sure the columns match between train_data and test_data
train_features <- setdiff(names(train_data), "most_common_report_reason")
test_features <- setdiff(names(test_data), "most_common_report_reason")

# Model: Random Forest Classifier
rf_start_time <- Sys.time()

# Train a Random Forest model
rf_model <- randomForest(most_common_report_reason ~ ., data = train_data, ntree = 100)

# Check ending time
rf_end_time <- Sys.time()

# Calcualte Elapsed time;
rf_elapsed_time <- rf_end_time - rf_start_time
cat("Training Time (SVM): ", rf_elapsed_time, "\n")

# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate the Random Forest model
confusion_matrix_rf <- table(Actual = test_data$most_common_report_reason, Predicted = rf_predictions)
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)

# Display confusion matrix and accuracy for Random Forest
print("Confusion Matrix (Random Forest):")
print(confusion_matrix_rf)
cat("Accuracy (Random Forest):", accuracy_rf, "\n")

# Visualization (Confusion Matrix Plot for Random Forest)
conf_mat_rf <- confusionMatrix(rf_predictions, test_data$most_common_report_reason)
# Extract confusion matrix values
conf_matrix_values <- conf_mat_rf$table

# Plot the confusion matrix
heatmap(conf_matrix_values,
        col = c("white", "lightblue", "blue"),
        main = "Confusion Matrix (Random Forest)",
        xlab = "Predicted",
        ylab = "Actual")

# ROC Curve
#The Receiver Operating Characteristic (ROC) curve is a graphical representation that illustrates the performance of a binary classification model across different discrimination thresholds.
# It plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) for various threshold values.
# Make predictions on the test set
rf_probs <- as.numeric(predict(rf_model, newdata = test_data, type = "response"))

# Create binary labels for ROC curve
actual_labels_binary <- as.numeric(test_data$most_common_report_reason) - 1  # Assuming binary classification

# Create ROC curve
roc_curve <- roc(actual_labels_binary, rf_probs)

# Plot the ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve for Random Forest",
     col.main = "darkblue", col.lab = "black", lwd = 2)
```

---

# V. Video Link

{% include youtube.html id="2JzGY2u0wz8" %}

---

# VI. Conclusion: Discussion

비지도 학습을 통해 toxic level이 높은 단어는 단순한 단어보다 길고 상세한 말뭉치라는 것을 깨달을 수 있었다. TF-IDF 결과를 토대로 나온 용어들을 보며 사람들이 불쾌함을 느낄 수 있는 말들을 배울 수 있었으며, 특히 그 언어 자체의 공격성보다는 빈도와 분량이 많을수록 더욱 불쾌함을 느꼈다는 것을 알 수 있었다.

지도학습에서는 모델에 대해서 알 수 있었다. LSTM의 training loss가 epoch의 증가에 따라 감소했다는 것은 학습을 많이 할수록 loss값을 줄일 수 있다는 것을 배울 수 있다. Light GBM에서는 Q-Q plot을 보며 regression 모델의 오차 가능성을 볼 수 있었다.

Random Forest에 대해서는 사유마다 예측 일치 비율이 다르다는 것을 heat map을 통해 직관적으로 알 수 있었다. 또한 ROC curve를 통해 그래프가 참조선 위에서는 그려졌다는 것을 알 수 있었다. 그러나 정확도가 30% 중반 수준에 머무른 것은 아쉬움으로 남는다. 추후에 프로젝트를 진행한다면 이보다 정확도를 높이는 방향으로 나아가야 할 것이다.

게임 내의 부적절한 채팅 목록을 비지도 학습의 결과물을 보며 배우며 유저 스스로가 채팅의 부적절성을 판단할 수 있다. 게임 내에서 사용하기 부적절한 단어는 일상생활에서도 사용해서는 안 되는 단어들이다. 이런 단어들에 대하여 경각심을 가지고 건전한 언어 습관을 만들어야 할 것이다.

또한 우리 조의 지도 학습 프로그램을 통해 게임 업계에서 부적절한 채팅을 감지하고 그런 채팅을 막는데 도움을 줄 수 있다. 프로그램의 사용으로 보다 깨끗한 채팅 문화를 이끌어 갈 수 있기를 희망한다.

해당 프로젝트에 사용된 코드는 해당 [Github Repository](https://github.com/KyumKyum/gamechatban_deep_learning_proj)에 open으로 공개를 해두었다.

---

# VI. Related Works

1. 이준명, 나정환, 도영임, 플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계, 한국게임학회 논문지,2016,

2. Rajaraman, A. Ullman, J.D, Mining of Massive Datasets, Data Mining, 2011, 1-17,
   Akiko Aizawa, An information-theoretic perspective of tf–idf measures, National Institute of Informatics, January 2003,

3. Manning, C. D. Raghavan, P. Schutze, 《Introduction to Information Retrieval》, Cambridge University Press, 1983, 100-123,

4. "Introducing dplyr", blog.rstudio.com, Retrieved 2023-11-24,

5. FEINERER, Ingo. Introduction to the tm Package Text Mining in R. Accessible en ligne: http://cran. r-project. org/web/packages/tm/vignettes/tm. pdf, 2023, 1

6. Manning, C.D., Raghavan, P. Schutze, et. al, "Scoring, term weighting, and the vector space model", Introduction to Information Retrieval.

7. Christopher M. Bishop, "Pattern Recognition and Machine Learning", 2006, Springer

8. Pedro Domingos, "A Few Useful Things to Know About Machine Learning", 2012, Association for Computing Machinery

9. Sak, Hasim; Senior, Andrew; Beaufays, Francoise (2014). "Long Short-Term Memory recurrent neural network architectures for large scale acoustic modeling"

10. LightGBM. (n.d.). LightGBM Documentation. Retrieved from https://lightgbm.readthedocs.io/en/stable/

11. A. Marden, J. I. (2004). Positions and QQ plots. Statistical Science, 606-614.

12. Ho, Tin Kam (1995). “Random Decision Forests.” Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal, QC, 14–16 August 1995.

13. Ho TK (1998). "The Random Subspace Method for Constructing Decision Forests". IEEE Transactions on Pattern Analysis and Machine Intelligence.

14. Breiman L (2001). "Random Forests". Machine Learning.

15. Dietterich, Thomas (2000). "An Experimental Comparison of Three Methods for Constructing Ensembles of Decision Trees: Bagging, Boosting, and Randomization". Machine Learning.

16. A. Deng, X., Liu, Q., Deng, Y., & Mahadevan, S. (2016). An improved method to construct basic probability assignment based on the confusion matrix for classification problem. Information Sciences, 340, 250-261.

17. A. Park, S. H., Goo, J. M., & Jo, C. H. (2004). Receiver operating characteristic (ROC) curve: practical review for radiologists. Korean journal of radiology, 5(1), 11-18.
