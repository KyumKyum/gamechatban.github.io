# Index

**I. Proposal**

1. Motivation
2. Objective

**II. Team Members**

**III. Dataset**

**IV. Project Strategy**

1. Unspervised Learning
2. Spervised Learning

**V. Evaluation & Analysis**

**VI. Conclusion: Discussion**

**VII. Related Works**

---

# I. Proposal

### **1. Motivation**

팀 게임을 하면 채팅 기능이 존재한다. 팀원들끼리 채팅을 통해 협력하고 소통하기 위함이다. 적들과 대화하는 것 또한 가능하다. 그러나 상대방을 실제로 마주하고 있지 않은 온라인의 특성 상 상대방을 불쾌하게 만드는 대화가 자주 등장한다.

해당 근거를 뒷받침하는 연구 결과 역시 존재한다. 논문 <<플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계>>[^1] 에서 밝히기를, 트롤링이 일어나는 원인을 '익명성', '시스템적 규제의 허술', '실시간', '다중성'으로 분석하고 있다. 또한 트롤링의 원인을 개인의 특성에서 둔 연구를 보면 트롤링이 재미, 지루함, 복수와 같은 심리적인 요인에서 나오는 것이라고 분석하였다.

[^1]: 이준명, 나정환, 도영임, 플레이어의 개인 성향과 게임 내의 트롤링 행위의 관계, 한국게임학회 논문지,2016,

**((내용 추가! - 브릿지 내용))**

**((해당 문단 내용을 조금 예쁘게 바꿔주세요!!))** 이러한 채팅을 단속하고 향후 보다 깨끗한 인터넷 채팅문화를 만들고자 이 프로젝트를 진행하였다. 우리 조는 리그오브레전드(League of Legends)라는 전세계적으로 유행하는 AOS 장르 게임의 채팅 신고 내역을 분석하여 해당 채팅이 사람으로 하여금 얼마나 불쾌하게 만드는지 비지도학습과 지도학습을 활용하여 분석하고자 한다.

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

- 2-3) Logistic Regression Model을 활용하여 분류 모델을 학습시킨 후, 오차율과 모델의 결과값을 시각화 한다.
- 2-4) SVM을 활용하여 분류 모델을 학습시킨 후, 오차율과 모델의 결과값을 시각화 한다.

**3. 그래프 및 학습 결과 분석 단계**

---

# II. Team Memebers

- 생명과학과 2018023427 이승현: 데이터 전처리 전략, 그래프 및 딥러닝 결과 분석
- 정보시스템학과 2019014266 임규민: 특징 공학 및 코드 작성
- 정치외교학과 2022094366 장정원: 자료 수집 및 글 작성, 영상 촬영

---

# III. Dataset

리그오브레전드(League of Legends)에서 report당한 채팅들에 관한 데이터셋이다. contents는 다음과 같다. (출처: Kaggle[^2])

[^2]: https://www.kaggle.com/datasets/simshengxue/league-of-legends-tribunal-chatlogs

- message: 신고당한 메세지 내용

- association_to_offender: 해당 채팅의 소속.

  - Emeny: 적군의 채팅
  - Ally: 아군의 채팅
  - Offender: 신고 당한 유저의 채팅

- time: 신고당한 시간

- case_total_reports: Tribunal이라는 게임 내 시스템으로 넘어가기까지의 신고 횟수

  - Tribunal에 대한 설명을 적어주세요!

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

### **1. Unspervised Learning**

비지도 학습(Unspervised Learning)은 기계학습의 일종이다. 데이터가 어떻게 구성되었는지를 알아내는 문제의 범주에 속하며 입력값에 대한 목표치가 주어지지 않는다.

#### Unsupervised Learning Methodology: TF-IDF

TF-IDF는 문서 내 단어마다 중요도를 고려하여 가중치를 주는 통계적인 단어 표현 방법이다. TF는 단어의 빈도를 고려하는 것이고, IDF는 역 문서 빈도를 고려하는 것으로 이 둘의 곱으로 TF-IDF를 구한다.

TF의 계산 방법은 카운트 기반 단어표현 방법인 DTM(Document Term Matrix)과 동일하다. 먼저 각 문서에서 나타난 전체 단어를 알파벳 순으로 배열한다. 이후 각 문서별로 사용된 단어들의 개수를 세서 벡터로 표현한다. 그러나 DTM은 단순하게 문서 데이터에서의 단어의 빈도수만을 고려한다. 때문에 중요한어와 불필요한 단어를 구분하기 어렵다는 한계점을 가지고 있다.

그렇기에 중요한 단어에는 높은 가중치를 부여하고, 덜 중요한 단어에는 낮은 가중치를 부여할 필요가 있다. 이를 해결하기 위한 것이 DF(Document Frequency)이다.

IDF는 Inverse Document Frequency이다. DF 값의 역수라는 뜻이다. DF는 전체 문서에서의 특정 단어가 등장한 문서 개수이다. 많은 문서에 등장할수록 그 단어의 가치는 줄어들게 된다. 그렇기에 역수를 취하면 적은 문서에 등장할수록 IDF의 값이 높아지게 된다. 이 둘 의 값을 곱한 TF\*IDF 값이 TF-IDF 값이다.

((**TF-IDF에 대한 마무리 글을 적어주세요!**))

#### Unsupervised Learning Strategy using TF-IDF

위의 전략에서 이야기 했듯이, 다음과 같은 분석 단계를 거쳐 비지도학습의 결과를 얻을 것이다.

- 1-1) TF-IDF (Term Frequency - Inverse Document Frequency) 가중치 값을 계산 및 활용하여, 각 채팅 단어의 빈도수를 정규화하여 많이 등장하고 영향력이 높은 단어의 가중치를 계산한다.
- 1-2) 각 계산된 가중치를 바탕으로 각 채팅에 대한 toxic score를 계산한다.
- 1-3) 이후 지도학습에 활용될 column만을 남겨놓는다.

#### 0. Initialize

**Package Import**

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

**Dir Setting and Read Dataset**

```R
setwd("~/your_folder")
chatlogs <- read.csv("./chatlogs.csv")
```

Working Directory를 현재 폴더로 설정하고, 데이터 셋을 불러온다. 이후. 전처리 과정을 시행한다.

#### 1. Feature Engineering

**Ⅰ. Datasets 가지치기**

```R
chatlogs <- chatlogs %>% filter(association_to_offender == 'offender')
chatlogs <- subset(chatlogs, select = -association_to_offender)
```

Datasets의 ‘association_to_offender’ 칼럼 중, “offender”에 해당하는 튜플만 추출한다. 이는 신고당한 사람의 채팅만을 선별하는 과정이다.
추출한 뒤에는 ‘association_to_offender’ 칼럼은 “offender”라는 튜플 외 다른 튜플은 삭제된 의미없는 칼럼이기에 삭제한다.

**Ⅱ. 일부 문법적 표현 및 게임특수성 관련 표현 제거**

```R
champion_names <- read.csv("./champion_names.csv")

pattern <- paste0("\\b(?:is|are|&gt|&lt|was|were|", paste(unique(c(champion_names$Champion, champion_names$Abbreviation)), collapse = "|"), ")\\b")

chatlogs$message <- gsub(pattern, "", chatlogs$message, ignore.case = TRUE)

write.csv(chatlogs, "processed.csv")
```

리그오브레전드라는 게임 특성상 채팅에서 챔피언의 이름을 자주 말하게 된다. 이는 채팅 신고와 상관없는 단어이기에 champion_names.csv 파일을 이용해서 제거한다.
이후 문법적으로 사용되는 영단어인 is, are, &lt, &gt 등도 불필요하기에 제거한다.
처리한 Datasets을 새로 csv 파일로 저장한다.

**Ⅲ. 심각도(severity)에 대한 feature engineering**

```R
chatlogs$severity <- cut(
  chatlogs$case_total_reports,
  breaks = c(-Inf, 3, 6, Inf),
  labels = c("Severe", "Normal", "Low"),
  include.lowest = TRUE
)
```

Datasets의 'case total reports' 값을 기반으로 "severity"를 만든다. severity는 3단계로 나누는데, case total reports가 3 이하인 경우 Severe(심각), 4이상 6이하인 경우 Normal(보통), 7 이상인 경우 Low(낮음)이다. 이는 case total reports가 낮은 경우 적은 사용으로도 신고될 만큼 심각성이 크다고 생각했기 때문이다.

**Ⅳ. 문자열 합치기**

```R
concatenated <- chatlogs %>%
  group_by(most_common_report_reason, severity) %>%
  summarise(concatenated_text = paste(message, collapse = " ")) %>%
  ungroup()

write.csv(concatenated, "concat.csv")
```

신고 사유를 바탕으로 chatlog들을 그룹화한다. 그룹화 한 이후 문자열 합치기(concatenate)한다. 이렇게 신고 사유 별로 분류된 하나의 문자열을 새로운 feature인 ‘심각도(severity)’를 고려하여 작성된다. 이러한 새로운 csv 파일을 저장한다.

이러한 전처리 과정 이후 TF-IDF matrix 분석을 한다. 이는 앞에서 합쳐진 문자열에 대해 TF-IDF를 처리하여 각 단어의 'toxic level'을 얻는 과정이다.

#### 2. TF-IDF

**Ⅰ. TF-IDF를 위한 말뭉치(corpus)를 생성하고 추가 전처리를 진행**

```R
corpus <- Corpus(VectorSource(concatenated$concatenated_text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
```

말뭉치(corpus)는 자연어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합을 뜻한다. 우리 조는 TF-IDF를 처리하기 위하여 말뭉치를 생성한다. 전처리 과정을 거친 텍스트에서 구두점, 여백, 관사 등을 제거하고 모든 문자열을 소문자로 변환한다.

**Ⅱ. DTM 생성 & TF-IDF matrix 생성**

```R
dtm <- DocumentTermMatrix(corpus)

tf_idf <- weightTfIdf(dtm)
tf_idf <- t(as.matrix(tf_idf))
```

TF-IDF를 위해 DTM(Document Term Matrix)를 생성한다. DTM은 Unspervised Learning 처음에 소개했던 대로 문서 단어 행렬이다. DTM을 바탕으로 TF-IDF matrix를 생성한다.

**Ⅲ. matrix 순서 바꾸기**

```R
tf_idf_col_name <- paste(concatenated$most_common_report_reason, concatenated$severity, sep = "_")
colnames(tf_idf) <- tf_idf_col_name
```

현재 matrix 순서대로면 메세지들이 한 행에 연달아 나오는 문제가 발생한다. 이를 해결하기 위하여 새로운 열(column)을 만들고 각 행마다 메세지들을 넣게 matrix의 순서를 바꾼다. 이제 신고 사유와 심각도를 분석하기 쉬워졌다.

**Ⅳ. 값 보정**

```R
tf_idf <- round((tf_idf * 1000), 2)
```

Toxic level을 얻기 위해 값을 보정하고 반올림한다.

**Ⅴ. 새로운 데이터 프레임으로 변환**

```R
tf_idf_df <- as.data.frame(tf_idf)

write.csv(tf_idf_df, "toxic_lev.csv")
```

완료된 TF-IDF matrix를 새로운 데이터 프레임으로 변환하고 'toxicity_lev.csv'라는 CSV 파일로 내보낸다.

**Ⅵ. toxic score 정의 & 계산**

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

#### Visualization

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

지금까지의 설명을 하나의 코드로 나타내면 다음과 같다.

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

이렇게 나온 결과물의 구성이 어떻게 되어있는지 정리하면 다음과 같다.

1. 각각의 열은 열은 각각 순번, 데이터셋에 적혀있던 순번, 채팅 내용, 신고 사유, toxic score 순서대로 정렬되어 있다. 예시는 다음과 같다.
   (csv 파일에서 0점이랑 98.52점 있는 부분 스캔해서 올려주세요)
2. 최저점은 0으로 실제 신고 사유와 무관한 채팅 속 단어들은 대부분 0점이다.
3. 최고점은 98.52이고, 채팅의 길이가 길수록 toxic score가 높게 분포하는 경향을 가지고 있다.
4. 비슷한 toxic score 임에도 채팅 길이의 차이가 있는 경우는 주로 적나라한 욕설이 있을수록 채팅이 짧아도 toxic score가 높게 측정되었다.
5. (그래프 사진 넣어주세요)
   위 그래프는 toxic score와 해당 유저의 신고 당한 횟수를 나타낸 것이다. 그래프는 toxic score이 높은 채팅일수록 적은 신고 횟수를 나타낸다. 이는 수위가 높은 채팅일수록 더 적은 횟수의 신고만으로도 처벌이 이루어졌음을 의미하고, 동시에 toxic score을 도출하는 과정이 정확히 이루어졌음을 시사한다.

### **2. Spervised Learning**

---

# V. Evaluation & Analysis

---

# VI. Conclusion: Discussion

---

# VI. Related Works

Rajaraman, A.; Ullman, J.D, Mining of Massive Datasets, Data Mining, 2011, 1-17,
Akiko Aizawa, An information-theoretic perspective of tf–idf measures, National Institute of Informatics, January 2003,
Manning, C. D.; Raghavan, P.; Schutze, H, 《Introduction to Information Retrieval》, Cambridge University Press, 1983, 100-123,

"Introducing dplyr", blog.rstudio.com, Retrieved 2023-11-24,

FEINERER, Ingo. Introduction to the tm Package Text Mining in R. Accessible en ligne: http://cran. r-project. org/web/packages/tm/vignettes/tm. pdf, 2023, 1

---
