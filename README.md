# 문학 작품 정량적 감정 구조화 프레임워크

문장 단위 감정 분류 모델과 키워드 감정 귀속 분석을 결합하여 문학 텍스트의 정서 구조를 정량적으로 해석하는 프로젝트입니다. 작품 내 주요 키워드가 어떤 감정과 어떤 비율로 연결되어 사용되는지를 수치적으로 & 정량적으로 분석하는 프레임워크를 제안합니다.
프로젝트를 마치고 명세화하는 과정에서 지도 교수님으로부터 학술적 가치를 인정받아 논문으로 출간하였으며, 학술발표회에서 프로시딩 심사를 받고 있습니다.(2025/05/19 작성)

<img width=100% alt="image" src="https://github.com/user-attachments/assets/472083ca-5173-4a05-9ef9-ffa8ba3bccfd" />


<img width=100% alt="image" src="https://github.com/user-attachments/assets/781e85fb-094b-4d05-bee5-e170f2d910d6" />


## Keywords : #Deep Learning #NLP # Emotion Analysis

---

## Project Workflow

1. **작품 원문 수집 및 전처리**

   * 웹 기반 아카이브에서 「운수 좋은 날」 원문 수집
   * 줄바꿈 및 특수문자 제거 후 문장 단위로 분할 (마침표 기준)

2. **TF-IDF 기반 주요 키워드 추출**

   * 형태소 분석(KLT2023)을 통해 명사/고유명사 추출
   * 불용어 제거 후 TF-IDF 점수로 상위 10개 키워드 선정

3. **감정 분류 모델 학습**

   * `emotion_korTran` 데이터셋(41만 문장)으로 Transformer 모델 학습 - 6가지 감정 범주 분류 (joy, sadness, anger, fear, surprise, love)

4. **문장 단위 감정 예측**

   * 학습된 모델로 「운수 좋은 날」 각 문장의 감정을 예측

5. **감정 결과의 키워드 귀속**

   * 예측된 감정 레이블을 해당 문장에 포함된 키워드([2번] - TF-IDF기반으로 추출한 키워드)에 귀속
   * 키워드별로 감정 분포 비율을 계산하여 6차원 벡터 생성

6. **정량 분석 및 시각화**

   * 감정 벡터를 통해 각 키워드의 정서적 기능 분석
   * 키워드-감정 분포 테이블 및 시각화 결과 출력

---

## 프로젝트 개요

* **분석 대상**: 현진건의 단편소설 「운수 좋은 날」
* **분석 목적**: 간접적이고 주체가 모호한 문학 감정 표현을 문맥 기반 Transformer 모델로 분류하고, 키워드 단위로 감정 분포를 정량화
* **감정 분류**: joy, sadness, anger, fear, surprise, love (총 6개 범주)

---

## 데이터 구성

* 원문 문장 수: 약 200문장
* 학습 데이터: `emotion_korTran` (41만 개 문장, 감정 레이블 포함)
* 키워드 추출: 형태소 분석(KLT2023) 후 명사 기반 TF-IDF 계산
* 상위 키워드 예시: 김첨지, 소리, 설렁탕, 인력거 등

---

## 모델 및 학습 정보

* Transformer 기반 감정 분류 모델
* From-scratch 학습 (사전학습 미사용)
* 손실 함수: CrossEntropyLoss
* 옵티마이저: Adam
* 최종 정확도: 86.3%

---

## 결과 요약

### 키워드별 감정 분포 벡터

        anger      fear       joy      love   sadness  surprise
| 키워드   | anger   | fear    | joy     | love    | sadness | surprise |
|----------|---------|---------|---------|---------|---------|----------|
| 김첨지   | 0.1489  | 0.0638  | 0.5532  | 0.0000  | 0.1915  | 0.0426   |
| 소리     | 0.2381  | 0.0476  | 0.3810  | 0.0000  | 0.2857  | 0.0476   |
| 사람     | 0.1818  | 0.3636  | 0.2727  | 0.0000  | 0.0909  | 0.0909   |
| 치삼     | 0.3333  | 0.1667  | 0.5000  | 0.0000  | 0.0000  | 0.0000   |
| 얼굴     | 0.0909  | 0.1818  | 0.2727  | 0.0000  | 0.4545  | 0.0000   |
| 인력거   | 0.0769  | 0.3077  | 0.5385  | 0.0000  | 0.0000  | 0.0769   |
| 정거장   | 0.1000  | 0.3000  | 0.4000  | 0.0000  | 0.2000  | 0.0000   |
| 다리     | 0.2000  | 0.0000  | 0.6000  | 0.0000  | 0.2000  | 0.0000   |
| 설렁탕   | 0.2857  | 0.0000  | 0.1429  | 0.2857  | 0.2857  | 0.0000   |
| 곱배기   | 0.0000  | 0.0000  | 0.8000  | 0.0000  | 0.2000  | 0.0000   |



![image](https://github.com/user-attachments/assets/9a98324b-c819-46c9-8aeb-6517ddcd6879)
* 감정 분포는 해당 키워드가 사용된 문장에서의 감정 비율을 의미함
* 키워드와 감정 간의 연결 관계를 수치적으로 파악 가능


---

## 주요 의의

* 문맥 기반 감정 분류로 감성 사전 방식의 한계 보완
* 문장 감정을 키워드에 귀속하여 감정-키워드 관계를 정량적으로 분석
* 서사 내 감정 흐름, 인물 간 감정 관계 등 확장 가능성 존재

---

## 🛠️ 기술 스택

* Python 3.10+
* PyTorch, Hugging Face Transformers
* scikit-learn (TF-IDF)
* KoNLPy / konlp.kma.klt2023

---

## 프로젝트 구조

```
.
├── check_sentiment_labels.ipynb
├── emotion_korTran
│   ├── about_dataset.ipynb
│   ├── emotion_korTran.data
│   └── README.txt
├── emotion_mapping_results.py
├── KnuSentiLex
│   ├── data
│   │   └── SentiWord_info.json
│   ├── KnuSentiLex
│   │   ├── data
│   │   │   └── SentiWord_info.json
│   │   ├── knusl.py
│   │   ├── neg_pol_word.txt
│   │   ├── obj_unknown_pol_word.txt
│   │   ├── pos_pol_word.txt
│   │   ├── README.md
│   │   ├── ReadMe.txt
│   │   ├── SentiWord_Dict.txt
│   │   └── src
│   │       └── knusl.py
│   ├── knusl.py
│   ├── neg_pol_word.txt
│   ├── obj_unknown_pol_word.txt
│   ├── pos_pol_word.txt
│   ├── README.md
│   ├── ReadMe.txt
│   └── SentiWord_Dict.txt
├── README.md
├── tf-idf.py
├── train.py
├── 운수좋은날_감정예측결과.csv
└── 운수좋은날.txt
```
