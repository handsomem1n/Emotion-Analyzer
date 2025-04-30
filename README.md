# Transformer + TF-IDF 기반 문학 텍스트 감정 분석

이 프로젝트는 문학 작품 내 감정 흐름을 정량적으로 분석하기 위한 실험으로, 현진건의 단편소설 「운수 좋은 날」을 분석 대상으로 삼았습니다. 단순한 감정 키워드 매칭이 아닌, 문장 단위 감정 분류 모델을 학습하고 이를 기반으로 주요 키워드의 정서적 특성을 수치화하는 작업을 중심으로 구성되어 있습니다.

## 구성원
1인

## 수행 기간
2025년 03월 17일 ~ 진행 중

## 목적

- 감정 주체가 불명확하고 정서 표현이 간접적인 문학 텍스트에서 전통적인 **감성 사전 기반 접근**(KNU 감성사전)이 가지는 한계 확인
- Transformer 모델을 활용한 문장 단위 감정 분류 실험
- 분류된 감정 결과를 TF-IDF 기반 키워드에 귀속시켜 키워드 감정 분포 정량화

## 사용 기술

- Python 3.11.5
- PyTorch (TransformerEncoder 기반 감정 분류 모델 구현)
- scikit-learn (TF-IDF 벡터화 및 시각화)
- KLT2023 (형태소 분석기 사용)
- emotion_korTran (한국어 감정 분류 학습용 데이터셋)

## 프로젝트 구조

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


## 주요 결과

- 86.3% 정확도의 Transformer 기반 문장 감정 분류 모델 구현
- TF-IDF 상위 키워드(예: 김첨지, 설렁탕, 소리 등)에 대해 감정 레이블 빈도 기반 6차원 분포 벡터 구성
- 각 키워드가 어떤 정서적 맥락에서 반복적으로 사용되었는지 수치적으로 파악 가능
- 분석 결과는 논문 형태로 정리되어 학술용으로 제출됨

## 논문 제목  
**Transformer와 TF-IDF 결합을 통한 문학 작품 정량 분석 연구** (확정은 아님)

---
