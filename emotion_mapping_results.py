from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import defaultdict

# 감정 레이블 리스트
emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]

df_emotion = pd.read_csv("/Users/newuser/한승민/2025-1/빅최기/감정분석/운수좋은날_감정예측결과.csv") # 감정 예측 결과 CSV 파일 불러오기

# TF-IDF 상위 키워드 리스트
top_keywords = ['김첨지', '소리', '사람', '치삼', '얼굴', '인력거', '정거장', '다리', '설렁탕', '곱배기']

emotion_counts = {kw: defaultdict(int) for kw in top_keywords}

for idx, row in df_emotion.iterrows():
    sentence = row['original_sentence']
    emotion = row['predicted_emotion']
    for keyword in top_keywords:
        if keyword in sentence:
            emotion_counts[keyword][emotion] += 1

keyword_vectors = {}
for kw in top_keywords:
    counts = np.array([emotion_counts[kw][emo] for emo in emotion_labels])
    total = counts.sum()
    if total > 0:
        vector = counts / total
    else:
        vector = np.zeros_like(counts)
    keyword_vectors[kw] = vector

df_vectors = pd.DataFrame.from_dict(keyword_vectors, orient='index', columns=emotion_labels)

print(df_vectors)# 감정 분포 벡터 출력


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
sns.heatmap(df_vectors, annot=True, cmap="Greys", cbar=True, linewidths=0.5, fmt=".2f") # 히트맵 생성

plt.xlabel("감정 레이블")
plt.ylabel("키워드")
plt.tight_layout()
plt.show()