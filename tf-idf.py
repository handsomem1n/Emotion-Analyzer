from konlp.kma.klt2023 import klt2023
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

with open("/Users/newuser/한승민/2025-1/빅최기/운수좋은날.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f]

analyzer = klt2023()

# 불용어 사전
stopwords = set([
    '것', '수', '때', '자기', '이', '그', '저', '너', '나', '내',
    '좀', '또', '더', '등', '등등',
    '하고', '부터', '까지', '에서', '으로', '에게', '께서', '이다', '하며', '였다', '그때', '이놈',
    '그의', '그는', '내가', '이의', '이런', '같은', '듯이', '라고', '하는', '오늘', '이날', '이때',
    '그것', '무엇', '즈음', '거기', '우리', '제', '좀', '하며', '라며', '하며', '하고서', '그러나', '그래서', '그러면', '그런데', 
'있다', '없다', '되다', '되었다', '가다', '오다', '보다', '싶다', '같다',
'이리', '저리', '여기', '저기', '그곳', '그리', '어디', '누구', '무엇'
])

# 명사만 추출
noun_sentences = []
for s in sentences:
    morphs = analyzer.pos(s)
    nouns = [w.split('/')[0] for w in morphs if '/N' in w or '/K' in w]
    filtered = [w for w in nouns if len(w) > 1 and w not in stopwords]
    noun_sentences.append(' '.join(filtered))

# TF-IDF 벡터화 (불용어 제거 적용)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(noun_sentences)

# TF-IDF 점수 계산
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.toarray().sum(axis=0)

# 키워드 상위 20개 추출
keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
print("TF-IDF 기반 상위 키워드 (불용어 제거 적용):")
for word, score in keywords[:10]:
    print(f"{word}: {round(score, 3)}")
    
    
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

top_keywords = keywords[:10]
words, scores = zip(*top_keywords)

plt.figure(figsize=(10, 6))
plt.barh(words[::-1], scores[::-1],color = 'gray')  # 점수가 높은 순으로 위에 오도록 뒤집음
plt.xlabel("TF-IDF 점수")
# plt.title("TF-IDF 기반 상위 키워드 (불용어 제거 적용)")
plt.tight_layout()
plt.show()

