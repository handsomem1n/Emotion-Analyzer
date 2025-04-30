import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score

# ==================== 데이터 로딩 ====================
texts, labels = [], []
with open("/data/seungmin/0412temp/emotion_korTran/emotion_korTran.data", "r", encoding="utf-8") as f:
    next(f)
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 3:
            label = row[-1].strip()
            text = ",".join(row[1:-1]).strip()
            texts.append(text)
            labels.append(label)

df = pd.DataFrame({"text": texts, "label": labels})

# 라벨 인코딩
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

# ==================== 토크나이징 및 인코딩 ====================
def tokenize(text):
    return text.lower().split()

vocab_counter = Counter()
for sentence in df["text"]:
    vocab_counter.update(tokenize(sentence))

vocab = {"<pad>": 0, "<unk>": 1}
for i, word in enumerate(vocab_counter.keys(), 2):
    vocab[word] = i

def encode(text, max_len=128):
    tokens = tokenize(text)
    token_ids = [vocab.get(tok, 1) for tok in tokens][:max_len]
    token_ids += [0] * (max_len - len(token_ids))
    return token_ids

# ==================== 데이터셋 클래스 ====================
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==================== 학습용 데이터 생성 ====================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label_id"], test_size=0.1, stratify=df["label_id"], random_state=42
)

train_encodings = [encode(t) for t in train_texts]
test_encodings = [encode(t) for t in test_texts]
train_dataset = EmotionDataset(train_encodings, list(train_labels))
test_dataset = EmotionDataset(test_encodings, list(test_labels))

# ==================== 모델 정의 ====================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=4, num_layers=3, num_classes=6, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1), :]
        x = self.transformer_encoder(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# ==================== 학습 설정 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size=len(vocab)).to(device)
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

best_val_loss = float('inf')
patience, counter = 3, 0

# ==================== 학습 루프 ====================
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} [Valid]"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Validation Loss = {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# ==================== 테스트 평가 ====================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Test Accuracy:", accuracy_score(all_labels, all_preds))