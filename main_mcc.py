#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
128차원 pitch contour + 억양 패턴(LLL~HHH)을 이용한
PyTorch 분류 모델 학습 스크립트.
"""

import os
import numpy as np
import pandas as pd
import pickle
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# =========================
# 설정값
# =========================
CSV_PATH = "/Users/donaldos/workspace/autointonation/modelling/intonationpattern.csv"   # <-- 여기에 실제 csv 파일 이름 넣으세요
BATCH_SIZE = 32
NUM_EPOCHS = 400
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_SAVE_PATH = "pitch_pattern_model.pt"
ENCODER_SAVE_PATH = "label_encoder.pkl"
SCALER_SAVE_PATH = "scaler.pkl"


# =========================
# 데이터 로딩 & 전처리
# =========================

def load_and_preprocess(csv_path: str) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                LabelEncoder, StandardScaler]:
    """
    CSV를 읽어 X, y를 만들고
    LabelEncoder, StandardScaler를 적용한 후
    train/test로 나눈다.
    """
    # header 없음으로 가정
    df = pd.read_csv(csv_path, header=None)

    # 마지막 column이 label
    labels = df.iloc[:, -1]

    # 클래스별 개수 확인
    counts = labels.value_counts()

    # 샘플이 2개 미만인 클래스를 제거
    valid_labels = counts[counts >= 2].index
    df = df[df.iloc[:, -1].isin(valid_labels)]    


    # 마지막 컬럼이 label, 나머지 컬럼이 feature
    X = df.iloc[:, :-1].values.astype(np.float32)   # (N, 128)
    y_str = df.iloc[:, -1].values                   # (N,)

    # 문자열 라벨 → 정수 인코딩 (0 ~ num_classes-1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE #, stratify=y
    )

    # feature 스케일링 (평균 0, 분산 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler


# =========================
# PyTorch Dataset
# =========================

class PitchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# 모델 정의 (MLP)
# =========================

class PitchClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 학습 & 평가 루프
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)           # (batch, num_classes)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)               # (batch, num_classes)
            _, predicted = torch.max(outputs, 1)   # (batch,)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc


# =========================
# 예측 함수 (새로운 데이터 1개용)
# =========================

def predict_pattern(model, scaler, label_encoder, pitch_vector_128, device):
    """
    pitch_vector_128: 길이 128짜리 numpy array 또는 list
    """
    model.eval()

    x = np.asarray(pitch_vector_128, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x_tensor)              # (1, num_classes)
        pred_idx = torch.argmax(logits, dim=1).item()

    label = label_encoder.inverse_transform([pred_idx])[0]
    return label


# =========================
# main
# =========================

def main():
    # 1) 데이터 로딩 & 전처리
    print(f"Loading data from {CSV_PATH} ...")
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess(CSV_PATH)

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")

    # 2) Dataset / DataLoader
    train_dataset = PitchDataset(X_train, y_train)
    test_dataset = PitchDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3) 모델 / 손실함수 / 옵티마이저
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    plt.ion()  # interactive mode 켜기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)
    train_losses = []
    test_accuracies = []


    model = PitchClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4) 학습 루프
    best_test_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc

                # ----- 실시간 그래프 업데이트 -----
        ax1.cla()
        ax2.cla()

        ax1.plot(train_losses)
        ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2.plot([a * 100 for a in test_accuracies])
        ax2.set_title("Test Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")

        plt.pause(0.01)

        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    plt.ioff()
    plt.show()

    # 5) 모델, 인코더, 스케일러 저장
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to   : {MODEL_SAVE_PATH}")
    print(f"LabelEncoder saved to: {ENCODER_SAVE_PATH}")
    print(f"Scaler saved to      : {SCALER_SAVE_PATH}")

    # 6) 샘플 하나 예측 테스트
    sample = X_test[0]  # 스케일 된 값이지만, predict 함수에서 다시 스케일하므로 원래 X 하나를 쓰는 게 더 자연스럽다.
    # 여기서는 예시로 raw vector를 다시 부르는 식으로 사용하려면 원래 X에서 하나 가져오면 됨.
    raw_sample = scaler.inverse_transform(sample.reshape(1, -1))[0]  # 되돌려서 형태만 맞춤

    pred_label = predict_pattern(model, scaler, label_encoder, raw_sample, device)
    true_label = label_encoder.inverse_transform([y_test[0]])[0]
    print(f"Sample prediction: pred={pred_label}, true={true_label}")


if __name__ == "__main__":
    main()
