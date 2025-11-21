#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
128차원 pitch contour + 억양 패턴(LLL~HHH)을 이용한
PyTorch LSTM 분류 모델 학습 스크립트.
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
CSV_PATH = "intonationpattern.csv"
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_SAVE_PATH = "pitch_lstm_model.pt"
ENCODER_SAVE_PATH = "label_encoder.pkl"
SCALER_SAVE_PATH = "scaler.pkl"


# =========================
# 데이터 로딩 & 전처리
# =========================

def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path, header=None)

    # 마지막 column이 label
    labels = df.iloc[:, -1]

    # 클래스별 개수 확인
    counts = labels.value_counts()

    # 샘플이 2개 미만인 클래스를 제거
    valid_labels = counts[counts >= 2].index
    df = df[df.iloc[:, -1].isin(valid_labels)]    

    X = df.iloc[:, :-1].values.astype(np.float32)   # (N, 128)
    y_str = df.iloc[:, -1].values                   # label 문자열

    # 문자열 라벨 → 정수 인덱스
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # feature 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler


# =========================
# PyTorch Dataset
# =========================

class PitchSequenceDataset(Dataset):
    """
    X: (N, 128)
    LSTM 입력으로는 (128, 1) 형태로 reshape 한다.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]                     # (128,)
        x = x.unsqueeze(-1)                 # (128, 1) → LSTM 입력형태
        y = self.y[idx]
        return x, y


# =========================
# LSTM 모델 정의
# =========================

class PitchLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,         # 1
            hidden_size=hidden_dim,       # 64
            num_layers=num_layers,        # 1 or 2
            batch_first=True              # (batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, 128, 1)
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]           # 마지막 time-step 출력
        logits = self.fc(out_last)
        return logits


# =========================
# 학습 루프
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item()

    return running / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            pred = torch.argmax(logits, dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

    return correct / total


# =========================
# Prediction 함수
# =========================

def predict_pattern(model, scaler, label_encoder, pitch128, device):
    model.eval()

    # scaling + reshape
    x = scaler.transform(np.array(pitch128).reshape(1, -1))
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(-1).to(device)        # (1, 128, 1)

    with torch.no_grad():
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()

    return label_encoder.inverse_transform([idx])[0]


# =========================
# main
# =========================

def main():
    print("Loading data ...")
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess(CSV_PATH)

    # Dataset
    train_ds = PitchSequenceDataset(X_train, y_train)
    test_ds = PitchSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 구성
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    plt.ion()  # interactive mode 켜기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)
    train_losses = []
    test_accuracies = []

    model = PitchLSTMClassifier(
        input_dim=1,
        hidden_dim=64,
        num_layers=1,
        num_classes=len(label_encoder.classes_)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Training LSTM model ...")
    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)

        if acc > best_acc:
            best_acc = acc

        train_losses.append(loss)
        test_accuracies.append(acc)
        print(f"[Epoch {epoch:02d}] Loss={loss:.4f}  TestAcc={acc*100:.2f}%")

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

    print(f"Best test accuracy: {best_acc*100:.2f}%")
    plt.ioff()
    plt.show()

    # 모델/인코더/스케일러 저장
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    with open(ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved model, encoder, scaler.")

    # 간단한 prediction 테스트
    raw_sample = X_test[0]
    unscaled = scaler.inverse_transform(raw_sample.reshape(1, -1))[0]

    pred = predict_pattern(model, scaler, label_encoder, unscaled, device)
    print("Example prediction:", pred)


if __name__ == "__main__":
    main()
