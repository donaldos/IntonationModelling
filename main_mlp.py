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
import sys

# =========================
# 설정값
# =========================
CSV_FILE_NAME = "intonationpattern.csv"   # <-- 여기에 실제 csv 파일 이름 넣으세요
BATCH_SIZE = 32
NUM_EPOCHS = 400
LEARNING_RATE = 1e-3
TEST_SIZE = 0.1
RANDOM_STATE = 42
MODEL_SAVE_PATH = "./mlp_model/pitch_pattern_model.pt"
ENCODER_SAVE_PATH = "./mlp_model/label_encoder.pkl"
SCALER_SAVE_PATH = "./mlp_model/scaler.pkl"


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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE#, stratify=y
    )

    # feature 스케일링 (평균 0, 분산 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler    


# =========================
# PyTorch Dataset
# =========================
class IntonationDataset(Dataset):
    """
    np.ndarray → torch.Tensor 변환을 자동 처리하는 Dataset.
    
    옵션:
        reshape_seq : True → (length,) → (length, 1)로 변환 (LSTM용)
        device      : 'cpu', 'cuda', 'mps' 중 선택 가능.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 reshape_seq: bool = False, 
                 device: str = 'cpu'):
        
        # numpy → tensor 변환
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.reshape_seq = reshape_seq

        # device 설정 ('cpu', 'cuda', 'mps')
        self.device = torch.device(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # LSTM용 reshape (128,) → (128,1)
        if self.reshape_seq:
            x = x.unsqueeze(-1)     # (128,) → (128,1)

        # 필요한 경우 device로 자동 이동
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y



# =========================
# 모델 정의 (MLP)
# =========================
class IntonationClassifier_01(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
class IntonationClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        #input → [Linear → ReLU] → [Linear → ReLU] → Output
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # FC Layer (Input → Hidden)
            nn.ReLU(),                 # Activation: 비선형활성화 함수
            nn.Linear(64, 64),         # FC Layer (Hidden → Hidden)
            nn.ReLU(),                 # Activation: 비선형 활성화 함수
            nn.Linear(64, num_classes) # FC Layer (Hidden → Output)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 학습 & 평가 루프
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    train_one_epoch() 함수는 DataLoader로부터 batch 단위 데이터를 받아
    forward → loss 계산 → backward → optimizer 업데이트
    """

    # 훈련 시 배치마다 달라져야 하는 레이어(Dropout, BatchNorm)를 제대로 작동시키는 필수 호출
    # 하지않으면, 
    # - Dropout이 꺼져서 더 이상 regularization이 되지 않음
    # - BatchNorm이 running_mean / running_var (평가용 값)만 사용해 학습이 망가짐
    # - Gradient 계산도 잘못될 수 있음
    model.train()
    
    running_loss = 0.0

    # dataloader 생성시, BATCH_SIZE 정의
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 반드시 backward 전에 호출해야 함.
        # PyTorch는 gradient를 누적시키므로,
        # 초기화 안 하면 여러 batch의 gradient가 섞여버림 → 오작동.
        optimizer.zero_grad()

        # forward 계산
        outputs = model(X_batch)            # (batch, num_classes)
        # loss 계산
        loss = criterion(outputs, y_batch)  # 정의된 손실함수

        # 출력에서 손실을 기준으로 모든 파라미터에 대해 ∂Loss/∂W 를 자동 계산(Autograd)
        loss.backward()     
        # 계산된 gradient로 실제 파라미터 업데이트(학습)                
        optimizer.step()                    

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    # Dropout OFF
    # BatchNorm 평가모드(fixed running mean/var 사용)
    # 실수 예측의 안정성 보장
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
    csv_file_path = f"/Users/donaldos/workspace/autointonation/IntonationModelling/input/{CSV_FILE_NAME}"
    print(f"Loading data from {csv_file_path} ...")
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess(csv_file_path)

    # X_train.shape = (4791, 128) 
    # 이 중 입력차원은 128, X_train.shape[0]은 데이터의 갯수
    input_dim = X_train.shape[1]
    # LLL ~ HHH  중 나타난 것이 약 20개 (즉 7개는 빈도 차이로 drop 시켰음)
    num_classes = len(label_encoder.classes_)
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")


    # 2) Dataset / DataLoader
    # - numpy.ndarray --> tensor
    train_dataset = IntonationDataset(X_train, y_train, reshape_seq=False, device="mps")
    test_dataset = IntonationDataset(X_test, y_test, reshape_seq=False, device="mps")
    
    

    # batch_size는 한번에 훈련할때, 몇개의 샘플을 동시에 집어 넣을지 정의
    # - 모든데이터를 한번에 처리는 불가능하기 때문에 GPU/CPU 상황에 맞게 올릴 수 있는 단위로 나누는것
    # - 학습 안정성 개선: 균현잡힌 gradient (작은 데이터는 불안정, 큰 데이터는 학습이 느려짐)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3) 모델 / 손실함수 / 옵티마이저
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Using device:", device)

    plt.ion()  # interactive mode 켜기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)
    train_losses = []
    test_accuracies = []

    # 모델 설정
    model = IntonationClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    # 손실함수 정의-교차엔트로피손실: 분류의 문제에서 모델의 출력과 실제 정답 레이블간의 오차를 계산
    criterion = nn.CrossEntropyLoss()
    # 최적화 알고리즘 선정 
    # - "**model**에 포함된 모든 학습 가능한 가중치들을 **LEARNING_RATE**로 시작하는 Adam 최적화 알고리즘을 사용하여 **업데이트(학습)**하겠다."
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4) 학습 루프
    best_test_acc = 0.0

    # epoch별 train
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
