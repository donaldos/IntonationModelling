# Intonation Modelling
📚 README.md (Intonation Pattern MLP Classifier)
📌 프로젝트 개요
이 프로젝트는 128차원 피치 컨투어(Pitch Contour) 데이터와 이에 대응하는 **억양 패턴 레이블(LLL~HHH)**을 사용하여 PyTorch 기반의 다층 퍼셉트론(MLP) 분류 모델을 학습시키는 스크립트입니다.

학습된 모델은 새로운 128차원 피치 벡터를 입력받아 해당 피치 패턴이 어떤 억양 패턴(클래스)에 속하는지 예측하는 것을 목표로 합니다.

📂 파일 구조 및 역할
파일/경로	역할	설명
main_mlp.py	메인 학습 스크립트	데이터 로딩, 전처리, 모델 정의, 학습 및 평가, 모델 저장까지 모든 과정을 포함합니다.
intonationpattern.csv	입력 데이터	(N, 129) 형태의 CSV 파일입니다. 처음 128개 컬럼은 피치 컨투어 feature이며, 마지막 컬럼은 문자열 레이블(억양 패턴)입니다.
./mlp_model/	저장 경로	학습 완료된 모델 가중치, LabelEncoder, StandardScaler가 저장될 디렉토리입니다.

Sheets로 내보내기

⚙️ 설정값 (Configuration)
스크립트 상단에 정의된 주요 설정값입니다. 사용 환경에 맞게 수정하여 사용하십시오.

설정 변수	기본값	설명
CSV_FILE_NAME	"intonationpattern.csv"	입력 데이터 CSV 파일 이름.
BATCH_SIZE	32	한 번의 학습 단계에 사용되는 샘플 수.
NUM_EPOCHS	400	전체 데이터셋을 반복 학습할 횟수.
LEARNING_RATE	1e-3 (0.001)	Adam 최적화에 사용되는 학습률.
MODEL_SAVE_PATH	"./mlp_model/pitch_pattern_model.pt"	학습된 모델의 가중치 저장 경로.

Sheets로 내보내기

📝 데이터 로딩 및 전처리 과정
load_and_preprocess 함수에서 다음 단계를 수행합니다.

CSV 로딩: intonationpattern.csv 파일을 로드합니다.

레이블 및 Feature 분리: 마지막 컬럼을 레이블 (y)로, 나머지를 Feature (X, 128차원)로 분리합니다.

데이터 클리닝: 샘플 수가 2개 미만인 희귀 클래스는 학습의 안정성을 위해 제거합니다.

레이블 인코딩: 문자열 억양 레이블(예: 'LLL', 'HLH')을 **정수(0, 1, 2...)**로 변환합니다 (LabelEncoder).

데이터 분할: 전체 데이터를 훈련(Train) 세트와 테스트(Test) 세트로 분할합니다 (TEST_SIZE = 0.1).

Feature 스케일링: 각 Feature 차원별로 평균을 0, 표준편차를 1로 맞추는 **표준화(StandardScaler)**를 적용합니다.

🧠 모델 구조 (IntonationClassifier)
사용된 모델은 피치 컨투어 분류를 위한 다층 퍼셉트론(MLP) 구조입니다.

Layer	출력 차원	활성화 함수	설명
Input	input_dim (128)	-	128차원 피치 컨투어 벡터.
Linear	64	ReLU	첫 번째 은닉층.
Linear	64	ReLU	두 번째 은닉층.
Output	num_classes	-	최종 출력층 (클래스 개수만큼의 Logits 출력).

Sheets로 내보내기

🚀 학습 및 평가
1. 주요 구성 요소 정의
Device: cuda, mps (Apple M1/M2), cpu 중 사용 가능한 환경으로 자동 설정됩니다.

모델: IntonationClassifier(input_dim, num_classes).to(device)로 초기화됩니다.

손실 함수 (criterion): nn.CrossEntropyLoss()를 사용합니다. 분류 문제에 적합하며, 내부적으로 Softmax와 NLL Loss 연산을 포함합니다.

최적화 함수 (optimizer): optim.Adam(model.parameters(), lr=LEARNING_RATE)를 사용하여 모델의 가중치를 업데이트합니다.

2. 학습 루프 (train_one_epoch)
model.train(): 모델을 훈련 모드로 설정합니다 (Dropout, BatchNorm 활성화).

optimizer.zero_grad(): 필수! 이전 배치에서 계산된 경사(Gradient)를 초기화합니다.

loss.backward(): 손실을 기준으로 모든 매개변수에 대한 경사를 계산합니다.

optimizer.step(): 계산된 경사를 사용하여 Adam 알고리즘으로 매개변수를 업데이트합니다.

3. 평가 (evaluate)
model.eval(): 모델을 평가 모드로 설정합니다 (Dropout, BatchNorm 비활성화).

with torch.no_grad(): 경사 계산을 비활성화하여 메모리를 절약하고 속도를 높입니다.

예측 결과와 실제 레이블을 비교하여 **정확도(Accuracy)**를 계산합니다.

4. 학습 결과 시각화
학습 과정 중 손실(Loss)과 정확도(Accuracy) 변화가 matplotlib을 통해 실시간으로 그래프로 표시됩니다.

💾 모델 저장 및 예측
학습이 완료된 후, 모델 재사용 및 예측을 위해 다음 파일들이 MODEL_SAVE_PATH에 저장됩니다.

모델 가중치: torch.save(model.state_dict(), ...)

레이블 인코더: pickle.dump(label_encoder, ...)

스케일러: pickle.dump(scaler, ...)

저장된 이 스케일러와 인코더는 새로운 데이터를 예측할 때 반드시 사용해야 합니다. predict_pattern 함수는 저장된 스케일러와 인코더를 사용하여 새로운 128차원 피치 벡터를 전처리하고 최종 억양 레이블을 반환합니다.

Python

# 샘플 예측
pred_label = predict_pattern(model, scaler, label_encoder, raw_sample, device)
print(f"Sample prediction: {pred_label}")