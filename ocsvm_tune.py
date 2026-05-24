import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 데이터 로드 및 헬퍼 함수
# ============================================================
DATA_DIR = "./data"

def load_split(name, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{name}.csv")
    raw = pd.read_csv(path)
    feature_cols = [c for c in raw.columns if c.startswith("x_")]
    if "label" in raw.columns:
        labels = raw["label"].to_numpy().astype(int)
        df = raw.drop(columns=["label"])
    else:
        labels = None
        df = raw
    return df, feature_cols, labels

def make_windows(values, window_size, stride=1):
    values = np.asarray(values)
    T = values.shape[0]
    n = (T - window_size) // stride + 1
    out = np.stack([values[i*stride : i*stride + window_size, :] for i in range(n)])
    return out

def windows_to_timestep_scores(window_scores, T, window_size, stride=1):
    timestep_scores = np.full(T, np.nan)
    n_windows = len(window_scores)
    for i in range(n_windows):
        end_idx = i * stride + window_size - 1
        timestep_scores[end_idx] = window_scores[i]
    timestep_scores[:window_size - 1] = window_scores[0]
    for i in range(1, T):
        if np.isnan(timestep_scores[i]):
            timestep_scores[i] = timestep_scores[i - 1]
    return timestep_scores

# ============================================================
# OC-SVM 파라미터 튜닝 파이프라인
# ============================================================
if __name__ == "__main__":
    # GMM에서 가장 성능이 좋았던 W=30으로 고정
    W = 30
    S = 1

    print("=== [1] 데이터 로드 및 스케일링 (StandardScaler) ===")
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")

    # OC-SVM은 거리 기반 모델이므로 스케일링이 생명입니다!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])

    print(f"=== [2] Sliding Window (W={W}) & 피처 추출 ===")
    train_windows = make_windows(X_train, W, S)
    val_windows   = make_windows(X_val,   W, S)

    # 차원의 저주를 피하기 위해 20차원으로 축소
    train_X = np.hstack([np.mean(train_windows, axis=1), np.std(train_windows, axis=1)])
    val_X   = np.hstack([np.mean(val_windows, axis=1), np.std(val_windows, axis=1)])

    # 탐색할 하이퍼파라미터 목록
    nu_list = [0.01, 0.05, 0.1, 0.2]
    gamma_list = ['scale', 'auto', 0.01, 0.1]

    best_aupr = 0
    best_params = {}

    print("\n=== [본격 실험] OC-SVM 하이퍼파라미터 탐색 시작 ===")
    for nu in nu_list:
        for gamma in gamma_list:
            # RBF 커널을 사용하여 고차원 울타리 생성
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            model.fit(train_X)

            # OC-SVM의 score_samples: 정상일수록 값이 큼. 이상치 점수화하기 위해 (-) 부호 붙임
            val_window_scores = -model.score_samples(val_X)
            val_scores = windows_to_timestep_scores(val_window_scores, len(val_df), W, S)

            val_auroc = roc_auc_score(val_labels, val_scores)
            val_aupr  = average_precision_score(val_labels, val_scores)

            print(f"nu: {nu:<4} | gamma: {str(gamma):<5} ➔ Val AUROC: {val_auroc:.4f} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_aupr:
                best_aupr = val_aupr
                best_params = {'nu': nu, 'gamma': gamma}

    print("\n🏆 [OC-SVM 튜닝] 최고 성능 파라미터:", best_params)
    print(f"🏆 [OC-SVM 튜닝] 최고 Val AUPR: {best_aupr:.4f}")