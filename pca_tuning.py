import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    return np.stack([values[i*stride : i*stride + window_size, :] for i in range(n)])

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
# PCA 하이퍼파라미터 튜닝 파이프라인
# ============================================================
if __name__ == "__main__":
    print("[1] 데이터 로드 및 스케일링 (Train, Val)")
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")
    # Data Leakage 방지를 위해 튜닝 시 test 데이터는 로드하지 않음

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled   = scaler.transform(val_df[feature_cols])

    print("[2] 하이퍼파라미터 탐색 설정")
    W = 30
    S = 1
    n_components_list = [2, 5, 10, 20, 50, 100]

    best_aupr = 0
    best_params = {}

    print(f"[3] Sliding Window (W={W}) 및 300차원 원본 유지(Flatten) 전처리")
    train_windows = make_windows(X_train_scaled, W, stride=S)
    val_windows   = make_windows(X_val_scaled,   W, stride=S)
    
    # 윈도우(W) 내의 채널을 1차원으로 길게 폅니다.
    train_X = train_windows.reshape(train_windows.shape[0], -1)
    val_X   = val_windows.reshape(val_windows.shape[0], -1)

    print("[4] Grid Search 진행")
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(train_X)

        # 재구성 오차(Reconstruction Error) 계산: 원본 - 복원본
        val_X_compressed = pca.transform(val_X)
        val_X_reconstructed = pca.inverse_transform(val_X_compressed)
        val_window_scores = np.mean((val_X - val_X_reconstructed) ** 2, axis=1)

        val_scores_timestep = windows_to_timestep_scores(val_window_scores, len(val_df), W, S)

        val_auroc = roc_auc_score(val_labels, val_scores_timestep)
        val_aupr  = average_precision_score(val_labels, val_scores_timestep)

        print(f" - n_components: {n_comp:<3} -> Val AUROC: {val_auroc:.4f} | Val AUPR: {val_aupr:.4f}")

        if val_aupr > best_aupr:
            best_aupr = val_aupr
            best_params = {'n_components': n_comp}

    print("\n[최종 결과] PCA 최적 파라미터:", best_params)
    print(f"[최종 결과] 최고 Val AUPR: {best_aupr:.4f}")