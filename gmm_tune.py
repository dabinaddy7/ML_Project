import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
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
# GMM 하이퍼파라미터 튜닝 파이프라인
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
    n_components_list = [5, 10, 15, 20]
    covariance_types = ['full', 'tied', 'diag']

    best_aupr = 0
    best_params = {}

    print(f"[3] Sliding Window (W={W}) 및 차원 압축 피처 추출")
    train_windows = make_windows(X_train_scaled, W, stride=1)
    val_windows   = make_windows(X_val_scaled,   W, stride=1)
    
    train_X = np.hstack([np.mean(train_windows, axis=1), np.std(train_windows, axis=1)])
    val_X   = np.hstack([np.mean(val_windows, axis=1), np.std(val_windows, axis=1)])

    print("[4] Grid Search 진행")
    for n_comp in n_components_list:
        for cov_type in covariance_types:
            model = GaussianMixture(
                n_components=n_comp, 
                covariance_type=cov_type, 
                random_state=42,
                max_iter=100
            )
            model.fit(train_X)

            val_window_scores = -model.score_samples(val_X)
            val_scores = windows_to_timestep_scores(val_window_scores, len(val_df), W, stride=1)
            
            val_auroc = roc_auc_score(val_labels, val_scores)
            val_aupr  = average_precision_score(val_labels, val_scores)

            print(f" - n_comp: {n_comp:2d} | cov_type: {cov_type:5s} -> Val AUROC: {val_auroc:.4f} | Val AUPR: {val_aupr:.4f}")

            if val_aupr > best_aupr:
                best_aupr = val_aupr
                best_params = {'n_components': n_comp, 'covariance_type': cov_type}

    print("\n[최종 결과] GMM 최적 파라미터:", best_params)
    print(f"[최종 결과] 최고 Val AUPR: {best_aupr:.4f}")