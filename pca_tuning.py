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
# 🎯 PCA 하이퍼파라미터 튜닝 시작
# ============================================================
if __name__ == "__main__":
    W = 30
    S = 1

    print("=== [1] 데이터 로드 및 스케일링 ===")
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])

    print(f"=== [2] Sliding Window (W={W}) 및 Flatten(300차원) 전처리 ===")
    train_windows = make_windows(X_train, W, S)
    val_windows   = make_windows(X_val,   W, S)

    # 선형적 관계 압축을 위해 300차원 구조 유지
    train_X = train_windows.reshape(len(train_windows), -1)
    val_X   = val_windows.reshape(len(val_windows), -1)

    print(f"=== Sliding window (W={W}, stride={S}) ===")
    print(f"train_X: {train_X.shape}")
    print(f"val_X:   {val_X.shape}\n")

    # ---------- 🎯 하이퍼파라미터 튜닝 (Grid Search) ----------
    # n_components: 보존할 주성분 축의 개수 후보군 (300차원 중 일부 선택)
    param_grid = {
        'n_components': [2, 5, 10, 20, 50, 100]
    }

    best_val_aupr = -1
    best_val_auroc = -1
    best_params = {}

    print("=== 🔍 PCA Reconstruction Error Grid Search 시작 (Val 기준 튜닝) ===")
    for n_comp in param_grid['n_components']:
        
        # PCA 모델 학습 (정상 데이터로만 주성분 축을 학습)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(train_X)

        # 🛠️ 재구성 오차(Reconstruction Error) 계산 과정
        # 1) 압축 (Transform)
        val_X_compressed = pca.transform(val_X)
        # 2) 복원 (Inverse Transform)
        val_X_reconstructed = pca.inverse_transform(val_X_compressed)
        # 3) 원본과 복원본 사이의 유클리디안 거리를 아노말리 스코어로 사용
        val_window_scores = np.mean((val_X - val_X_reconstructed) ** 2, axis=1)

        # 타임스탬프 스코어로 환산
        val_scores_timestep = windows_to_timestep_scores(val_window_scores, len(val_df), W, S)

        # 성능 평가
        current_val_auroc = roc_auc_score(val_labels, val_scores_timestep)
        current_val_aupr  = average_precision_score(val_labels, val_scores_timestep)

        print(f"[후보] n_components: {n_comp} -> Val AUROC: {current_val_auroc:.4f} | Val AUPR: {current_val_aupr:.4f}")

        # AUPR 기준 최적 모델 갱신
        if current_val_aupr > best_val_aupr:
            best_val_aupr = current_val_aupr
            best_val_auroc = current_val_auroc
            best_params = {'n_components': n_comp}

    print("\n==================================================")
    print(f"🏆 최적 파라미터 선정 결과: {best_params}")
    print("==================================================")
    print(f"{'':15s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val (Best)':15s} {best_val_auroc:>8.4f} {best_val_aupr:>8.4f}")
    print("==================================================")
    print("➔ 최적 하이퍼파라미터 탐색 완료. 이 파라미터를 최종 실행 코드에 적용하세요.")