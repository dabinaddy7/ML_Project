import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
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
# 🎯 LOF 하이퍼파라미터 튜닝 시작
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

    print(f"=== [2] Sliding Window (W={W}) 및 20차원 통계량 전처리 ===")
    train_windows = make_windows(X_train, W, S)
    val_windows   = make_windows(X_val,   W, S)

    # LOF의 거리 계산 왜곡을 막기 위해 20차원(평균, 표준편차) 압축 적용
    train_X = np.hstack([np.mean(train_windows, axis=1), np.std(train_windows, axis=1)])
    val_X   = np.hstack([np.mean(val_windows, axis=1), np.std(val_windows, axis=1)])

    print(f"=== Sliding window (W={W}, stride={S}) ===")
    print(f"train_X: {train_X.shape}")
    print(f"val_X:   {val_X.shape}\n")

    # ---------- 🎯 하이퍼파라미터 튜닝 (Grid Search) ----------
    # n_neighbors: 이웃의 개수 후보군
    # contamination: 이상치 비율 후보군
    # novelty=True 옵션을 켜야 train으로만 fit하고 val을 predict할 수 있습니다!
    param_grid = {
        'n_neighbors': [10, 20, 50, 100],
        'contamination': [0.001, 0.01, 0.05]
    }

    best_val_aupr = -1
    best_val_auroc = -1
    best_params = {}

    print("=== 🔍 Local Outlier Factor Grid Search 시작 (Val 기준 튜닝) ===")
    for n_neigh in param_grid['n_neighbors']:
        for cont in param_grid['contamination']:
            
            # 모델 정의 및 학습
            model = LocalOutlierFactor(
                n_neighbors=n_neigh,
                contamination=cont,
                novelty=True, # 중요: 새로운 데이터(val)를 평가하기 위해 필수 설정
                n_jobs=-1
            )
            model.fit(train_X)

            # LOF의 score_samples도 "정상일수록 큰 값(-)"을 주므로 부호 반전
            val_window_scores = -model.score_samples(val_X)
            val_scores_timestep = windows_to_timestep_scores(val_window_scores, len(val_df), W, S)

            # 성능 평가
            current_val_auroc = roc_auc_score(val_labels, val_scores_timestep)
            current_val_aupr  = average_precision_score(val_labels, val_scores_timestep)

            print(f"[후보] n_neighbors: {n_neigh}, contamination: {cont} -> Val AUROC: {current_val_auroc:.4f} | Val AUPR: {current_val_aupr:.4f}")

            # AUPR 기준 최적 모델 갱신
            if current_val_aupr > best_val_aupr:
                best_val_aupr = current_val_aupr
                best_val_auroc = current_val_auroc
                best_params = {'n_neighbors': n_neigh, 'contamination': cont}

    print("\n==================================================")
    print(f"🏆 최적 파라미터 선정 결과: {best_params}")
    print("==================================================")
    print(f"{'':15s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val (Best)':15s} {best_val_auroc:>8.4f} {best_val_aupr:>8.4f}")
    print("==================================================")
    print("➔ 최적 하이퍼파라미터 탐색 완료. 이 파라미터를 최종 실행 코드에 적용하세요.")