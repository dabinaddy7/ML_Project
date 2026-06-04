import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score


# 데이터 디렉토리. 본인 환경에 맞게 수정하세요.
DATA_DIR = "./data"


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_split(name, data_dir=DATA_DIR):
    """
    하나의 split CSV를 로드합니다.

    Parameters
    ----------
    name : str
        "train", "val", "test_public", "test_hidden_no_labels" 중 하나.
    data_dir : str
        CSV들이 있는 디렉토리 경로.

    Returns
    -------
    df : pd.DataFrame
        timestep + feature 컬럼 (label은 분리되어 있음)
    feature_cols : list[str]
        feature 컬럼 이름들 (x_로 시작하는 것들)
    labels : np.ndarray | None
        timestep별 라벨 (0=정상, 1=이상). 라벨이 없는 split이면 None.
    """
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


# ============================================================
# 2. Sliding window
# ============================================================

def make_windows(values, window_size, stride=1):
    """
    시계열을 sliding window로 변환합니다.

    Parameters
    ----------
    values : np.ndarray
        shape (T,) 또는 (T, D)
    window_size : int
    stride : int

    Returns
    -------
    windows : np.ndarray
        - 입력이 (T,)면 출력은 (N, window_size)
        - 입력이 (T, D)면 출력은 (N, window_size, D)
        N = (T - window_size) // stride + 1
    """
    values = np.asarray(values)
    T = values.shape[0]
    if T < window_size:
        raise ValueError(f"입력 길이({T})가 window_size({window_size})보다 짧습니다.")

    n = (T - window_size) // stride + 1
    if values.ndim == 1:
        out = np.stack([values[i*stride : i*stride + window_size]
                        for i in range(n)])
    elif values.ndim == 2:
        out = np.stack([values[i*stride : i*stride + window_size, :]
                        for i in range(n)])
    else:
        raise ValueError(f"지원하지 않는 차원: {values.ndim}")
    return out


def windows_to_timestep_scores(window_scores, T, window_size, stride=1):
    """
    window별 score를 timestep별 score로 환산합니다.

    가장 단순한 방식: window의 score를 그 window의 마지막 timestep에 할당.
    첫 (window_size - 1) timestep은 첫 window의 score로 패딩.
    중간에 빈 timestep이 있으면 forward-fill로 채움.

    이 변환 방식은 baseline일 뿐입니다. 더 나은 방식 (예: window 중심에 할당,
    겹치는 window들의 평균 등)을 직접 구현해보세요.

    Parameters
    ----------
    window_scores : np.ndarray, shape (N,)
        각 window의 anomaly score
    T : int
        원래 시계열 길이
    window_size : int
    stride : int

    Returns
    -------
    timestep_scores : np.ndarray, shape (T,)
    """
    timestep_scores = np.full(T, np.nan)
    n_windows = len(window_scores)

    # 각 window의 score를 그 window의 마지막 timestep에 할당
    for i in range(n_windows):
        end_idx = i * stride + window_size - 1
        timestep_scores[end_idx] = window_scores[i]

    # 앞쪽 패딩 (첫 window가 끝나기 전 구간)
    timestep_scores[:window_size - 1] = window_scores[0]

    # stride > 1인 경우 중간에 nan이 남을 수 있어 forward-fill
    for i in range(1, T):
        if np.isnan(timestep_scores[i]):
            timestep_scores[i] = timestep_scores[i - 1]

    return timestep_scores


# ============================================================
# 3. Baseline 파이프라인
# ============================================================

if __name__ == "__main__":
    # ---------- 데이터 로드 ----------
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")

    print("=== 데이터 형태 ===")
    print(f"train:        {train_df.shape}, anomaly=없음 (정상만)")
    print(f"val:          {val_df.shape}, anomaly={val_labels.sum()}개 timestep")
    print(f"feature_cols: {feature_cols}")
    print()

    # ---------- 전처리: 스케일링 ----------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])

    # ---------- Sliding window ----------
    W = 30   
    S = 1    

    train_windows = make_windows(X_train, W, S)  # (N, W, D)
    val_windows   = make_windows(X_val,   W, S)

    # 300차원 Flatten 전처리 유지
    train_X = train_windows.reshape(len(train_windows), -1)
    val_X   = val_windows.reshape(len(val_windows), -1)

    print(f"=== Sliding window (W={W}, stride={S}) ===")
    print(f"train_X: {train_X.shape}")
    print(f"val_X:   {val_X.shape}\n")

    # ---------- 🎯 하이퍼파라미터 튜닝 (Grid Search) ----------
    # 오직 val 데이터의 성능만 보고 최적의 조합을 선택합니다 (교수님 가이드 반영)
    from sklearn.model_selection import ParameterGrid

    param_grid = {
        'contamination': [0.001, 0.005, 0.01, 0.02, 0.05], 
        'n_estimators': [100, 200, 300]                     
    }

    best_val_aupr = -1
    best_val_auroc = -1  # 최적 파라미터 시점의 AUROC 저장을 위한 변수 추가
    best_params = {}

    print("=== 🔍 Isolation Forest Grid Search 시작 (Val 기준 튜닝) ===")
    for params in ParameterGrid(param_grid):
        # 모델 정의 및 학습
        model = IsolationForest(
            n_estimators=params['n_estimators'],
            contamination=params['contamination'],
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_X)

        # 검증 데이터 스코어 계산 및 부호 반전
        val_window_scores = -model.score_samples(val_X)
        val_scores_timestep = windows_to_timestep_scores(val_window_scores, len(val_df), W, S)

        # 성능 평가 (AUROC, AUPR)
        current_val_auroc = roc_auc_score(val_labels, val_scores_timestep)
        current_val_aupr  = average_precision_score(val_labels, val_scores_timestep)

        print(f"[후보] {params} -> Val AUROC: {current_val_auroc:.4f} | Val AUPR: {current_val_aupr:.4f}")

        # 최적의 모델 갱신 (핵심 지표인 AUPR 기준)
        if current_val_aupr > best_val_aupr:
            best_val_aupr = current_val_aupr
            best_val_auroc = current_val_auroc  # 최적 시점의 AUROC 기록
            best_params = params

    # ⭐️ 요청사항 반영: 최적 세트와 함께 val의 AUROC, AUPR을 표 형태로 명시적 출력
    print("\n==================================================")
    print(f"🏆 최적 파라미터 선정 결과: {best_params}")
    print("==================================================")
    print(f"{'':15s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val (Best)':15s} {best_val_auroc:>8.4f} {best_val_aupr:>8.4f}")
    print("==================================================")
    print("➔ 최적 하이퍼파라미터 탐색 완료. 이 파라미터를 최종 실행 코드에 적용하세요.")