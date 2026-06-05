import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata

# ============================================================
# 데이터 디렉토리 설정
# ============================================================
DATA_DIR = "./data"

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

if __name__ == "__main__":
    # 보고서에 작성한 최종 설정값
    W = 30
    S = 1
    W_GMM = 0.75
    W_LOF = 0.25

    print("[1] 히든 테스트 데이터 로드")
    test_hidden = pd.read_csv(os.path.join(DATA_DIR, "test_hidden_no_labels.csv"))
    feature_cols = [c for c in test_hidden.columns if c.startswith("x_")]

    print("[2] 학습 완료된 GMM, LOF 모델 및 스케일러 불러오기")
    gmm_scaler = joblib.load("models/gmm_scaler.pkl")
    gmm_model = joblib.load("models/gmm_model.pkl")

    lof_scaler = joblib.load("models/lof_scaler.pkl")
    lof_model = joblib.load("models/lof_model.pkl")

    print("[3] 데이터 전처리 (스케일링 ➔ Sliding Window ➔ 피처 추출)")
    # GMM 피처 처리
    X_hidden_gmm = gmm_scaler.transform(test_hidden[feature_cols])
    hidden_windows_gmm = make_windows(X_hidden_gmm, W, S)
    hidden_X_gmm = np.hstack([np.mean(hidden_windows_gmm, axis=1), np.std(hidden_windows_gmm, axis=1)])

    # LOF 피처 처리
    X_hidden_lof = lof_scaler.transform(test_hidden[feature_cols])
    hidden_windows_lof = make_windows(X_hidden_lof, W, S)
    hidden_X_lof = np.hstack([np.mean(hidden_windows_lof, axis=1), np.std(hidden_windows_lof, axis=1)])

    print("[4] 모델별 Anomaly Score 추론")
    gmm_window_scores = -gmm_model.score_samples(hidden_X_gmm)
    lof_window_scores = -lof_model.score_samples(hidden_X_lof)

    gmm_scores = windows_to_timestep_scores(gmm_window_scores, len(test_hidden), W, S)
    lof_scores = windows_to_timestep_scores(lof_window_scores, len(test_hidden), W, S)

    print("[5] Rank-based 스케일링 및 최적 가중치(0.75 : 0.25) 앙상블 결합")
    gmm_r = rankdata(gmm_scores) / len(gmm_scores)
    lof_r = rankdata(lof_scores) / len(lof_scores)

    final_ensemble_scores = (W_GMM * gmm_r) + (W_LOF * lof_r)

    print("[6] 제출용 CSV 파일 생성")
    submission_df = pd.DataFrame({
        't': test_hidden['t'],
        'score': final_ensemble_scores
    })

    # 제출 파일 규격: 학번_이름_submission.csv
    out_name = "202201519_임다빈_submission.csv"
    submission_df.to_csv(out_name, index=False)
    print(f"➔ 🏆 최종 제출 파일 생성 완료: {out_name}")