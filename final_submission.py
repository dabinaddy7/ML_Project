import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

# ============================================================
# 1. 데이터 로드 및 헬퍼 함수 정의
# ============================================================
DATA_DIR = "./data"

def load_split(name, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{name}.csv")
    raw = pd.read_csv(path)
    feature_cols = [c for c in raw.columns if c.startswith("x_")]
    if "label" in raw.columns:
        df = raw.drop(columns=["label"])
    else:
        df = raw
    return df, feature_cols

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
# 2. 메인 파이프라인 실행
# ============================================================
if __name__ == "__main__":
    # 프로젝트 공통 하이퍼파라미터 세팅
    W = 30
    S = 1
    
    print("=== [1] 데이터 로드 및 모델 맞춤형 전처리 ===")
    train_df, feature_cols = load_split("train")
    hidden_df, _ = load_split("test_hidden_no_labels")
    
    # 1) 전체 데이터 스케일링 (Data Leakage 완벽 방지)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_hidden = scaler.transform(hidden_df[feature_cols])
    
    # 2) Sliding Window 변환 ($W=30, S=1$)
    train_windows = make_windows(X_train, W, S)
    hidden_windows = make_windows(X_hidden, W, S)
    
    # 3) GMM & LOF 공통 특화 피처 추출 (평균 + 표준편차 = 20차원)
    train_X = np.hstack([np.mean(train_windows, axis=1), np.std(train_windows, axis=1)])
    hidden_X = np.hstack([np.mean(hidden_windows, axis=1), np.std(hidden_windows, axis=1)])
    
    print(f"➔ 전처리 완료 | Train 형상: {train_X.shape} | Hidden 형상: {hidden_X.shape}")

    print("\n=== [2] 최적 싱글 모델 독립 학습 및 히든 스코어 추정 ===")
    # 1️⃣ GMM 최종 최적 파라미터 기반 추정
    print("➔ GMM 최종 모델 학습 및 스코어 계산 중...")
    gmm = GaussianMixture(n_components=20, covariance_type='full', random_state=42)
    gmm.fit(train_X)
    hidden_win_scores_gmm = -gmm.score_samples(hidden_X)
    hidden_scores_gmm = windows_to_timestep_scores(hidden_win_scores_gmm, len(hidden_df), W, S)
    
    # 2️⃣ LOF 최종 최적 파라미터 기반 추정
    print("➔ LOF 최종 모델 학습 및 스코어 계산 중...")
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.001, novelty=True, n_jobs=-1)
    lof.fit(train_X)
    hidden_win_scores_lof = -lof.score_samples(hidden_X)
    hidden_scores_lof = windows_to_timestep_scores(hidden_win_scores_lof, len(hidden_df), W, S)

    print("\n=== [3] Rank-based 앙상블 융합 (GMM 0.75 : LOF 0.25) ===")
    # 스케일 불일치 문제를 해결하기 위해 순위 분위수화 적용
    hidden_gmm_rank = rankdata(hidden_scores_gmm) / len(hidden_scores_gmm)
    hidden_lof_rank = rankdata(hidden_scores_lof) / len(hidden_scores_lof)
    
    # 도출된 최적 가중치로 최종 결합
    final_ensemble_score = (0.75 * hidden_gmm_rank) + (0.25 * hidden_lof_rank)

    print("\n=== [4] 최종 제출용 결과 CSV 파일 생성 ===")
    submission_df = pd.DataFrame({
        't': hidden_df['t'],
        'score': final_ensemble_score
    })
    
    # 가이드라인에 명시된 포맷으로 결과 저장
    output_filename = "test_hidden_no_labels_result.csv"
    submission_df.to_csv(output_filename, index=False)
    
    print("==================================================")
    print(f"🏆 최종 파일 생성 완료: '{output_filename}'")
    print(f"➔ 총 {len(submission_df)}개 타임스탬프 스코어가 성공적으로 추출되었습니다.")
    print("==================================================")