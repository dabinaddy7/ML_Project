import os
import numpy as np
import pandas as pd
import joblib  # 학습된 모델을 파일로 저장하기 위한 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 데이터 디렉토리 및 헬퍼 함수
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
# GMM 최종 모델 실행 파이프라인
# ============================================================
if __name__ == "__main__":
    # 우리가 찾은 최적의 하이퍼파라미터 고정!
    W = 30
    S = 1
    N_COMPONENTS = 20
    COV_TYPE = 'full'

    print("=== [1] 데이터 로드 및 스케일링 ===")
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")
    test_df,  _, test_labels  = load_split("test_public")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])

    print(f"=== [2] Sliding Window (W={W}) & 피처 추출 ===")
    train_windows = make_windows(X_train, W, S)
    val_windows   = make_windows(X_val,   W, S)
    test_windows  = make_windows(X_test,  W, S)

    train_X = np.hstack([np.mean(train_windows, axis=1), np.std(train_windows, axis=1)])
    val_X   = np.hstack([np.mean(val_windows, axis=1), np.std(val_windows, axis=1)])
    test_X  = np.hstack([np.mean(test_windows, axis=1), np.std(test_windows, axis=1)])

    print("=== [3] 최종 GMM 모델 학습 중 ===")
    model = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COV_TYPE, random_state=42)
    model.fit(train_X)
    print("학습 완료")

    print("=== [4] Anomaly Score 계산 및 저장 ===")
    val_window_scores  = -model.score_samples(val_X)
    test_window_scores = -model.score_samples(test_X)

    val_scores  = windows_to_timestep_scores(val_window_scores,  len(val_df),  W, S)
    test_scores = windows_to_timestep_scores(test_window_scores, len(test_df), W, S)

    val_auroc  = roc_auc_score(val_labels,  val_scores)
    val_aupr   = average_precision_score(val_labels,  val_scores)
    test_auroc = roc_auc_score(test_labels, test_scores)
    test_aupr  = average_precision_score(test_labels, test_scores)

    print("\n🏆 === [GMM 최종] 앙상블 대기 준비 완료 ===")
    print(f"{'':12s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val':12s} {val_auroc:>8.4f} {val_aupr:>8.4f}")
    print(f"{'test_public':12s} {test_auroc:>8.4f} {test_aupr:>8.4f}")

   # ------------------------------------------------------------
    # ⭐️ 눈으로 확인하기 쉬운 CSV 엑셀 파일로 저장
    # ------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("scores", exist_ok=True)

    # 1. 스케일러와 모델 저장 (.pkl은 파이썬 전용 모델 파일이므로 그대로 둠)
    joblib.dump(scaler, "models/gmm_scaler.pkl")
    joblib.dump(model, "models/gmm_model.pkl")

    # 2. DataFrame을 만들어 t(타임스텝)와 모델 점수를 예쁘게 짝지어줍니다.
    val_result_df = pd.DataFrame({
        't': val_df['t'],
        'GMM_score': val_scores
    })
    
    test_result_df = pd.DataFrame({
        't': test_df['t'],
        'GMM_score': test_scores
    })

    # 3. CSV 파일로 저장 (index=False를 해야 쓸데없는 번호 열이 안 생깁니다)
    val_result_df.to_csv("scores/gmm_val_scores.csv", index=False)
    test_result_df.to_csv("scores/gmm_test_scores.csv", index=False)
    
    print("\n➔ 모델 가중치와 Anomaly Score(CSV)가 안전하게 저장되었습니다.")