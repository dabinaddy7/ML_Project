import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 1. 데이터 로드 및 헬퍼 함수
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
# 2. PCA 최종 모델 실행
# ============================================================
if __name__ == "__main__":
    W = 30
    S = 1
    
    # 🔍 튜닝 단계에서 찾아낸 최적의 하이퍼파라미터 고정
    BEST_N_COMPONENTS = 50

    print("=== [1] 데이터 로드 및 스케일링 ===")
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")
    test_df,  _, test_labels  = load_split("test_public")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])

    print(f"=== [2] Sliding Window (W={W}) 및 Flatten(300차원) 전처리 ===")
    train_windows = make_windows(X_train, W, S)
    val_windows   = make_windows(X_val,   W, S)
    test_windows  = make_windows(X_test,  W, S)

    # 300차원 구조 유지
    train_X = train_windows.reshape(len(train_windows), -1)
    val_X   = val_windows.reshape(len(val_windows), -1)
    test_X  = test_windows.reshape(len(test_windows), -1)

    print("=== [3] 확정된 최적 파라미터로 PCA 최종 학습 ===")
    model = PCA(n_components=BEST_N_COMPONENTS, random_state=42)
    model.fit(train_X)

    print("=== [4] 재구성 오차(Reconstruction Error) 계산 및 타임스탬프 변환 ===")
    # 검증셋 오차 계산
    val_X_compressed = model.transform(val_X)
    val_X_reconstructed = model.inverse_transform(val_X_compressed)
    val_window_scores = np.mean((val_X - val_X_reconstructed) ** 2, axis=1)

    # 테스트셋 오차 계산
    test_X_compressed = model.transform(test_X)
    test_X_reconstructed = model.inverse_transform(test_X_compressed)
    test_window_scores = np.mean((test_X - test_X_reconstructed) ** 2, axis=1)

    val_scores  = windows_to_timestep_scores(val_window_scores,  len(val_df),  W, S)
    test_scores = windows_to_timestep_scores(test_window_scores, len(test_df), W, S)

    # 최종 점수 평가
    val_auroc  = roc_auc_score(val_labels,  val_scores)
    val_aupr   = average_precision_score(val_labels,  val_scores)
    test_auroc = roc_auc_score(test_labels, test_scores)
    test_aupr  = average_precision_score(test_labels, test_scores)

    print("\n=== [5] 🏆 PCA 최종 평가 결과 🏆 ===")
    print(f"{'':15s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val':15s} {val_auroc:>8.4f} {val_aupr:>8.4f}")
    print(f"{'test_public':15s} {test_auroc:>8.4f} {test_aupr:>8.4f}")
    print("===========================================")

    # 모델 파일 및 후속 앙상블을 위한 스코어 CSV 저장
    os.makedirs("models", exist_ok=True)
    os.makedirs("scores", exist_ok=True)
    
    joblib.dump(scaler, "models/pca_scaler.pkl")
    joblib.dump(model, "models/pca_model.pkl")

    pd.DataFrame({'t': val_df['t'], 'PCA_score': val_scores}).to_csv("scores/pca_val_scores.csv", index=False)
    pd.DataFrame({'t': test_df['t'], 'PCA_score': test_scores}).to_csv("scores/pca_test_scores.csv", index=False)
    print("➔ PCA 최종 모델 및 CSV 스코어 저장 완료.")