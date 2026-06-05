import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 데이터 로드 환경 설정
# ============================================================
DATA_DIR = "./data"
SCORES_DIR = "./scores"

def load_labels(name, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{name}.csv")
    raw = pd.read_csv(path)
    if "label" in raw.columns:
        return raw["label"].to_numpy().astype(int)
    return None

if __name__ == "__main__":
    print("[1] 검증셋(Val) 정답 라벨 및 스코어 파일 로드")
    val_labels = load_labels("val")
    # Data Leakage 방지를 위해 test 데이터는 튜닝 단계에서 로드하지 않음

    # 모델별 스코어 로드
    v_gmm = pd.read_csv(os.path.join(SCORES_DIR, "gmm_val_scores.csv"))["GMM_score"].to_numpy()
    v_lof = pd.read_csv(os.path.join(SCORES_DIR, "lof_val_scores.csv"))["LOF_score"].to_numpy()
    v_pca = pd.read_csv(os.path.join(SCORES_DIR, "pca_val_scores.csv"))["PCA_score"].to_numpy()

    print("[2] 스케일 불일치 타파를 위한 Rank-based 분위수 변환 진행")
    v_gmm_r = rankdata(v_gmm) / len(v_gmm)
    v_lof_r = rankdata(v_lof) / len(v_lof)
    v_pca_r = rankdata(v_pca) / len(v_pca)

    print("[3] Grid Search 기반 최적 앙상블 가중치 탐색 (Val AUPR 기준)")
    weights = np.arange(0.0, 1.05, 0.05)
    
    best_val_aupr = 0.0
    best_weights = None

    for w_gmm in weights:
        for w_lof in weights:
            for w_pca in weights:
                if not np.isclose(w_gmm + w_lof + w_pca, 1.0):
                    continue

                # 앙상블 스코어 계산
                ensemble_score = (w_gmm * v_gmm_r) + (w_lof * v_lof_r) + (w_pca * v_pca_r)
                
                current_aupr = average_precision_score(val_labels, ensemble_score)
                
                if current_aupr > best_val_aupr:
                    best_val_aupr = current_aupr
                    best_weights = (w_gmm, w_lof, w_pca)

    print("\n==================================================")
    print(" [가중치 최적화 완료] 오직 Val 데이터 기준 최적의 비율")
    print("--------------------------------------------------")
    print(f" - GMM 비율: {best_weights[0]:.2f}")
    print(f" - LOF 비율: {best_weights[1]:.2f}")
    print(f" - PCA 비율: {best_weights[2]:.2f}")
    print(f" - 최고 검증셋 성능 (Val AUPR): {best_val_aupr:.4f}")
    print("==================================================\n")
    print("➔ 도출된 최적 가중치를 'ensemble_final.py'에 적용하여 최종 평가를 수행하세요.")