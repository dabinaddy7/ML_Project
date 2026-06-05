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
    # ==========================================
    # 튜닝 단계에서 도출된 최적 가중치 하드코딩
    # ==========================================
    W_GMM = 0.75
    W_LOF = 0.25
    W_PCA = 0.00
    
    print("[1] 평가용 정답 라벨 및 스코어 로드")
    val_labels = load_labels("val")
    test_labels = load_labels("test_public")

    # Val 모델 스코어 로드
    v_gmm = pd.read_csv(os.path.join(SCORES_DIR, "gmm_val_scores.csv"))["GMM_score"].to_numpy()
    v_lof = pd.read_csv(os.path.join(SCORES_DIR, "lof_val_scores.csv"))["LOF_score"].to_numpy()
    v_pca = pd.read_csv(os.path.join(SCORES_DIR, "pca_val_scores.csv"))["PCA_score"].to_numpy()

    # Test 모델 스코어 로드
    t_gmm = pd.read_csv(os.path.join(SCORES_DIR, "gmm_test_scores.csv"))["GMM_score"].to_numpy()
    t_lof = pd.read_csv(os.path.join(SCORES_DIR, "lof_test_scores.csv"))["LOF_score"].to_numpy()
    t_pca = pd.read_csv(os.path.join(SCORES_DIR, "pca_test_scores.csv"))["PCA_score"].to_numpy()

    print("[2] Rank-based 스케일링 적용")
    v_gmm_r = rankdata(v_gmm) / len(v_gmm)
    v_lof_r = rankdata(v_lof) / len(v_lof)
    v_pca_r = rankdata(v_pca) / len(v_pca)

    t_gmm_r = rankdata(t_gmm) / len(t_gmm)
    t_lof_r = rankdata(t_lof) / len(t_lof)
    t_pca_r = rankdata(t_pca) / len(t_pca)

    print(f"[3] 최적 가중치 결합 (GMM: {W_GMM}, LOF: {W_LOF}, PCA: {W_PCA})")
    val_ensemble_score  = (W_GMM * v_gmm_r) + (W_LOF * v_lof_r) + (W_PCA * v_pca_r)
    test_ensemble_score = (W_GMM * t_gmm_r) + (W_LOF * t_lof_r) + (W_PCA * t_pca_r)

    print("[4] 융합 앙상블 지표 최종 평가")
    val_auroc  = roc_auc_score(val_labels,  val_ensemble_score)
    val_aupr   = average_precision_score(val_labels,  val_ensemble_score)
    test_auroc = roc_auc_score(test_labels, test_ensemble_score)
    test_aupr  = average_precision_score(test_labels, test_ensemble_score)

    print("\n===========================================")
    print(f" [최종 융합 앙상블 성능 평가 결과]")
    print("-------------------------------------------")
    print(f" {'Data Split':15s} | {'AUROC':>8s} | {'AUPR':>8s}")
    print("-------------------------------------------")
    print(f" {'Validation':15s} | {val_auroc:>8.4f} | {val_aupr:>8.4f}")
    print(f" {'Test (Public)':15s} | {test_auroc:>8.4f} | {test_aupr:>8.4f}")
    print("===========================================\n")

    # 결과 저장 (보고서 요구사항: test_hidden_no_labels 예측을 대비한 최종 파이프라인 형태)
    pd.DataFrame({'t': np.arange(len(val_ensemble_score)), 'score': val_ensemble_score}).to_csv(os.path.join(SCORES_DIR, "ensemble_val_scores.csv"), index=False)
    pd.DataFrame({'t': np.arange(len(test_ensemble_score)), 'score': test_ensemble_score}).to_csv(os.path.join(SCORES_DIR, "ensemble_test_scores.csv"), index=False)
    
    print("[5] 결과물 저장 완료: scores/ 디렉토리 확인")