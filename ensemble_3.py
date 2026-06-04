import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# 1. 정답 라벨 및 누적 스코어 로드
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
    # 정답 라벨 로드 (평가용)
    val_labels = load_labels("val")
    test_labels = load_labels("test_public")

    # 기존 실행 코드들이 저장해 둔 싱글 모델 스코어 파일 불러오기
    print("=== [1] 각 싱글 모델별 Anomaly Score 파일 로드 ===")
    v_gmm = pd.read_csv(os.path.join(SCORES_DIR, "gmm_val_scores.csv"))["GMM_score"].to_numpy()
    t_gmm = pd.read_csv(os.path.join(SCORES_DIR, "gmm_test_scores.csv"))["GMM_score"].to_numpy()

    v_lof = pd.read_csv(os.path.join(SCORES_DIR, "lof_val_scores.csv"))["LOF_score"].to_numpy()
    t_lof = pd.read_csv(os.path.join(SCORES_DIR, "lof_test_scores.csv"))["LOF_score"].to_numpy()

    v_pca = pd.read_csv(os.path.join(SCORES_DIR, "pca_val_scores.csv"))["PCA_score"].to_numpy()
    t_pca = pd.read_csv(os.path.join(SCORES_DIR, "pca_test_scores.csv"))["PCA_score"].to_numpy()

    # ============================================================
    # 2. Rank-based 스케일링 변환 (스케일 불일치 극복)
    # ============================================================
    # 각 score 배열을 0~1 사이의 순위 분위수 형태로 균일화합니다.
    v_gmm_r, t_gmm_r = rankdata(v_gmm) / len(v_gmm), rankdata(t_gmm) / len(t_gmm)
    v_lof_r, t_lof_r = rankdata(v_lof) / len(v_lof), rankdata(t_lof) / len(t_lof)
    v_pca_r, t_pca_r = rankdata(v_pca) / len(v_pca), rankdata(t_pca) / len(t_pca)

    # ============================================================
    # 3. 🎯 가중치 최적화 Grid Search (오직 Val 셋 기준)
    # ============================================================
    # 세 모델의 가중치 합이 1.0이 되도록 0.05 단위로 촘촘하게 순회 탐색합니다.
    best_val_aupr = -1
    best_weights = (0, 0, 0)
    
    print("\n=== [2] 최적 앙상블 가중치 조합 탐색 (오직 Val 셋 AUPR 기준) ===")
    
    step = 0.05
    for w_gmm in np.arange(0.0, 1.01, step):
        for w_lof in np.arange(0.0, 1.01 - w_gmm, step):
            w_pca = 1.0 - w_gmm - w_lof
            
            # 부동소수점 오차 방지 및 올바른 가중치만 필터링
            if w_pca < -1e-9: 
                continue
            w_pca = max(0.0, w_pca)
            
            # 검증셋 스코어 선형 결합
            val_ensemble_score = (w_gmm * v_gmm_r) + (w_lof * v_lof_r) + (w_pca * v_pca_r)
            
            # 오직 val 라벨로만 성능 측정 (Data Leakage 차단)
            current_val_aupr = average_precision_score(val_labels, val_ensemble_score)
            
            if current_val_aupr > best_val_aupr:
                best_val_aupr = current_val_aupr
                best_weights = (w_gmm, w_lof, w_pca)

    print("\n==================================================")
    print("🏆 [가중치 최적화 완료] 오직 Val 데이터 기준 최적의 비율")
    print(f"➔ GMM 비율: {best_weights[0]:.2f}")
    print(f"➔ LOF 비율: {best_weights[1]:.2f}")
    print(f"➔ PCA 비율: {best_weights[2]:.2f}")
    print(f"🏆 최적 검증셋 성능 (Val AUPR): {best_val_aupr:.4f}")
    print("==================================================")

    # ============================================================
    # 4. 📊 최종 일반화 성능 검증 (test_public 평가 딱 1회 진행)
    # ============================================================
    # 검증셋에서 채택된 '최적의 가중치' 비율을 그대로 테스트셋에 주입합니다.
    final_w_gmm, final_w_lof, final_w_pca = best_weights
    
    test_ensemble_score = (final_w_gmm * t_gmm_r) + (final_w_lof * t_lof_r) + (final_w_pca * t_pca_r)
    val_ensemble_score_best = (final_w_gmm * v_gmm_r) + (final_w_lof * v_lof_r) + (final_w_pca * v_pca_r)

    # 최종 지표 계산
    val_auroc  = roc_auc_score(val_labels,  val_ensemble_score_best)
    val_aupr   = average_precision_score(val_labels,  val_ensemble_score_best)
    test_auroc = roc_auc_score(test_labels, test_ensemble_score)
    test_aupr  = average_precision_score(test_labels, test_ensemble_score)

    # 🏆 교수님 보고서용 최종 통합 성적표 인쇄
    print("\n=== [3] 🏆 융합 앙상블 파이프라인 최종 결과 🏆 ===")
    print(f"{'':15s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val (Best)':15s} {val_auroc:>8.4f} {val_aupr:>8.4f}")
    print(f"{'test_public':15s} {test_auroc:>8.4f} {test_aupr:>8.4f}")
    print("==================================================")
    print("➔ 앙상블 점수 산출 완료. 이 비율 서사를 보고서에 인용하세요.")