import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def load_scores():
    DATA_DIR = "./data"
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    val_labels = val_df["label"].to_numpy().astype(int)

    gmm_val = pd.read_csv("scores/gmm_val_scores.csv")
    ocsvm_val = pd.read_csv("scores/ocsvm_val_scores.csv")
    return val_labels, gmm_val, ocsvm_val

if __name__ == "__main__":
    print("=== [1] 데이터 및 모델 점수 로드 ===")
    val_labels, gmm_val, ocsvm_val = load_scores()

    print("=== [2] 순위(Rank) 변환 ===")
    val_rank_gmm   = gmm_val['GMM_score'].rank(method='min')
    val_rank_ocsvm = ocsvm_val['OCSVM_score'].rank(method='min')

    print("=== [3] 최적의 앙상블 가중치 탐색 시작 ===\n")
    best_w = 0.5
    best_aupr = 0
    best_auroc = 0

    # 0.0부터 1.0까지 0.05 단위로 세밀하게 탐색
    for w in np.arange(0.0, 1.05, 0.05):
        w_gmm = round(w, 2)
        w_ocsvm = round(1.0 - w_gmm, 2)
        
        # 가중치 적용
        val_scores = (val_rank_gmm * w_gmm) + (val_rank_ocsvm * w_ocsvm)
        
        auroc = roc_auc_score(val_labels, val_scores)
        aupr  = average_precision_score(val_labels, val_scores)
        
        print(f"GMM: {w_gmm:.2f} | OC-SVM: {w_ocsvm:.2f} ➔ Val AUROC: {auroc:.4f} | Val AUPR: {aupr:.4f}")
        
        if aupr > best_aupr:
            best_aupr = aupr
            best_auroc = auroc
            best_w = w_gmm

    print("\n" + "="*50)
    print(f"🏆 [탐색 완료] 최적의 가중치 ➔ GMM: {best_w:.2f} | OC-SVM: {1.0 - best_w:.2f}")
    print(f"🏆 최고 Val AUPR: {best_aupr:.4f}")
    print("="*50)