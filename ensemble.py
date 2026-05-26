import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

if __name__ == "__main__":
    # 튜닝에서 찾은 최적의 가중치 하드코딩
    BEST_WEIGHT_GMM = 0.80
    BEST_WEIGHT_OCSVM = 0.20

    DATA_DIR = "./data"
    print("=== [1] 데이터 정답(Label) 및 모델 점수 로드 ===")
    
    # 1-1. Val 세트 로드 (다빈 님이 말씀하신 부분!)
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    val_labels = val_df["label"].to_numpy().astype(int)
    gmm_val   = pd.read_csv("scores/gmm_val_scores.csv")
    ocsvm_val = pd.read_csv("scores/ocsvm_val_scores.csv")

    # 1-2. Test 세트 로드
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_public.csv"))
    test_labels = test_df["label"].to_numpy().astype(int)
    gmm_test   = pd.read_csv("scores/gmm_test_scores.csv")
    ocsvm_test = pd.read_csv("scores/ocsvm_test_scores.csv")

    print("=== [2] 순위(Rank) 변환 및 가중치 결합 ===")
    
    # 2-1. Val 세트 앙상블 계산
    val_rank_gmm   = gmm_val['GMM_score'].rank(method='min')
    val_rank_ocsvm = ocsvm_val['OCSVM_score'].rank(method='min')
    final_val_scores = (val_rank_gmm * BEST_WEIGHT_GMM) + (val_rank_ocsvm * BEST_WEIGHT_OCSVM)

    # 2-2. Test 세트 앙상블 계산
    test_rank_gmm   = gmm_test['GMM_score'].rank(method='min')
    test_rank_ocsvm = ocsvm_test['OCSVM_score'].rank(method='min')
    final_test_scores = (test_rank_gmm * BEST_WEIGHT_GMM) + (test_rank_ocsvm * BEST_WEIGHT_OCSVM)

    print("\n=== [3] 🏆 최종 앙상블 평가 결과 🏆 ===")
    
    # Val 점수 계산
    val_auroc = roc_auc_score(val_labels, final_val_scores)
    val_aupr  = average_precision_score(val_labels, final_val_scores)
    
    # Test 점수 계산
    test_auroc = roc_auc_score(test_labels, final_test_scores)
    test_aupr  = average_precision_score(test_labels, final_test_scores)
    
    # ⭐️ 다빈 님이 원하셨던 GMM, OC-SVM과 완벽히 똑같은 표(Table) 출력
    print(f"{'':15s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val':15s} {val_auroc:>8.4f} {val_aupr:>8.4f}")
    print(f"{'test_public':15s} {test_auroc:>8.4f} {test_aupr:>8.4f}")
    print("===========================================")

    print("\n=== [4] 최종 제출용 CSV 파일 생성 ===")
    submission_df = pd.DataFrame({
        't': gmm_test['t'],
        'score': final_test_scores
    })

    # 본인의 학번으로 변경하세요
    submission_filename = "2026xxxx_임다빈_submission.csv"
    submission_df.to_csv(submission_filename, index=False)

    print(f"➔ 성공! '{submission_filename}' 파일이 생성(덮어쓰기) 되었습니다.")