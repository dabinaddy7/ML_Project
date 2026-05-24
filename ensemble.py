import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

if __name__ == "__main__":
    # 튜닝에서 찾은 최적의 가중치 하드코딩
    BEST_WEIGHT_GMM = 0.80
    BEST_WEIGHT_OCSVM = 0.20

    print("=== [1] Test 세트 정답 및 모델 점수 로드 ===")
    # test_public의 정답(label) 로드
    DATA_DIR = "./data"
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_public.csv"))
    test_labels = test_df["label"].to_numpy().astype(int)

    gmm_test  = pd.read_csv("scores/gmm_test_scores.csv")
    ocsvm_test = pd.read_csv("scores/ocsvm_test_scores.csv")

    print("=== [2] 순위(Rank) 변환 및 가중치 결합 ===")
    test_rank_gmm   = gmm_test['GMM_score'].rank(method='min')
    test_rank_ocsvm = ocsvm_test['OCSVM_score'].rank(method='min')

    final_test_scores = (test_rank_gmm * BEST_WEIGHT_GMM) + (test_rank_ocsvm * BEST_WEIGHT_OCSVM)

    # ============================================================
    # ⭐️ 빼먹었던 핵심 부분: test_public 평가 점수 출력
    # ============================================================
    print("\n=== [3] 🏆 test_public 최종 평가 점수 🏆 ===")
    test_auroc = roc_auc_score(test_labels, final_test_scores)
    test_aupr  = average_precision_score(test_labels, final_test_scores)
    
    print(f"➔ 최종 앙상블 Test AUROC: {test_auroc:.4f}")
    print(f"➔ 최종 앙상블 Test AUPR : {test_aupr:.4f}")
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