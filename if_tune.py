import numpy as np
import pandas as pd
import itertools
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

# ==========================================
# 1. 스타터 코드 헬퍼 함수 연동 (본인 환경에 맞게 수정!)
# ==========================================
# from starter import load_split, make_windows, make_window_labels

def get_flattened_windows(w_size):
    """
    스타터 코드의 make_windows를 사용하여 (샘플수, 윈도우, 채널수)를 만든 뒤,
    스케일링과 PCA 적용을 위해 2차원(Flatten)으로 펼쳐줍니다.
    """
    # [실제 환경에서는 아래 주석을 풀고 본인 코드를 연결하세요]
    # train_df = load_split('train')
    # val_df = load_split('val')
    # train_windows = make_windows(train_df, window_size=w_size)
    # val_windows = make_windows(val_df, window_size=w_size)
    # y_val = make_window_labels(val_df, window_size=w_size)
    
    # --- 가상 데이터 (코드 실행 테스트용) ---
    feature_cols = 10
    train_windows = np.random.rand(30000 - w_size + 1, w_size, feature_cols)
    val_windows = np.random.rand(15000 - w_size + 1, w_size, feature_cols)
    y_val = np.random.choice([0, 1], size=val_windows.shape[0], p=[0.85, 0.15])
    # ----------------------------------------
    
    X_train_raw = train_windows.reshape(train_windows.shape[0], -1)
    X_val_raw = val_windows.reshape(val_windows.shape[0], -1)
    
    return X_train_raw, X_val_raw, y_val

# ==========================================
# 2. 극한의 파라미터 그리드 정의 (모든 경우의 수)
# ==========================================
param_grid = {
    # [전처리 영역]
    'window_size': [30, 60, 90],          # 단기, 중기, 장기 시계열 패턴 탐색
    'pca_variance': [0.90, 0.95, 0.99],   # 정보 보존율: 90%, 95%, 99% 보존
    
    # [Isolation Forest 영역]
    'n_estimators': [200, 400],           # 트리 개수 (많을수록 안정적)
    'max_samples': [256, 512, 'auto'],    # 트리를 만들 때 사용할 샘플 수
    'max_features': [0.7, 1.0]            # 변수 무작위 선택 비율
}

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

total_combinations = len(experiments)
print(f"🔥 총 {total_combinations}개의 전처리+모델 파라미터 최적화 탐색을 시작합니다 🔥\n")

results = []
best_aupr = -1.0
best_config = None

start_time = time.time()

# ==========================================
# 3. 최적화 루프 실행
# ==========================================
for idx, config in enumerate(experiments):
    w_size = config['window_size']
    pca_var = config['pca_variance']
    
    # 1. 윈도우 동적 생성
    X_train_raw, X_val_raw, y_val = get_flattened_windows(w_size)
    
    # 2. 스케일링 (채널 간 분산 차이 보정 - 매우 중요!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    # 3. PCA 차원 축소 (분산 보존율 기반)
    pca = PCA(n_components=pca_var, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    # 4. 모델 학습
    model = IsolationForest(
        n_estimators=config['n_estimators'],
        max_samples=config['max_samples'],
        max_features=config['max_features'],
        contamination='auto', 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_pca)
    
    # 5. 이상치 점수 산출 및 부호 반전 (-) 처리
    val_scores = -model.decision_function(X_val_pca)
    
    # 6. 평가 지표 계산
    val_aupr = average_precision_score(y_val, val_scores)
    val_auroc = roc_auc_score(y_val, val_scores)
    
    # 기록 저장
    results.append({
        'window_size': w_size,
        'pca_variance': pca_var,
        'pca_n_components_used': pca.n_components_, # 실제로 몇 차원으로 줄었는지 확인
        'n_estimators': config['n_estimators'],
        'max_samples': config['max_samples'],
        'max_features': config['max_features'],
        'Val_AUPR': val_aupr,
        'Val_AUROC': val_auroc
    })
    
    # 최고 점수 갱신 시 알림
    if val_aupr > best_aupr:
        best_aupr = val_aupr
        best_config = results[-1]
        print(f"  🏆 [NEW BEST] W:{w_size}, PCA_Var:{pca_var} -> AUPR: {val_aupr:.4f} (AUROC: {val_auroc:.4f})")
    
    # 진행 상황 출력 (10번마다)
    if (idx + 1) % 10 == 0:
        print(f"[{idx+1}/{total_combinations}] 진행 중... (현재 최고 AUPR: {best_aupr:.4f})")

# ==========================================
# 4. 최종 결과 출력 및 분석 파일 저장
# ==========================================
elapsed = time.time() - start_time
df_results = pd.DataFrame(results).sort_values(by='Val_AUPR', ascending=False).reset_index(drop=True)

print("\n" + "="*50)
print(f"🏁 탐색 완료! (총 소요시간: {elapsed/60:.1f}분) 🏁")
print("="*50)
print("★ 압도적 1위 최적 하이퍼파라미터 조합 ★")
for key, value in best_config.items():
    if 'AUPR' in key or 'AUROC' in key:
        print(f"  ▶ {key}: {value:.4f}")
    else:
        print(f"  - {key}: {value}")

df_results.to_csv("isolation_forest_optimal_params.csv", index=False)
print("\n모든 조합의 결과가 'isolation_forest_optimal_params.csv'에 저장되었습니다. 보고서에 표로 첨부하세요!")