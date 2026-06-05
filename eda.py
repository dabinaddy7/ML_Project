import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 데이터 로드 및 환경 설정
# ============================================================
DATA_DIR = "./data"

def run_eda():
    print("[1] 데이터 로드 및 기본 구조 확인")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    
    print(f" - Train 데이터 크기 (정상 데이터): {train_df.shape}")
    print(f" - Validation 데이터 크기 (라벨 포함): {val_df.shape}")
    
    if 'label' in val_df.columns:
        anomaly_count = val_df['label'].sum()
        anomaly_rate = (anomaly_count / len(val_df)) * 100
        print(f" - Validation 내 이상치(Anomaly) 시점 수: {anomaly_count}개 ({anomaly_rate:.2f}%)")

    print("\n[2] 변수 타입 분류 (연속형 vs 이산형)")
    feature_cols = [c for c in train_df.columns if c.startswith("x_")]
    continuous_cols = []
    discrete_cols = []
    
    for col in feature_cols:
        unique_cnt = train_df[col].nunique()
        if unique_cnt <= 10:
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
            
    print(f" - 연속형 변수 (Continuous): {continuous_cols}")
    print(f" - 이산형 변수 (Discrete): {discrete_cols}")

    print("\n[3] 정상 데이터 변수 간 상관관계 분석 및 히트맵 저장")
    if continuous_cols:
        corr = train_df[continuous_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Matrix of Continuous Features")
        plt.tight_layout()
        
        plt.savefig("eda_correlation_matrix.png")
        print(" - 저장 완료: 'eda_correlation_matrix.png'")

    print("\n[4] 시계열 주기성 및 다중 운영 상태(Multi-modal) 패턴 시각화")
    sample_df = train_df.head(1000)
    
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(12, 2 * len(feature_cols)), sharex=True)
    for i, col in enumerate(feature_cols):
        axes[i].plot(sample_df['t'], sample_df[col], linewidth=1)
        axes[i].set_ylabel(col, rotation=0, labelpad=20, ha='right')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time Step (t)")
    plt.suptitle("Time Series Pattern (Top 1000 Timesteps)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig("eda_timeseries_patterns.png")
    print(" - 저장 완료: 'eda_timeseries_patterns.png'")
    print("\n[EDA 완료] 모든 탐색적 데이터 분석이 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    run_eda()