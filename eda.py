import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 디렉토리 설정
DATA_DIR = "./data"

def run_eda():
    print("=== [1] 데이터 로드 및 기본 구조 확인 ===")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    
    print(f"Train 데이터 크기 (정상만 존재): {train_df.shape}")
    print(f"Validation 데이터 크기 (라벨 포함): {val_df.shape}")
    
    if 'label' in val_df.columns:
        anomaly_count = val_df['label'].sum()
        anomaly_rate = (anomaly_count / len(val_df)) * 100
        print(f"Validation 내 이상치 Timestep 수: {anomaly_count}개 ({anomaly_rate:.2f}%)")
    print("-" * 50)

    print("=== [2] 변수 타입 분류 (연속형 vs 이산형) ===")
    # t와 label을 제외한 피처 컬럼 선택
    feature_cols = [c for c in train_df.columns if c.startswith("x_")]
    
    continuous_cols = []
    discrete_cols = []
    
    for col in feature_cols:
        unique_cnt = train_df[col].nunique()
        print(f"컬럼 {col}의 고유값 개수: {unique_cnt}개")
        if unique_cnt <= 5:  # 고유값이 몇 개 없는 경우 이산형으로 분류
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
            
    print(f"\n* 이산형(Binary/Discrete) 변수: {discrete_cols}")
    print(f"* 연속형(Continuous) 변수: {continuous_cols}")
    print("-" * 50)

    print("=== [3] 정상 데이터 변수 간 상관관계 분석 ===")
    # 연속형 변수들 간의 상관계수 계산
    if continuous_cols:
        corr = train_df[continuous_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Matrix of Continuous Features (Normal Train)")
        plt.tight_layout()
        
        # 그림 파일로 저장 (노트북/PC 어디서든 결과 확인 가능)
        plt.savefig("eda_correlation_matrix.png")
        print("➔ 상관관계 히트맵이 'eda_correlation_matrix.png'로 저장되었습니다.")
    print("-" * 50)

    print("=== [4] 시계열 주기성 및 트렌드 시각화 (상위 1000 시점) ===")
    # 흐름을 눈으로 확인하기 위해 상위 1000개의 타임스텝 샘플링
    sample_df = train_df.head(1000)
    
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(12, 2 * len(feature_cols)), sharex=True)
    
    for i, col in enumerate(feature_cols):
        axes[i].plot(sample_df['t'], sample_df[col], linewidth=1, color='tab:blue')
        axes[i].set_ylabel(col, rotation=0, labelpad=20, ha='right')
        axes[i].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Timestep (t)")
    plt.suptitle("Time-Series Pattern Analysis (First 1000 Timesteps)", y=0.99, fontsize=14)
    plt.tight_layout()
    
    plt.savefig("eda_timeseries_patterns.png")
    print("➔ 시계열 패턴 그래프가 'eda_timeseries_patterns.png'로 저장되었습니다.")
    print("=" * 50)
    print("EDA 분석 완료! 생성된 이미지 파일 2개를 확인해 보세요.")

if __name__ == "__main__":
    run_eda()