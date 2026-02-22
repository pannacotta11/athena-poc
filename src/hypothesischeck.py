import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def perform_hypothesis_testing(df):
    """
    EDAの発見事項に対して統計的検定を行い、レポート用の根拠を出力する。
    """
    print("=== Hypothesis Testing Start ===")
    
    # ---------------------------------------------------------
    # 検定1: Concept Drift (分布シフト) の確認
    # 帰無仮説: 2016年と2017年の油温(OT)分布は同じである
    # ---------------------------------------------------------
    print("\n[Test 1] Distribution Shift Check (2016 vs 2017)")
    
    # 年ごとのデータ抽出
    ot_2016 = df[df.index.year == 2016]['OT'].dropna()
    ot_2017 = df[df.index.year == 2017]['OT'].dropna()
    
    # Kolmogorov-Smirnov検定
    ks_stat, p_value = stats.ks_2samp(ot_2016, ot_2017)
    
    print(f"  - 2016 Samples: {len(ot_2016)}")
    print(f"  - 2017 Samples: {len(ot_2017)}")
    print(f"  - KS Statistic: {ks_stat:.4f}")
    print(f"  - P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("  => 判定: 有意差あり (p < 0.05)。2016年と2017年の分布は統計的に異なります。")
        print("  => 示唆: 過去データをそのまま学習させると、将来予測で精度が落ちるリスクがあります。")
    else:
        print("  => 判定: 有意差なし。分布は同一とみなせます。")

    # ---------------------------------------------------------
    # 検定2: 時間遅れ (Time Lag) の特定
    # 問い: 負荷(HUFL)が変化してから、何時間後に油温(OT)が最も相関するか？
    # ---------------------------------------------------------
    print("\n[Test 2] Cross-Correlation Analysis (Lag Detection)")
    
    target = 'OT'
    feature = 'HUFL' # High Use Full Load
    max_lag = 12 # 前後12時間を見る
    
    correlations = []
    lags = range(0, max_lag + 1)
    
    for lag in lags:
        # featureをずらして、未来のOTとの相関を見る
        # (例: lag=1 は、1時間前の負荷と現在の温度の相関)
        corr = df[target].corr(df[feature].shift(lag))
        correlations.append(corr)
        
    best_lag = lags[np.argmax(correlations)]
    max_corr = max(correlations)
    
    print(f"  - Check Lags: 0 to {max_lag} hours")
    print(f"  - Max Correlation: {max_corr:.4f} at Lag {best_lag}")
    print(f"  - Correlation at Lag 0: {correlations[0]:.4f}")
    
    if best_lag > 0:
        print(f"  => 発見: 負荷変動から{best_lag}時間後に温度相関が最大化します。")
        print("  => 対策: モデル特徴量に単純な負荷だけでなく、ラグ特徴量(t-1, t-2...)を追加すべきです。")
    else:
        print("  => 発見: 遅れは見られません。")

    return best_lag

# --- 実行用メインブロック ---
if __name__ == "__main__":
    # データの読み込み (パスは適宜調整してください)
    data_path = '../data/ETTh1.csv' 
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    perform_hypothesis_testing(df)