import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_thermal_lag_evidence(filepath):
    # データ読み込み
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # ---------------------------------------------------------
    # 図1: 直感的な証拠（時系列のピークズレ）
    # ---------------------------------------------------------
    # 負荷変動が分かりやすい期間を切り抜く (2016年7月のある3日間)
    # ※データを見て、山がきれいな期間を選びました
    start_date = '2016-07-22 00:00'
    end_date = '2016-07-25 00:00'
    subset = df[start_date:end_date]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左軸: 負荷 (Load)
    color_load = 'tab:blue'
    ax1.set_xlabel('Date / Time')
    ax1.set_ylabel('Load (HUFL)', color=color_load, fontsize=12)
    ax1.plot(subset.index, subset['HUFL'], color=color_load, label='Load (HUFL)', linewidth=2, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color_load)
    
    # 右軸: 温度 (Oil Temp)
    ax2 = ax1.twinx()  
    color_temp = 'tab:red'
    ax2.set_ylabel('Oil Temperature (OT)', color=color_temp, fontsize=12)
    ax2.plot(subset.index, subset['OT'], color=color_temp, label='Oil Temp (OT)', linewidth=2, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color_temp)
    
    # ズレを強調する矢印と注釈
    # (適当なピークを見つけて矢印を引く)
    peak_time = pd.Timestamp('2016-07-23 14:00') # 負荷ピーク付近
    lag_time = pd.Timestamp('2016-07-23 16:00')  # 温度ピーク付近
    
    # 負荷のピークへの線
    ax1.axvline(peak_time, color='blue', linestyle=':', alpha=0.5)
    # 温度のピークへの線
    ax2.axvline(lag_time, color='red', linestyle=':', alpha=0.5)
    
    # 矢印
    plt.annotate('', xy=(lag_time, 40), xytext=(peak_time, 40),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.text(peak_time, 41, ' Time Lag (~2h)', fontsize=12, fontweight='bold')
    
    plt.title("Evidence 1: Visual Time Lag (Load vs Temperature)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 図2: 統計的な証拠（相互相関）
    # ---------------------------------------------------------
    # 全期間で計算
    lags = range(0, 7) # 0時間〜6時間前まで
    correlations = [df['OT'].corr(df['HUFL'].shift(lag)) for lag in lags]
    
    plt.figure(figsize=(8, 5))
    colors = ['gray'] * len(lags)
    # 最大の相関を持つラグを赤くする
    max_idx = np.argmax(correlations)
    colors[max_idx] = 'red'
    
    bars = plt.bar(lags, correlations, color=colors, alpha=0.8)
    
    plt.title("Evidence 2: Cross-Correlation Analysis", fontsize=14)
    plt.xlabel("Lag (Hours ago)", fontsize=12)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.xticks(lags)
    plt.ylim(min(correlations)*0.95, max(correlations)*1.005)
    
    # 値を表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # パスは適宜変更してください
    path = '../data/ETTh1.csv' 
    visualize_thermal_lag_evidence(path)