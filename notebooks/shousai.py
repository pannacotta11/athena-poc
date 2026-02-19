import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# データの読み込み
file_path = '../data/ETTh1.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

def plot_average_hysteresis(df):
    # ---------------------------------------------------------
    # 修正ポイント: .loc を使って期間を指定する
    # ---------------------------------------------------------
    try:
        # 2016年7月のデータだけを抽出
        subset = df.loc['2016-07'].copy()
    except KeyError:
        print("エラー: データの日付範囲を確認してください。'2016-07'が含まれていない可能性があります。")
        return
    
    # 時間ごとの平均をとる（0時の平均、1時の平均...）
    hourly_avg = subset.groupby(subset.index.hour).mean()
    
    # ---------------------------------------------------------
    # プロット作成
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    # 1. 軌跡を描く（滑らかな線）
    plt.plot(hourly_avg['HUFL'], hourly_avg['OT'], 
             color='gray', linewidth=2, alpha=0.6, zorder=1)
    
    # 2. 点を打つ（時間帯で色分け）
    sc = plt.scatter(hourly_avg['HUFL'], hourly_avg['OT'], 
                     c=hourly_avg.index, cmap='twilight_shifted', 
                     s=100, edgecolor='k', zorder=2)
    
    # 3. 矢印で「時間の流れ」を示す
    for i in range(len(hourly_avg)): 
        x = hourly_avg['HUFL'].iloc[i]
        y = hourly_avg['OT'].iloc[i]
        
        # 次の時刻（23時の次は0時に戻る処理）
        next_idx = (i + 1) % len(hourly_avg)
        dx = hourly_avg['HUFL'].iloc[next_idx] - x
        dy = hourly_avg['OT'].iloc[next_idx] - y
        
        # 矢印を描画
        plt.arrow(x, y, dx*0.5, dy*0.5, 
                  head_width=0.2, head_length=0.2, fc='black', ec='black', zorder=3)

    # ---------------------------------------------------------
    # 装飾
    # ---------------------------------------------------------
    cbar = plt.colorbar(sc)
    cbar.set_label('Hour of Day (0-23)')
    
    plt.title('Average Thermal Hysteresis Loop (July 2016)\nCounter-Clockwise Loop = Thermal Lag', fontsize=14)
    plt.xlabel('Average Load (HUFL)', fontsize=12)
    plt.ylabel('Average Oil Temperature (OT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_average_hysteresis(df)