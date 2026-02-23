import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import os

# ==========================================
# 設定
# ==========================================
TARGET = 'OT'
WEEKS_TO_SHOW = 2

def plot_autocorrelation_wide(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    filename = os.path.basename(filepath)
    if 'm' in filename:
        steps_per_day = 96
        freq_name = "15-min"
    else:
        steps_per_day = 24
        freq_name = "Hourly"
        
    print(f"Data Frequency: {freq_name} ({steps_per_day} steps/day)")
    
    series = df[TARGET].dropna()
    nlags = steps_per_day * 7 * WEEKS_TO_SHOW
    
    # 計算
    acf_values = acf(series, nlags=nlags, fft=True)
    
    # ==========================================
    # グラフ描画（スライド用・横長ズーム版）
    # ==========================================
    # 横長にする (幅16, 高さ4)
    plt.figure(figsize=(16, 4))
    
    x_days = np.arange(len(acf_values)) / steps_per_day
    
    # メインプロット
    plt.plot(x_days, acf_values, color='#1f77b4', linewidth=2.0, label='Autocorrelation')
    
    # --- Y軸のズーム設定 (ここが重要) ---
    # データの最小値より少し下を下限に設定して、上部の波形を拡大する
    y_min = np.min(acf_values)
    y_range = 1.0 - y_min
    y_bottom = y_min - (y_range * 0.2) # 余白を20%とる
    # ただし下限が0を下回るようなら0にする（今回は相関高いので不要かもだが念のため）
    # y_bottom = max(0, y_bottom) 
    
    plt.ylim(bottom=y_bottom, top=1.02)
    
    # --- 周期性の強調 ---
    # 1. 日次周期 (赤点線)
    for day in range(1, WEEKS_TO_SHOW * 7 + 1):
        plt.axvline(x=day, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        
    # 2. 週次周期 (緑破線)
    for week in range(1, WEEKS_TO_SHOW + 1):
        day = week * 7
        plt.axvline(x=day, color='green', linestyle='--', alpha=0.8, linewidth=2.0)
        
        # テキスト位置をグラフの上端に合わせて調整
        plt.text(day, 1.0, f'{week} Week', 
                 color='green', fontweight='bold', ha='center', va='bottom',
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    plt.title(f'Autocorrelation of {TARGET} ({filename}) - Zoomed View', fontsize=16, fontweight='bold')
    plt.xlabel('Lag (Days)', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 目盛り文字サイズ
    plt.tick_params(labelsize=12)
    
    # 凡例
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#1f77b4', lw=2),
                    Line2D([0], [0], color='red', linestyle=':', lw=2),
                    Line2D([0], [0], color='green', linestyle='--', lw=2)]
    # 凡例をグラフの外（右上）に出して邪魔にならないようにする
    plt.legend(custom_lines, ['Autocorrelation', 'Daily Cycle (24h)', 'Weekly Cycle (7d)'], 
               loc='upper right', framealpha=0.9)

    plt.tight_layout()
    
    # 保存と表示
    save_path = 'autocorrelation_slide.png'
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    path = '../ETTm1.csv'
    if not os.path.exists(path):
        path = '../data/ETTm1.csv'
    
    if os.path.exists(path):
        plot_autocorrelation_wide(path)
    else:
        print("ファイルが見つかりません。")