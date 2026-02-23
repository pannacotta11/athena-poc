import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def visualize_seasonality(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # オイル温度（OT）
    series = df['OT']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 自己相関プロット (最大72時間=3日分)
    # これで24, 48, 72のところに山ができれば「強い周期性」の証明
    plot_acf(series, lags=72, ax=ax, title="Autocorrelation of Oil Temperature (Evidence of Seasonality)")
    
    # 24時間ごとのグリッド線を入れる
    for i in range(24, 73, 24):
        ax.axvline(i, color='red', linestyle='--', alpha=0.5)
        ax.text(i, 0.5, f'{i}h', color='red', ha='center')
        
    plt.xlabel("Lag (Hours)")
    plt.ylabel("Autocorrelation")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    path = '../data/ETTh1.csv'
    visualize_seasonality(path)