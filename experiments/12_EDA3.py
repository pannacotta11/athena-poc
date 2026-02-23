import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

# ---------------------------------------------------------
# 1. データ読み込み & 前処理
# ---------------------------------------------------------
# ※パスは環境に合わせてください
df = pd.read_csv('../data/ETTh1.csv', parse_dates=['date'], index_col='date')
target = 'OT'

# インデックスが確実にDatetime型であることを保証
df.index = pd.to_datetime(df.index)

# 頻度設定 (FutureWarning対応: 'H' -> 'h')
if df.index.freq is None:
    df = df.asfreq('h') # 小文字の 'h' に変更

# ---------------------------------------------------------
# Slide 4用: 周期性の証明 (ACF)
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plot_acf(df[target].dropna(), lags=168, alpha=0.05, title='Autocorrelation: Evidence of Daily(24h) & Weekly(168h) Seasonality')
plt.axvline(x=24, color='red', linestyle='--', alpha=0.8, label='24h (Daily)')
plt.axvline(x=48, color='red', linestyle=':', alpha=0.5)
plt.axvline(x=168, color='green', linestyle='--', alpha=0.8, label='168h (Weekly)')
plt.xlabel("Lag (Hours)")
plt.ylabel("Autocorrelation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Slide 5用: 分布シフトの証明 (KDE Plot)
# ---------------------------------------------------------
# 【修正箇所】 .loc を使って行（期間）を指定します
plt.figure(figsize=(10, 6))

# df.loc['2016'] で「2016年の行すべて」を取得
sns.kdeplot(df.loc['2016'][target], label='2016 (Training Period)', fill=True, color='blue', alpha=0.3)

# df.loc['2017':] で「2017年以降の行すべて」を取得
sns.kdeplot(df.loc['2017':][target], label='2017-2018 (Testing Period)', fill=True, color='red', alpha=0.3)

plt.title('Distribution Shift: Why Standard Models Fail', fontsize=14)
plt.xlabel('Oil Temperature (°C)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Slide 6用: NLinearのメカニズム (Window Histogram)
# ---------------------------------------------------------
np.random.seed(42)
lookback = 336 
num_samples = 5 

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
axes = axes.flatten()

for i in range(num_samples):
    # ランダムに期間を切り出す
    start = np.random.randint(0, len(df)-lookback-100)
    window = df[target].iloc[start:start+lookback].values
    
    # NLinear処理: 最後の値を引く
    norm_window = window - window[-1]
    
    # ヒストグラム描画
    axes[i].hist(window, bins=20, alpha=0.5, label='Raw Data', color='blue', density=True)
    axes[i].hist(norm_window, bins=20, alpha=0.7, label='NLinear Input\n(Last-val Subtracted)', color='orange', density=True)
    
    axes[i].axvline(0, color='orange', linestyle='--', linewidth=2)
    
    axes[i].set_title(f'Sample Window {i+1}')
    if i == 0:
        axes[i].legend(loc='upper right')

plt.suptitle('Why NLinear Works: "Last-Value Subtraction" aligns all distributions to Zero', fontsize=16)
plt.tight_layout()
plt.show()