import pandas as pd
import matplotlib.pyplot as plt

# 1. データ読み込み (15分足データ推奨)
df = pd.read_csv('../data/ETTm1.csv') # パスは環境に合わせて調整してください

# 日付型へ変換
df['date'] = pd.to_datetime(df['date'])

# 分析対象の期間を絞る（全期間だと重いため、動きが激しい数日間をピックアップ）
# 例: 2016年7月の1週間分
subset = df[(df['date'] >= '2016-07-20') & (df['date'] < '2016-07-27')]

# 2. 可視化 (2軸プロット)
fig, ax1 = plt.subplots(figsize=(15, 6))

# 左軸: 負荷 (例: MUFL) -> これが「冷却のトリガー」
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Load (MUFL)', color=color)
ax1.plot(subset['date'], subset['MUFL'], color=color, label='Load (MUFL)')
ax1.tick_params(axis='y', labelcolor=color)

# 右軸: 温度 (OT) -> これが「オーバーシュートを見る対象」
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Oil Temp (OT)', color=color)
ax2.plot(subset['date'], subset['OT'], color=color, linestyle='--', label='Temp (OT)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Load vs Temperature: Looking for Overshoot/Lag')
plt.show()