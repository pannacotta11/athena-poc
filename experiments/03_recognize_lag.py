import pandas as pd
import matplotlib.pyplot as plt

# 1. データ読み込み
df = pd.read_csv('../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# わかりやすいように「2016年7月の最初の1週間」だけ切り出し
subset = df['2016-07-01':'2016-07-07']

# 2. グラフ描画
fig, ax1 = plt.subplots(figsize=(15, 8))

# --- 左軸（負荷グループ） ---
# HUFL (Main): 濃い青
ax1.plot(subset.index, subset['HUFL'], color='navy', label='HUFL (High)', linewidth=1.5)

# MUFL (Sub): 水色・点線（HUFLと重なるか見るため）
ax1.plot(subset.index, subset['MUFL'], color='cyan', linestyle='--', label='MUFL (Mid)', linewidth=1.5)

# LUFL (Independent): 緑色（動きの違いを見るため）
ax1.plot(subset.index, subset['LUFL'], color='green', label='LUFL (Low)', linewidth=1.5, alpha=0.8)

ax1.set_ylabel('Load (HUFL / MUFL / LUFL)', color='navy', fontsize=14)
ax1.tick_params(axis='y', labelcolor='navy')
ax1.set_ylim(bottom=0) # 0からスタートさせる

# --- 右軸（温度） ---
ax2 = ax1.twinx()
# OT: 赤色・太線
ax2.plot(subset.index, subset['OT'], color='red', label='Temp (OT)', linewidth=2.5)
ax2.set_ylabel('Temperature (OT)', color='red', fontsize=14)
ax2.tick_params(axis='y', labelcolor='red')

# タイトルとグリッド
plt.title("Verification: All Loads vs Temperature (Who is the main driver?)", fontsize=16)
plt.grid(True)

# 凡例をまとめて表示する魔法のコード
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

plt.show()

# --- おまけ：それぞれのラグ相関も計算してみる ---
lags = range(0, 7) # 0〜6時間
print(f"{'Lag':<5} | {'HUFL vs OT':<12} | {'MUFL vs OT':<12} | {'LUFL vs OT':<12}")
print("-" * 50)

for lag in lags:
    c_hu = df['HUFL'].shift(lag).corr(df['OT'])
    c_mu = df['MUFL'].shift(lag).corr(df['OT'])
    c_lu = df['LUFL'].shift(lag).corr(df['OT'])
    print(f"{lag:<5} | {c_hu:.4f}       | {c_mu:.4f}       | {c_lu:.4f}")