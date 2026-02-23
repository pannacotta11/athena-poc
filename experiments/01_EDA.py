import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # 日付の表示調整用
import seaborn as sns

# 1. データを読み込む
df = pd.read_csv('../data/ETTh1.csv')

# 【重要】ここが修正ポイント！
# 文字列の 'date' 列を、Pythonが理解できる「日付型」に変換します
df['date'] = pd.to_datetime(df['date'])

# 2. 油温(OT)をグラフにする
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['OT'])
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # 3ヶ月ごとにメモリを打つ
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # "2016-07" のように表示

plt.gcf().autofmt_xdate() # ラベルが重ならないように少し斜めにする
plt.title("Oil Temperature History")
plt.show()

# 3. 相関を見る（ヒートマップ）
plt.figure(figsize=(10, 8))
# date列は計算できないので除外して、数値データだけで相関を出す
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()
# 期間を指定してズームイン（2016年7月の1ヶ月間）
subset = df[(df['date'] >= '2016-07-01') & (df['date'] < '2016-08-01')]

plt.figure(figsize=(12, 6))
plt.plot(subset['date'], subset['OT'], marker='o', markersize=2) # 点を打つと見やすい
plt.title("Oil Temperature (Zoom In: July 2016)")
plt.grid(True)
plt.show()