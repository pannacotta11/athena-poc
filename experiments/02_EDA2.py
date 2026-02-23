import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(file_path):
    """データの読み込みと前処理"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def plot_social_activity(df):
    """1. 社会活動の可視化：曜日・時間帯による負荷の箱ひげ図"""
    df_plot = df.copy()
    df_plot['dow'] = df_plot.index.dayofweek # 0=Mon, 6=Sun
    df_plot['hour'] = df_plot.index.hour
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.boxplot(x='dow', y='HUFL', data=df_plot, ax=axes[0], palette="Blues")
    axes[0].set_title('Load Distribution by Day of Week (0=Mon, 6=Sun)')
    axes[0].set_xlabel('Day of Week')
    axes[0].set_ylabel('High Use Full Load (HUFL)')
    
    sns.boxplot(x='hour', y='HUFL', data=df_plot, ax=axes[1], palette="coolwarm")
    axes[1].set_title('Load Distribution by Hour of Day')
    axes[1].set_xlabel('Hour (0-23)')
    axes[1].set_ylabel('High Use Full Load (HUFL)')
    
    plt.tight_layout()
    plt.show()

def plot_hysteresis(df):
    """2. 昇温と降温（ヒステリシス）：負荷 vs 温度の散布図"""
    plt.figure(figsize=(10, 8))
    plt.hexbin(df['HUFL'], df['OT'], gridsize=50, cmap='inferno', mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel('Load (HUFL)')
    plt.ylabel('Oil Temperature (OT)')
    plt.title('Hysteresis Loop Check: Load vs Temperature\n(Ideally forms a loop due to thermal lag)')
    plt.show()

def plot_differential(df):
    """3. 微分係数の確認：変化量 vs 変化量"""
    df_diff = df.copy()
    df_diff['delta_HUFL'] = df_diff['HUFL'].diff()
    df_diff['delta_OT'] = df_diff['OT'].diff()
    
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(x='delta_HUFL', y='delta_OT', data=df_diff, alpha=0.1, s=10)
    
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    
    plt.title('Differential Analysis: Delta Load vs Delta Temp\n(How temp reacts to load change)')
    plt.xlabel('Change in Load (Delta HUFL)')
    plt.ylabel('Change in Temp (Delta OT)')
    plt.show()

def plot_yearly_comparison(df):
    """4. ピーク乖離の原因特定：2016年 vs 2017年の分布比較"""
    df_comp = df.copy()
    df_comp['year'] = df_comp.index.year
    
    df_comp = df_comp[df_comp['year'].isin([2016, 2017])]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.kdeplot(data=df_comp, x='HUFL', hue='year', fill=True, ax=axes[0], palette='Set1')
    axes[0].set_title('Load (HUFL) Distribution: 2016 vs 2017')
    axes[0].set_xlabel('High Use Full Load')
    
    sns.kdeplot(data=df_comp, x='OT', hue='year', fill=True, ax=axes[1], palette='Set1')
    axes[1].set_title('Oil Temp (OT) Distribution: 2016 vs 2017')
    axes[1].set_xlabel('Oil Temperature')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = '../data/ETTh1.csv' 
    df = load_data(file_path)
    print("Plotting 1: Social Activity...")
    plot_social_activity(df)
    
    print("Plotting 2: Hysteresis...")
    plot_hysteresis(df)
    
    print("Plotting 3: Differential Analysis...")
    plot_differential(df)
    
    print("Plotting 4: Yearly Comparison...")
    plot_yearly_comparison(df)
    
    print("Done.")