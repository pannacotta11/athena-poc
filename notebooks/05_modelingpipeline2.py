import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# データ読み込み & 基本特徴量作成
# ---------------------------------------------------------
def prepare_data(filepath, horizon):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # Cyclic Features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Lag Features (現在時点 t で利用可能な情報)
    cols = ['HUFL', 'HULL', 'MUFL', 'MULL']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_roll_mean_6h'] = df[col].rolling(6).mean()
        df[f'{col}_diff'] = df[col].diff()

    # ★重要: 「現在(t)」の温度は使える
    df['Current_OT'] = df['OT'] 
    
    # Target: 「Horizon時間先」の温度を予測する
    # shift(-horizon) で未来の値を現在行に持ってくる
    df['Target_OT'] = df['OT'].shift(-horizon)
    
    df = df.dropna()
    return df

# ---------------------------------------------------------
# 検証ループ (Horizonを変えながら対決)
# ---------------------------------------------------------
def run_horizon_experiment(filepath):
    horizons = [1, 3, 6, 12, 24] # 1時間後, 3時間後... 24時間後
    results = []
    
    print(f"{'Horizon':<10} | {'Linear MAE':<12} | {'LGBM MAE':<12} | {'Winner':<10}")
    print("-" * 55)

    for h in horizons:
        df = prepare_data(filepath, horizon=h)
        
        # Split (ラスト20%をテスト)
        split_idx = int(len(df) * 0.8)
        feature_cols = [c for c in df.columns if c != 'Target_OT']
        
        X_train = df.iloc[:split_idx][feature_cols]
        y_train = df.iloc[:split_idx]['Target_OT']
        X_test = df.iloc[split_idx:][feature_cols]
        y_test = df.iloc[split_idx:]['Target_OT']
        
        # --- 1. Linear Regression (Ridge) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = Ridge(alpha=1.0)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        
        # --- 2. LightGBM ---
        # パラメータは少し強めに設定
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03, # 学習率を下げて丁寧に
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_pred = model.predict(X_test)
        lgb_mae = mean_absolute_error(y_test, lgb_pred)
        
        # 判定
        winner = "Linear" if lr_mae < lgb_mae else "LightGBM"
        results.append({'Horizon': h, 'Linear': lr_mae, 'LGBM': lgb_mae})
        
        print(f"{h:<2} hours   | {lr_mae:.4f}       | {lgb_mae:.4f}       | {winner}")

    return pd.DataFrame(results)

# ---------------------------------------------------------
# 可視化
# ---------------------------------------------------------
if __name__ == "__main__":
    path = '../data/ETTh1.csv' # パスは環境に合わせて修正
    res_df = run_horizon_experiment(path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Horizon'], res_df['Linear'], marker='o', label='Linear Regression', linestyle='--')
    plt.plot(res_df['Horizon'], res_df['LGBM'], marker='o', label='LightGBM', color='red', linewidth=2)
    
    plt.title("Linear vs LightGBM: MAE by Prediction Horizon")
    plt.xlabel("Prediction Horizon (Hours ahead)")
    plt.ylabel("MAE (Lower is Better)")
    plt.legend()
    plt.grid(True)
    plt.xticks(res_df['Horizon'])
    plt.show()