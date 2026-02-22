import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# データ準備（さっきと同じ）
# ---------------------------------------------------------
def prepare_data(filepath, horizon):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # Cyclic Features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    cols = ['HUFL', 'HULL', 'MUFL', 'MULL']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_roll_mean_6h'] = df[col].rolling(6).mean()
        df[f'{col}_diff'] = df[col].diff()

    # ★ Lag特徴量としては「現在の温度」を使うが、
    # ターゲットはあくまで「Horizon時間後」
    df['Current_OT'] = df['OT']
    
    # ターゲット作成
    df['Target_OT'] = df['OT'].shift(-horizon)
    df = df.dropna()
    return df

# ---------------------------------------------------------
# ハイブリッド検証
# ---------------------------------------------------------
def run_hybrid_experiment(filepath):
    horizons = [1, 3, 6, 12, 24]
    results = []
    
    print(f"{'Hz':<3} | {'Linear':<8} | {'LGBM Only':<10} | {'Hybrid':<8} | {'Winner'}")
    print("-" * 55)

    for h in horizons:
        df = prepare_data(filepath, horizon=h)
        
        # Split
        split_idx = int(len(df) * 0.8)
        feature_cols = [c for c in df.columns if c != 'Target_OT']
        
        X_train = df.iloc[:split_idx][feature_cols]
        y_train = df.iloc[:split_idx]['Target_OT']
        X_test = df.iloc[split_idx:][feature_cols]
        y_test = df.iloc[split_idx:]['Target_OT']
        
        # --- 1. Linear Regression (Base) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = Ridge(alpha=1.0)
        lr.fit(X_train_scaled, y_train)
        
        lr_pred_train = lr.predict(X_train_scaled) # 学習データでの予測
        lr_pred_test = lr.predict(X_test_scaled)   # テストデータでの予測
        
        mae_lr = mean_absolute_error(y_test, lr_pred_test)
        
        # --- 2. Calculate Residuals (残差) ---
        # Linearが外しちゃった分 = 正解 - Linear予測
        y_residuals_train = y_train - lr_pred_train
        
        # --- 3. LightGBM for Residuals ---
        # LGBMは「温度」ではなく「Linearの誤差」を予測する
        lgb_res = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42
        )
        lgb_res.fit(
            X_train, y_residuals_train, # ターゲットが残差
            eval_set=[(X_train, y_residuals_train)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        lgb_residual_pred = lgb_res.predict(X_test)
        
        # --- 4. Hybrid Prediction ---
        hybrid_pred = lr_pred_test + lgb_residual_pred
        mae_hybrid = mean_absolute_error(y_test, hybrid_pred)
        
        # (参考) 純粋なLGBM
        lgb_pure = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
        lgb_pure.fit(X_train, y_train, callbacks=[lgb.early_stopping(50, verbose=False)], eval_set=[(X_test, y_test)])
        mae_pure_lgb = mean_absolute_error(y_test, lgb_pure.predict(X_test))

        # 判定
        min_mae = min(mae_lr, mae_pure_lgb, mae_hybrid)
        if min_mae == mae_hybrid: winner = "Hybrid"
        elif min_mae == mae_lr: winner = "Linear"
        else: winner = "LGBM"
        
        results.append({'Horizon': h, 'Linear': mae_lr, 'LGBM': mae_pure_lgb, 'Hybrid': mae_hybrid})
        print(f"{h:<3} | {mae_lr:.4f}   | {mae_pure_lgb:.4f}     | {mae_hybrid:.4f}   | {winner}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    path = '../data/ETTh1.csv'
    res_df = run_hybrid_experiment(path)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Horizon'], res_df['Linear'], '--', label='Linear', marker='o')
    plt.plot(res_df['Horizon'], res_df['LGBM'], ':', label='LGBM Only', color='gray', alpha=0.7)
    plt.plot(res_df['Horizon'], res_df['Hybrid'], '-', label='Hybrid (Linear+LGBM)', color='red', linewidth=2, marker='*')
    
    plt.title("Can Hybrid Model Beat Linear? (Residual Learning)")
    plt.xlabel("Horizon (Hours)")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()
    plt.show()