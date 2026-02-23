import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. データ読み込み & 前処理
# ==========================================
def load_and_process_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    return df

# ==========================================
# 2. 特徴量エンジニアリング
# ==========================================
def create_features(df_input):
    df = df_input.copy()
    
    # --- 時間特徴量 (Cyclic Encoding) ---
    # 時間の連続性を保つ (23時の次は0時)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # --- ラグ特徴量 & 移動平均 (EDAの知見を反映) ---
    # 対象: HUFL (負荷)
    # 仮説検定でラグ=1~2時間で相関が高い場合、それを含める
    cols_to_lag = ['HUFL', 'HULL', 'MUFL', 'MULL']
    
    for col in cols_to_lag:
        # 直近の変動
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        
        # 蓄熱効果 (過去の蓄積) -> 移動平均
        df[f'{col}_roll_mean_3h'] = df[col].rolling(window=3).mean()
        df[f'{col}_roll_mean_6h'] = df[col].rolling(window=6).mean()
        
        # 変動の勢い (微分成分)
        df[f'{col}_diff'] = df[col].diff()

    # ターゲットのラグ (自己回帰成分)
    # ※ 注意: 推論時に未来の正解データは使えないため、
    #  「1時間先予測」をするモデルと仮定して、t-1の正解値を使う
    df['OT_lag1'] = df['OT'].shift(1)

    # 欠損値除去 (ラグ生成で発生した先頭行を消す)
    df = df.dropna()
    
    return df

# ==========================================
# 3. 学習・評価パイプライン
# ==========================================
def train_evaluate_compare(df):
    target_col = 'OT'
    feature_cols = [c for c in df.columns if c != target_col]
    
    # --- 時系列分割 (Hold-out) ---
    # テストデータ: 最後の20% (または特定の期間)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")

    # ==========================================
    # Model A: Baseline (Linear Regression)
    # 線形回帰はスケールに敏感なため正規化する
    # ==========================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = Ridge(alpha=1.0) # Ridgeで少し正則化
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    lr_mae = mean_absolute_error(y_test, lr_pred)
    print(f"[Baseline] Linear Regression MAE: {lr_mae:.4f}")

    # ==========================================
    # Model B: LightGBM
    # 非線形性、相互作用を考慮可能
    # ==========================================
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='mae',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # ログ出力を抑制
        ]
    )
    
    lgb_pred = lgb_model.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    print(f"[Champion] LightGBM MAE       : {lgb_mae:.4f}")
    
    # 改善率の計算
    improvement = (lr_mae - lgb_mae) / lr_mae * 100
    print(f">> LightGBM improved accuracy by: {improvement:.2f}%")

    return y_test, lr_pred, lgb_pred, lgb_model

# ==========================================
# 4. 可視化 (スライド用素材)
# ==========================================
def visualize_results(y_test, lr_pred, lgb_pred, lgb_model):
    # 1. 予測比較 (全体だと見づらいので最初の300点を拡大)
    plt.figure(figsize=(15, 6))
    limit = 300 
    plt.plot(y_test.index[:limit], y_test.values[:limit], label='Actual (GT)', color='black', alpha=0.6)
    plt.plot(y_test.index[:limit], lr_pred[:limit], label='Linear Regression', linestyle='--', color='blue', alpha=0.5)
    plt.plot(y_test.index[:limit], lgb_pred[:limit], label='LightGBM', color='red', alpha=0.8)
    
    plt.title("Prediction Comparison: Baseline vs LightGBM (First 300 Hours of Test Data)")
    plt.ylabel("Oil Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. 残差プロット (誤差の傾向を見る)
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_test - lgb_pred, alpha=0.1, color='red', label='LGBM Residuals')
    plt.scatter(y_test, y_test - lr_pred, alpha=0.1, color='blue', label='Linear Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Residual Plot (Actual vs Error)")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Error (Actual - Pred)")
    plt.legend()
    plt.show()

    # 3. Feature Importance
    lgb.plot_importance(lgb_model, max_num_features=15, importance_type='gain', figsize=(10, 6))
    plt.title("Feature Importance (Gain)")
    plt.show()

# --- 実行 ---
if __name__ == "__main__":
    path = '../data/ETTh1.csv'
    
    # 1. Load
    df_raw = load_and_process_data(path)
    
    # 2. Feature Engineering
    df_features = create_features(df_raw)
    
    # 3. Modeling & Compare
    y_test, lr_pred, lgb_pred, model = train_evaluate_compare(df_features)
    
    # 4. Visualize
    visualize_results(y_test, lr_pred, lgb_pred, model)