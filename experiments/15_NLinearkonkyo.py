import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import random
import os

# ==========================================
# 0. 初期設定
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. モデル定義 (NLinear & RevIN)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False): # affine=Falseで過学習抑制
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self.stdev + self.mean
        return x

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim):
        super(NLinear, self).__init__()
        self.revin = RevIN(input_dim, affine=False)
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.Linear(x) # Shared Weights
        x = x.permute(0, 2, 1)
        x = self.revin(x, 'denorm')
        return x

# ==========================================
# 2. データ処理
# ==========================================
def load_data(filepath, seq_len, pred_len, target_col='OT'):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    df = df.dropna()
    df = df[[target_col]] # Univariate
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i : i+seq_len])
            y.append(data[i+seq_len : i+seq_len+pred_len])
        return np.array(X), np.array(y)

    print("Generating sequences...")
    X_train, y_train = create_sequences(train_scaled)
    X_test, y_test = create_sequences(test_scaled)
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(y_train), 
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)), scaler

# ==========================================
# 3. メイン処理 & グラフ描画
# ==========================================
def run_comparison(path):
    # 設定（長期予測の強みを出す設定）
    SEQ_LEN = 96
    PRED_LEN = 720  # 30日分 (ETTmの場合)
    EPOCHS = 50
    BATCH_SIZE = 32
    LR = 0.001
    
    print(f"Config: PRED_LEN={PRED_LEN} (Long-term forecasting)")
    
    # データロード
    (X_train, y_train, X_test, y_test), scaler = load_data(path, SEQ_LEN, PRED_LEN)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # --- 1. Linear (OLS) ---
    print("Training Linear Regression...")
    reg = LinearRegression()
    # Scikit-learn用に次元変換
    X_train_np = X_train[:, :, 0].numpy()
    y_train_np = y_train[:, :, 0].numpy()
    X_test_np = X_test[:, :, 0].numpy()
    
    reg.fit(X_train_np, y_train_np)
    pred_linear = reg.predict(X_test_np)

    # --- 2. NLinear ---
    print("Training NLinear...")
    model = NLinear(SEQ_LEN, PRED_LEN, input_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    pred_nlinear_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            pred_nlinear_list.append(out.cpu().numpy())
    pred_nlinear = np.concatenate(pred_nlinear_list, axis=0)[:, :, 0]

    # --- 評価 (HorizonごとのMAE計算) ---
    y_true = y_test[:, :, 0].numpy()
    
    # スケール戻し
    ot_std = scaler.scale_[0]
    ot_mean = scaler.mean_[0]
    
    real_true = y_true * ot_std + ot_mean
    real_lin = pred_linear * ot_std + ot_mean
    real_nlin = pred_nlinear * ot_std + ot_mean
    
    # Horizon(1~720)ごとの平均絶対誤差(MAE)を計算
    mae_lin_h = np.mean(np.abs(real_true - real_lin), axis=0)
    mae_nlin_h = np.mean(np.abs(real_true - real_nlin), axis=0)

    # ==========================================
    # ★ ここがスライド用グラフ生成部 ★
    # ==========================================
    print("Generating Slide-Ready Graph...")
    
    plt.style.use('default') # スタイルリセット
    fig, ax = plt.subplots(figsize=(10, 6)) # スライドに貼りやすいアスペクト比
    
    steps = range(1, PRED_LEN + 1)
    
    # Linear: グレー破線 (ベースラインであることを強調)
    ax.plot(steps, mae_lin_h, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='Linear (Baseline)')
    
    # NLinear: オレンジ太線 (主役であることを強調)
    ax.plot(steps, mae_nlin_h, color='#ED7D31', linewidth=3.5, label='NLinear (Proposed)')
    
    # 差分エリアの塗りつぶし (NLinearの優位性を視覚化)
    # 後半部分(360step以降)の差分を薄いオレンジで塗る
    ax.fill_between(steps, mae_lin_h, mae_nlin_h, 
                    where=(mae_lin_h > mae_nlin_h), 
                    interpolate=True, color='#ED7D31', alpha=0.1, label='Error Reduction')

    # ラベルとタイトル
    ax.set_title('Error Degradation Curve (Long-term Forecast)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Forecast Horizon (Steps)', fontsize=14)
    ax.set_ylabel('Mean Absolute Error (°C)', fontsize=14)
    
    # 目盛りを見やすく
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 凡例
    ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9)
    
    # 注釈 (矢印で差をアピール)
    end_diff = mae_lin_h[-1] - mae_nlin_h[-1]
    ax.annotate(f'Stable in Long-term\n(Diff: {end_diff:.2f}°C)', 
                xy=(PRED_LEN, mae_nlin_h[-1]), 
                xytext=(PRED_LEN-200, mae_nlin_h[-1]-0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7),
                fontsize=12, fontweight='bold', color='#333333')

    plt.tight_layout()
    
    # 保存
    save_path = 'nlinear_vs_linear_slide.png'
    plt.savefig(save_path, dpi=300) # 高解像度で保存
    print(f"Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # ETTm2推奨 (無ければm1)
    path = '../ETTm2.csv'
    if not os.path.exists(path):
        path = '../ETTm1.csv'
    if not os.path.exists(path):
         path = '../data/ETTm2.csv'
         if not os.path.exists(path):
             path = '../data/ETTm1.csv'
             
    run_comparison(path)