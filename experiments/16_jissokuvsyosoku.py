import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
# 1. NLinear & RevIN (最適化版)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False): 
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
    def __init__(self, seq_len, pred_len, input_dim=1):
        super(NLinear, self).__init__()
        self.revin = RevIN(input_dim, affine=False)
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.Linear(x)
        x = x.permute(0, 2, 1)
        x = self.revin(x, 'denorm')
        return x

# ==========================================
# 2. データロード & 学習
# ==========================================
def run_visualization(path):
    # 設定
    SEQ_LEN = 96
    PRED_LEN = 336 # 2週間 (ETTmの場合3.5日)
    EPOCHS = 50
    BATCH_SIZE = 32
    LR = 0.005
    TARGET = 'OT'
    
    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index().dropna()
    df = df[[TARGET]] # Univariate
    
    # Split & Scale
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    # Sequence作成
    def create_seq(data):
        X, y = [], []
        for i in range(len(data) - SEQ_LEN - PRED_LEN + 1):
            X.append(data[i : i+SEQ_LEN])
            y.append(data[i+SEQ_LEN : i+SEQ_LEN+PRED_LEN])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    X_train, y_train = create_seq(train_scaled)
    X_test, y_test = create_seq(test_scaled)
    
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 学習
    print("Training NLinear...")
    model = NLinear(SEQ_LEN, PRED_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    # ==========================================
    # 3. グラフ描画 (ここがメイン)
    # ==========================================
    print("Generating Forecast Plots...")
    model.eval()
    
    # スケール戻し用関数
    mu, std = scaler.mean_[0], scaler.scale_[0]
    def inv(x): return x * std + mu

    # 描画したいサンプルのインデックス (変動がある場所を選ぶ)
    # ETTm2の場合、データが多いので適当に分散させる
    sample_indices = [100, 450] 
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 5 * len(sample_indices)))
    if len(sample_indices) == 1: axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # データ取得
        input_x = X_test[idx:idx+1].to(device) # [1, Seq, 1]
        true_y = y_test[idx:idx+1].numpy().flatten() # [Pred]
        
        with torch.no_grad():
            pred_y = model(input_x).cpu().numpy().flatten()
            
        # スケール戻し
        history = inv(input_x.cpu().numpy().flatten())
        true_y = inv(true_y)
        pred_y = inv(pred_y)
        
        # X軸の作成
        x_history = np.arange(-SEQ_LEN, 0)
        x_future = np.arange(0, PRED_LEN)
        
        # --- プロット ---
        # 1. 過去データ (入力)
        ax.plot(x_history, history, color='black', alpha=0.4, label='History (Input)')
        
        # 2. 実測値 (正解)
        ax.plot(x_future, true_y, color='black', linewidth=2, label='Actual (Ground Truth)')
        
        # 3. NLinear予測
        ax.plot(x_future, pred_y, color='#ED7D31', linewidth=2.5, label='NLinear Prediction')
        
        # 4. 境界線
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)
        
        # デザイン調整
        ax.set_title(f'NLinear Forecast vs Actual (Sample #{idx})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Oil Temperature (°C)', fontsize=12)
        if i == len(sample_indices) - 1:
            ax.set_xlabel('Time Steps (Future)', fontsize=12)
            
        ax.legend(loc='upper left', frameon=True, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 予測精度の表示
        mae = np.mean(np.abs(true_y - pred_y))
        ax.text(0.02, 0.05, f'MAE: {mae:.2f}°C', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('nlinear_forecast_plot.png', dpi=300)
    print("Saved to nlinear_forecast_plot.png")
    plt.show()

if __name__ == "__main__":
    # パス設定
    path = '../ETTm2.csv'
    if not os.path.exists(path): path = '../ETTm1.csv'
    if not os.path.exists(path): path = '../data/ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm1.csv'
    
    run_visualization(path)