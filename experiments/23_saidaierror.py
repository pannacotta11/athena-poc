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
# 1. モデル定義
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
# 2. メイン処理
# ==========================================
def analyze_max_error(path):
    # 設定
    SEQ_LEN = 96
    PRED_LEN = 96  # 24時間分
    TARGET = 'OT'
    
    # 3時間後とは？ (15分データなら 3時間 * 4step = 12step目)
    # インデックスは0始まりなので 11
    is_minute = 'm' in os.path.basename(path)
    steps_3h = (3 * 4) if is_minute else 3
    target_idx = steps_3h - 1
    
    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index().dropna()
    df = df[[TARGET]]
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    def create_seq(data):
        X, y = [], []
        for i in range(len(data) - SEQ_LEN - PRED_LEN + 1):
            X.append(data[i : i+SEQ_LEN])
            y.append(data[i+SEQ_LEN : i+SEQ_LEN+PRED_LEN])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    X_train, y_train = create_seq(train_scaled)
    X_test, y_test = create_seq(test_scaled)
    
    # 学習
    print("Training NLinear...")
    model = NLinear(SEQ_LEN, PRED_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model.train()
    for epoch in range(10): 
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    # 推論 (3時間後だけを抽出)
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    errors_3h = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx).cpu().numpy() # [Batch, Pred, 1]
            true = by.numpy()             # [Batch, Pred, 1]
            
            # 3時間後の値
            pred_3h = out[:, target_idx, 0]
            true_3h = true[:, target_idx, 0]
            
            # 絶対誤差
            abs_err = np.abs(pred_3h - true_3h)
            errors_3h.extend(abs_err)
            
    # 配列化してスケール戻し
    errors_3h = np.array(errors_3h)
    std = scaler.scale_[0]
    errors_real = errors_3h * std # ℃単位の誤差
    
    # ==========================================
    # 統計量の計算
    # ==========================================
    max_error = np.max(errors_real)
    mean_error = np.mean(errors_real)
    p95_error = np.percentile(errors_real, 95)
    p99_error = np.percentile(errors_real, 99)
    
    print("\n" + "="*40)
    print(f" ERROR ANALYSIS (3 Hours Ahead)")
    print("="*40)
    print(f" Mean Absolute Error (MAE): {mean_error:.4f} °C")
    print(f" 95% Percentile Error   : {p95_error:.4f} °C")
    print(f" 99% Percentile Error   : {p99_error:.4f} °C")
    print(f" MAX Error (Worst Case) : {max_error:.4f} °C")
    print("="*40)

    # ==========================================
    # グラフ描画 (ヒストグラム)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # ヒストグラム
    plt.hist(errors_real, bins=50, color='#ED7D31', alpha=0.7, edgecolor='white', label='Error Distribution')
    
    # 垂直線
    plt.axvline(mean_error, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}°C')
    plt.axvline(p99_error, color='red', linestyle='--', linewidth=2, label=f'99% Limit: {p99_error:.2f}°C')
    plt.axvline(max_error, color='black', linestyle=':', linewidth=2, label=f'Max: {max_error:.2f}°C')
    
    plt.title(f'Error Distribution for 3-Hour Forecast', fontsize=16, fontweight='bold')
    plt.xlabel('Absolute Error (°C)', fontsize=14)
    plt.ylabel('Frequency (Count)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # テキスト注釈
    text_str = (f"Safety Buffer Logic:\n"
                f"99% of errors are < {p99_error:.2f}°C\n"
                f"Worst case is {max_error:.2f}°C")
    
    plt.text(0.95, 0.75, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('error_distribution_3h.png', dpi=300)
    print("Saved error_distribution_3h.png")
    plt.show()

if __name__ == "__main__":
    path = '../ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm1.csv'
    analyze_max_error(path)