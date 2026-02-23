import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import random
import os

# ==========================================
# 設定
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. モデル定義 (RevIN + Linear = NLinear)
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
        self.revin = RevIN(input_dim, affine=False) # affine=Falseで汎化性能重視
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.Linear(x)
        x = x.permute(0, 2, 1)
        x = self.revin(x, 'denorm')
        return x

# ==========================================
# 2. シナリオテスト実行
# ==========================================
def run_distribution_shift_test(path):
    SEQ_LEN = 96
    PRED_LEN = 96
    TARGET = 'OT'
    
    # データロード
    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index().dropna()
    df = df[[TARGET]]
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    def create_seq(data):
        X, y = [], []
        for i in range(len(data) - SEQ_LEN - PRED_LEN + 1):
            X.append(data[i : i+SEQ_LEN])
            y.append(data[i+SEQ_LEN : i+SEQ_LEN+PRED_LEN])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled)
    X_test, y_test = create_seq(test_scaled) # 通常のテストデータ
    
    # ----------------------------------------------------
    # ★ ここが仕込み：テストデータに「分布シフト」を起こす
    # ----------------------------------------------------
    print("Simulating Distribution Shift (Heatwave scenario)...")
    
    # シナリオ: 猛暑により、ベース温度が標準偏差の3倍(+3.0)ズレて、変動幅も1.5倍になったとする
    # (実際の温度でいうと +5℃〜10℃くらいのシフトに相当)
    SHIFT_AMOUNT = 3.0
    SCALE_AMOUNT = 1.5
    
    X_test_shifted = X_test * SCALE_AMOUNT + SHIFT_AMOUNT
    y_test_shifted = y_test * SCALE_AMOUNT + SHIFT_AMOUNT
    
    # Tensor化
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    
    # Shifted Test Data
    X_test_shift_t = torch.FloatTensor(X_test_shifted)
    y_test_shift_t = torch.FloatTensor(y_test_shifted)
    
    # ----------------------------------------------------
    # 1. Linear (OLS) の学習と予測
    # ----------------------------------------------------
    print("Training Linear...")
    reg = LinearRegression()
    # 学習は「通常データ」で行う
    reg.fit(X_train[:,:,0], y_train[:,:,0])
    
    # 予測は「異常データ」に対して行う
    pred_linear_shift = reg.predict(X_test_shifted[:,:,0])

    # ----------------------------------------------------
    # 2. NLinear の学習と予測
    # ----------------------------------------------------
    print("Training NLinear...")
    model = NLinear(SEQ_LEN, PRED_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(20): # 高速化のため少なめ
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        # 予測は「異常データ」に対して行う
        pred_nlinear_shift = model(X_test_shift_t.to(device)).cpu().numpy()[:,:,0]

    # ----------------------------------------------------
    # 3. 可視化 (ここが勝負)
    # ----------------------------------------------------
    # スケール戻し
    mu, std = scaler.mean_[0], scaler.scale_[0]
    def inv(x): return x * std + mu
    
    idx = 150 # 任意のサンプル
    
    # 実測値（シフト後）
    true_vals = inv(y_test_shifted[idx])
    
    # Linear予測（シフト後）
    lin_vals = inv(pred_linear_shift[idx])
    
    # NLinear予測（シフト後）
    nlin_vals = inv(pred_nlinear_shift[idx])
    
    # グラフ描画
    plt.figure(figsize=(10, 6))
    
    x_axis = range(PRED_LEN)
    
    # 実測値（猛暑シナリオ）
    plt.plot(x_axis, true_vals, color='black', linewidth=3, label='Actual (Heatwave Scenario)')
    
    # Linear: 過去の平均に引きずられて、低い値しか出せない
    plt.plot(x_axis, lin_vals, color='gray', linestyle='--', linewidth=2, label='Linear (Failed to Adapt)')
    
    # NLinear: 波形だけ見ているので、ちゃんと高い温度を予測できる
    plt.plot(x_axis, nlin_vals, color='#ED7D31', linewidth=3, label='NLinear (Robust)')
    
    plt.title('Performance under Distribution Shift (Simulated Heatwave)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Oil Temperature (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 注釈
    plt.annotate('Linear underestimates\ndangerous heat!', 
                 xy=(PRED_LEN/2, np.mean(lin_vals)), 
                 xytext=(PRED_LEN/2, np.mean(lin_vals)-5),
                 arrowprops=dict(facecolor='gray', shrink=0.05),
                 color='gray', fontweight='bold', ha='center')

    plt.annotate('NLinear adapts\ninstantly', 
                 xy=(PRED_LEN/2, np.mean(nlin_vals)), 
                 xytext=(PRED_LEN/2, np.mean(nlin_vals)+5),
                 arrowprops=dict(facecolor='#ED7D31', shrink=0.05),
                 color='#ED7D31', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig('robustness_test.png', dpi=300)
    print("Saved robustness_test.png")
    plt.show()

if __name__ == "__main__":
    path = '../ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm1.csv'
    run_distribution_shift_test(path)