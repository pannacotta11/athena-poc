import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
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
# 1. モデル定義 (NLinear)
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
# 2. データ処理
# ==========================================
def run_client_visualization(path):
    # ETTm (15分足) を想定
    # 1時間=4step, 3時間=12step, 5時間=20step
    is_minute_data = 'm' in os.path.basename(path)
    steps_per_hour = 4 if is_minute_data else 1
    
    SEQ_LEN = 96
    PRED_LEN = 96  # 24時間分予測
    TARGET = 'OT'
    
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
    
    # 学習 (簡易版)
    print("Training NLinear Model...")
    model = NLinear(SEQ_LEN, PRED_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model.train()
    for epoch in range(10): # デモ用に短縮
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    # 推論
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    pred_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            pred_list.append(out.cpu().numpy())
    pred_array = np.concatenate(pred_list, axis=0) # [Samples, Pred_Len, 1]
    
    # 実測値
    true_array = y_test.numpy()
    
    # スケール戻し
    mu, std = scaler.mean_[0], scaler.scale_[0]
    pred_real = pred_array * std + mu
    true_real = true_array * std + mu
    
    # ==========================================
    # グラフ1: 予測時間ごとのMAE推移 (棒グラフ)
    # ==========================================
    # 評価したい時間 (Hours)
    target_hours = [1, 3, 5, 12, 24]
    maes = []
    labels = []
    
    print("\n=== Accuracy Report (MAE) ===")
    for h in target_hours:
        step = h * steps_per_hour
        if step > PRED_LEN: break
        
        # 配列インデックスは step-1
        idx = step - 1
        
        mae = mean_absolute_error(true_real[:, idx, 0], pred_real[:, idx, 0])
        maes.append(mae)
        labels.append(f'{h}h')
        print(f"{h} hours ahead MAE: {mae:.4f} °C")
        
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, maes, color='#ED7D31', alpha=0.9, width=0.6)
    
    plt.title('Prediction Error by Forecast Horizon', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Hours Ahead', fontsize=14)
    plt.ylabel('Mean Absolute Error (°C)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, max(maes)*1.2)
    
    # 値を表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}°C',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('mae_by_horizon.png', dpi=300)
    print("Saved mae_by_horizon.png")
    plt.show()

    # ==========================================
    # グラフ2: 3時間後予測の追従性 (時系列プロット)
    # ==========================================
    target_h = 3
    target_step = target_h * steps_per_hour - 1 # index
    
    # 3時間後の予測値と実測値のシリーズ
    series_pred = pred_real[:, target_step, 0]
    series_true = true_real[:, target_step, 0]
    
    # わかりやすい区間を切り出し (例: 300ポイント分)
    start_idx = 500
    end_idx = 800
    
    plt.figure(figsize=(12, 5))
    x_axis = range(end_idx - start_idx)
    
    plt.plot(x_axis, series_true[start_idx:end_idx], label='Actual Temperature', 
             color='black', linewidth=2.5, alpha=0.8)
    plt.plot(x_axis, series_pred[start_idx:end_idx], label=f'NLinear Prediction ({target_h}h ahead)', 
             color='#ED7D31', linewidth=2.5, linestyle='--')
    
    plt.title(f'Tracking Performance: {target_h}-Hour Ahead Forecast', fontsize=16, fontweight='bold')
    plt.ylabel('Oil Temperature (°C)', fontsize=14)
    plt.xlabel('Time Steps (15-min intervals)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 精度アピールの注釈
    mae_3h = maes[target_hours.index(3)]
    plt.annotate(f'Avg Error: Only {mae_3h:.2f}°C', 
                 xy=(0.02, 0.9), xycoords='axes fraction',
                 fontsize=14, fontweight='bold', color='#E65100',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ED7D31'))
    
    plt.tight_layout()
    plt.savefig('forecast_tracking_3h.png', dpi=300)
    print("Saved forecast_tracking_3h.png")
    plt.show()

if __name__ == "__main__":
    path = '../data/ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm2.csv'
    if not os.path.exists(path): path = '../data/ETTm1.csv'
    
    try:
        run_client_visualization(path)
    except Exception as e:
        print(e)