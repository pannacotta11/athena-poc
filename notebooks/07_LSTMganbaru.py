import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ==========================================
# 1. データ準備 (Horizon対応)
# ==========================================
def prepare_data_lstm(filepath, horizon, seq_len=24):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # 特徴量（Cyclic + Raw）
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # ターゲット: Horizon先
    df['Target'] = df['OT'].shift(-horizon)
    df = df.dropna()
    
    # スケーリング (NNは必須)
    feature_cols = [c for c in df.columns if c != 'Target']
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[['Target']])
    
    # Sequenceデータ作成 (過去 seq_len 時間分を入力とする)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i : i+seq_len])
        y_seq.append(y_scaled[i+seq_len])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Split
    split = int(len(X_seq) * 0.8)
    
    # Tensor化
    train_dataset = TensorDataset(
        torch.FloatTensor(X_seq[:split]), torch.FloatTensor(y_seq[:split])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_seq[split:]), torch.FloatTensor(y_seq[split:])
    )
    
    return train_dataset, test_dataset, scaler_y

# ==========================================
# 2. LSTMモデル定義
# ==========================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)
        # h_n: [1, batch, hidden_dim] -> 最後の時刻の隠れ層を使う
        out = self.fc(h_n[-1])
        return out

# ==========================================
# 3. 学習 & 評価ループ
# ==========================================
def run_lstm_experiment(filepath, horizon=24):
    # 設定
    SEQ_LEN = 24  # 過去24時間を見て、Horizon時間後を予測
    BATCH_SIZE = 64
    EPOCHS = 10 # 時間なければ減らす
    LR = 0.001
    
    # データ
    train_ds, test_ds, scaler_y = prepare_data_lstm(filepath, horizon, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデル
    input_dim = train_ds[0][0].shape[1]
    model = SimpleLSTM(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 学習
    print(f"Training LSTM for Horizon={horizon}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
            
    # 評価
    model.eval()
    preds_list = []
    actuals_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            preds_list.append(preds.numpy())
            actuals_list.append(y_batch.numpy())
            
    # スケールを戻す
    preds_inv = scaler_y.inverse_transform(np.concatenate(preds_list))
    actuals_inv = scaler_y.inverse_transform(np.concatenate(actuals_list))
    
    mae = mean_absolute_error(actuals_inv, preds_inv)
    print(f"LSTM Horizon {horizon} MAE: {mae:.4f}")
    return mae

# --- 実行 ---
if __name__ == "__main__":
    path = '../data/ETTh1.csv'
    
    # 試しにHorizon 24時間でやってみる
    run_lstm_experiment(path, horizon=24)