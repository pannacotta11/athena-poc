import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ==========================================
# 1. NLinearモデル定義 (超シンプル)
# ==========================================
class NLinear(nn.Module):
    """
    Normalization-Linear
    入力の最後の値を引いて0付近にし、予測後に足し戻すことで
    「分布シフト（Distribution Shift）」に強くする。
    """
    def __init__(self, seq_len, pred_len, input_dim):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 単純な線形層 (Linear Regressionと同じ)
        self.Linear = nn.Linear(seq_len, pred_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        
        # 1. Normalization (最後の値を引く)
        seq_last = x[:,-1:,:].detach() # [Batch, 1, Channel]
        x = x - seq_last

        # 2. Linear Prediction
        # チャネルごとに線形変換したいので転置
        x = x.permute(0, 2, 1) # [Batch, Channel, Seq_Len]
        x = self.Linear(x)     # [Batch, Channel, Pred_Len]
        x = x.permute(0, 2, 1) # [Batch, Pred_Len, Channel]

        # 3. De-Normalization (最後の値を足す)
        x = x + seq_last
        return x

# ==========================================
# 2. データ準備
# ==========================================
def prepare_data_nlinear(filepath, seq_len=96, pred_len=96):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # ターゲットはOTだが、全変数(Multivariate)を入力にしてOTを予測する設定にする
    # NLinearは本来「全変数を独立して予測(Channel Independence)」が得意
    data = df.drop(columns=['date'], errors='ignore').values
    
    # 標準化 (NLinearには必須ではないが、学習安定のため)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    X, y = [], []
    # seq_len分見て、pred_len分先までを一気に予測する
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i+seq_len])
        y.append(data[i+seq_len : i+seq_len+pred_len])
        
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y))
    
    # Split
    train_size = int(len(X) * 0.8)
    train_ds = TensorDataset(X[:train_size], y[:train_size])
    test_ds = TensorDataset(X[train_size:], y[train_size:])
    
    return train_ds, test_ds, scaler

# ==========================================
# 3. 学習ループ
# ==========================================
def run_nlinear(filepath):
    SEQ_LEN = 96   # 過去4日分 (96時間) を見て
    PRED_LEN = 24  # 未来1日分 (24時間) を予測する
    
    train_ds, test_ds, scaler = prepare_data_nlinear(filepath, SEQ_LEN, PRED_LEN)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # モデル初期化
    # input_dimはデータの列数（OT, HUFL, ... 全7列）
    input_dim = train_ds[0][0].shape[1]
    model = NLinear(SEQ_LEN, PRED_LEN, input_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Start Training NLinear...")
    for epoch in range(10): # 時間なければ5回とかでOK
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
            
    # 評価
        model.eval()
        total_mae = 0
        count = 0
        
        # OTは最後の列（index = -1）
        ot_idx = -1 
        
        # 標準偏差を取得（スケールを戻すため）
        # scaler.scale_ には各カラムの標準偏差が入っている
        ot_std = scaler.scale_[ot_idx]
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                
                pred_ot = output[:, :, ot_idx]
                true_ot = batch_y[:, :, ot_idx]
                
                # Scaledの差分を計算
                diff = torch.abs(pred_ot - true_ot)
                
                # 標準偏差を掛けて、元の「℃」に戻す
                mae_in_degree = diff * ot_std
                
                total_mae += mae_in_degree.mean().item()
                count += 1
                
        final_mae = total_mae / count
        print(f"NLinear MAE (Original Scale): {final_mae:.4f} ℃")
        
        if final_mae < 1.96:
            print(">> 判定: Linear Regression (Baseline) に勝利しました！")
        else:
            print(">> 判定: 惜しい！もう少しチューニングが必要です。")

if __name__ == "__main__":
    path = '../data/ETTh1.csv'
    run_nlinear(path)