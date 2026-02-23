import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    def __init__(self, seq_len, pred_len, input_dim):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN(input_dim, affine=False)
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.Linear(x)
        x = x.permute(0, 2, 1)
        x = self.revin(x, 'denorm')
        return x

def load_and_process_data(filepath, seq_len, pred_len, target_col='OT'):
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    df = df.dropna()
    df = df[[target_col]] 
    input_dim = 1
    
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
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    return (X_train_t, y_train_t, X_test_t, y_test_t), scaler, input_dim

def run_comparison(filepath):
    PRED_LEN = 720
    SEQ_LEN = 96
    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 0.0005
    TARGET = 'OT'
    
    print(f"=== Config: Seq={SEQ_LEN}, Pred={PRED_LEN}, Epochs={EPOCHS}, Mode=Univariate ===")
    
    (X_train, y_train, X_test, y_test), scaler, input_dim = \
        load_and_process_data(filepath, SEQ_LEN, PRED_LEN, TARGET)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n--- Training Linear (vanilla OLS) ---")
    X_train_lin = X_train[:, :, 0].numpy()
    y_train_lin = y_train[:, :, 0].numpy()
    X_test_lin = X_test[:, :, 0].numpy()
    
    reg = LinearRegression()
    reg.fit(X_train_lin, y_train_lin)
    pred_linear = reg.predict(X_test_lin)

    print("\n--- Training NLinear ---")
    model = NLinear(SEQ_LEN, PRED_LEN, input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    pred_nlinear_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            pred_nlinear_list.append(out.cpu().numpy())
            
    pred_nlinear = np.concatenate(pred_nlinear_list, axis=0)[:, :, 0]

    print("\n--- Evaluation ---")
    y_true = y_test[:, :, 0].numpy()
    
    ot_std = scaler.scale_[0]
    ot_mean = scaler.mean_[0]
    
    def inverse(d): return d * ot_std + ot_mean
    
    real_true = inverse(y_true)
    real_lin = inverse(pred_linear)
    real_nlin = inverse(pred_nlinear)
    
    mae_lin = mean_absolute_error(real_true, real_lin)
    mae_nlin = mean_absolute_error(real_true, real_nlin)
    
    print(f"Linear (OLS) MAE   : {mae_lin:.4f}")
    print(f"NLinear (Tuned) MAE: {mae_nlin:.4f}")
    
    if mae_nlin < mae_lin:
        diff = mae_lin - mae_nlin
        print(f"\n>> WINNER: NLinear! (Beat Linear by {diff:.4f} ℃)")
    else:
        print("\n>> Linear Win.")

    mae_lin_h = np.mean(np.abs(real_true - real_lin), axis=0)
    mae_nlin_h = np.mean(np.abs(real_true - real_nlin), axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(mae_lin_h, label='Linear (OLS)', linestyle='--', color='gray')
    plt.plot(mae_nlin_h, label='NLinear', linewidth=2, color='#ED7D31')
    plt.title(f'Prediction Error over {PRED_LEN} Steps')
    plt.xlabel('Future Time Steps')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    idx = 200
    if idx >= len(real_true): idx = 0
    
    plt.figure(figsize=(12, 5))
    past_x = np.arange(-100, 0)
    past_y = inverse(X_test[idx, -100:, 0].numpy())
    future_x = np.arange(0, PRED_LEN)
    
    plt.plot(past_x, past_y, color='black', alpha=0.3, label='History')
    plt.plot(future_x, real_true[idx], color='black', alpha=0.8, label='Actual')
    plt.plot(future_x, real_lin[idx], color='gray', linestyle='--', label='Linear')
    plt.plot(future_x, real_nlin[idx], color='#ED7D31', linewidth=2, label='NLinear')
    
    plt.title(f'Forecast Visualization (Sample #{idx})')
    plt.axvline(0, color='silver', linestyle='-')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    path = '../ETTm2.csv'
    if not os.path.exists(path):
        print(f"ETTm2 not found at {path}, trying ETTm1...")
        path = '../ETTm1.csv'
    if not os.path.exists(path):
         path = '../data/ETTm2.csv'
         if not os.path.exists(path):
             path = '../data/ETTm1.csv'

    try:
        run_comparison(path)
    except Exception as e:
        print(f"Error: {e}")