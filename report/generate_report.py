import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# srcフォルダからモジュールをインポート
from src.model import NLinear
from src.data_loader import DataManager

# ==========================================
# 設定 (Config)
# ==========================================

DATA_PATH = os.path.join(parent_dir, 'data', 'ETTm2.csv')

# もしETTm2がない場合はETTm1を使う（保険）
if not os.path.exists(DATA_PATH):
    print(f"ETTm2 not found at {DATA_PATH}, checking ETTm1...")
    DATA_PATH = os.path.join(parent_dir, 'data', 'ETTm1.csv')

print(f"[Info] Target Data Path: {DATA_PATH}")

SEQ_LEN = 336   # 過去3.5日分を入力
PRED_LEN = 96   # 未来24時間分を予測
TARGET_HOUR = 3 # メインで評価したい「n時間後」

BATCH_SIZE = 32
EPOCHS = 15     # デモ用
LR = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(train_loader, val_loader):
    print(">>> Model Training Start...")
    model = NLinear(SEQ_LEN, PRED_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")
            
    return model

def generate_graphs(model, test_loader, scaler):
    print(">>> Generating Graphs...")
    model.eval()
    
    preds = []
    trues = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx).cpu().numpy()
            preds.append(out)
            trues.append(by.numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    preds_real = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_real = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    target_idx = (TARGET_HOUR * 4) - 1
    
    save_dir = os.path.join(current_dir,"output")
    
    # --- Graph 1 ---
    plt.figure(figsize=(12, 5))
    start_plot = 500
    end_plot = 800
    series_true = trues_real[:, target_idx, 0]
    series_pred = preds_real[:, target_idx, 0]
    
    plt.plot(series_true[start_plot:end_plot], label=f'Actual Temp ({TARGET_HOUR}h later)', color='black', linewidth=2, alpha=0.7)
    plt.plot(series_pred[start_plot:end_plot], label=f'AI Prediction ({TARGET_HOUR}h later)', color='#ED7D31', linewidth=2, linestyle='--')
    
    plt.title(f'Graph 1: Forecast Tracking ({TARGET_HOUR} Hours Ahead)', fontsize=14, fontweight='bold')
    plt.ylabel('Temperature (°C)')
    plt.xlabel('Time Steps (15min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graph1_tracking.png'), dpi=300)
    print("Saved: graph1_tracking.png")

    # --- Graph 2 ---
    plt.figure(figsize=(10, 6))
    horizons_hr = [1, 3, 6, 12, 24]
    maes = []
    labels = []
    
    for h in horizons_hr:
        step = h * 4 - 1
        if step >= PRED_LEN: break
        mae = mean_absolute_error(trues_real[:, step, 0], preds_real[:, step, 0])
        maes.append(mae)
        labels.append(f"{h}h")
    
    bars = plt.bar(labels, maes, color='#ED7D31', width=0.6, alpha=0.9)
    plt.title('Graph 2: Prediction Error by Forecast Horizon', fontsize=14, fontweight='bold')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Mean Absolute Error (°C)')
    plt.ylim(0, max(maes) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}°C', ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graph2_mae_horizon.png'), dpi=300)
    print("Saved: graph2_mae_horizon.png")

    # --- Graph 3 ---
    plt.figure(figsize=(10, 6))
    errors = np.abs(series_pred - series_true)
    mean_err = np.mean(errors)
    p99_err = np.percentile(errors, 99)
    max_err = np.max(errors)
    
    plt.hist(errors, bins=50, color='#F4A460', alpha=0.8, edgecolor='white', label='Error Distribution')
    plt.axvline(mean_err, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}°C')
    plt.axvline(p99_err, color='red', linestyle='--', linewidth=2, label=f'99% Limit: {p99_err:.2f}°C')
    plt.axvline(max_err, color='black', linestyle=':', linewidth=2, label=f'Max: {max_err:.2f}°C')
    
    text_str = (f"Safety Buffer Logic:\n"
                f"99% of errors are < {p99_err:.2f}°C\n"
                f"Worst case is {max_err:.2f}°C")
    plt.gca().text(0.95, 0.75, text_str, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    plt.title(f'Graph 3: Error Distribution for {TARGET_HOUR}-Hour Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Error (°C)')
    plt.ylabel('Frequency (Count)')
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graph3_error_dist.png'), dpi=300)
    print("Saved: graph3_error_dist.png")

def main():
    try:
        dm = DataManager(DATA_PATH, target_col='OT')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    train_data, val_data, test_data = dm.get_processed_data()
    
    train_loader = dm.create_loaders(train_data, SEQ_LEN, PRED_LEN, BATCH_SIZE, shuffle=True)
    val_loader   = dm.create_loaders(val_data, SEQ_LEN, PRED_LEN, BATCH_SIZE, shuffle=False)
    test_loader  = dm.create_loaders(test_data, SEQ_LEN, PRED_LEN, BATCH_SIZE, shuffle=False)
    
    model = train_model(train_loader, val_loader)
    
    generate_graphs(model, test_loader, dm.scaler)
    
    print("\n>>> All tasks completed successfully.")

if __name__ == "__main__":
    main()