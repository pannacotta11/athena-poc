import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

class DataManager:
    def __init__(self, data_path, target_col='OT'):
        self.data_path = data_path
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.df = None
        
        # データのロード
        self._load_source()

    def _load_source(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path}")
            
        print(f"[DataManager] Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, parse_dates=['date'])
        self.df = self.df.set_index('date').sort_index().dropna()
        self.df = self.df[[self.target_col]] # Univariate

    def get_processed_data(self, split_ratios=(0.7, 0.1, 0.2)):
        """
        Train/Val/Testに分割し、スケーリングして返す
        """
        n = len(self.df)
        train_end = int(n * split_ratios[0])
        val_end = int(n * (split_ratios[0] + split_ratios[1]))
        
        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]
        
        # Trainデータのみでfitする（リーク防止）
        train_scaled = self.scaler.fit_transform(train_df)
        val_scaled = self.scaler.transform(val_df)
        test_scaled = self.scaler.transform(test_df)
        
        return train_scaled, val_scaled, test_scaled

    def create_loaders(self, data, seq_len, pred_len, batch_size, shuffle=True):
        """
        時系列データをスライディングウィンドウ処理し、DataLoaderを返す
        """
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i : i+seq_len])
            y.append(data[i+seq_len : i+seq_len+pred_len])
            
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader

    def inverse_transform(self, data):
        """
        スケーリングされたデータを元の値(℃)に戻す
        """
        return self.scaler.inverse_transform(data)