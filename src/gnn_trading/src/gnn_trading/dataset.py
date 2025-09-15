import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch.utils.data import Dataset

class OHLCVGraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window=60, pred_horizon=5, use_labels=True, scaler=None, news_feat=None):
        self.df = df.copy()
        self.window = window
        self.pred_h = pred_horizon
        self.use_labels = use_labels
        self.news_feat = news_feat  # optional dict: date->vector (per day)

        self.feat_cols = [c for c in self.df.columns if c not in ['breakout_label']]
        self.scaler = scaler or StandardScaler()
        X = self.df[self.feat_cols].values
        if scaler is None:
            self.scaler.fit(X)
        self.X_scaled = self.scaler.transform(X)

        self.labels = self.df['breakout_label'].astype(int).values if use_labels else None
        self.valid_idx = [i for i in range(window, len(self.df)-pred_horizon)]

    def __len__(self): return len(self.valid_idx)

    def _edge_index(self):
        src = torch.arange(0, self.window-1, dtype=torch.long)
        dst = torch.arange(1, self.window, dtype=torch.long)
        return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    def __getitem__(self, k):
        i = self.valid_idx[k]
        sl = slice(i-self.window, i)
        x = torch.tensor(self.X_scaled[sl, :], dtype=torch.float)
        edge_index = self._edge_index()
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(self.window, dtype=torch.long)
        if self.use_labels:
            future = self.labels[i:i+self.pred_h]
            y_cls = torch.tensor([1 if future.sum()>0 else 0], dtype=torch.long)
            # regression targets
            future_returns = (self.df['close'].iloc[i+self.pred_h-1] - self.df['close'].iloc[i-1]) / (self.df['close'].iloc[i-1] + 1e-12)
            future_vol = float(self.df['atr_pct'].iloc[i:i+self.pred_h].mean())
            data.y = y_cls
            data.y_reg = torch.tensor([future_returns], dtype=torch.float)
            data.y_vol = torch.tensor([future_vol], dtype=torch.float)
        # attach optional news embedding for the last day in window (late fusion)
        if self.news_feat is not None:
            dt = self.df.index[i-1]
            vec = self.news_feat.get(str(dt.date()), None)
            if vec is not None:
                data.news = torch.tensor(vec, dtype=torch.float)
        return data

    # for SSL
    def get_ssl_pair(self, k, feat_mask_ratio=0.15, jitter_std=0.01):
        i = self.valid_idx[k]
        sl = slice(i-self.window, i)
        x_np = self.X_scaled[sl, :].copy()
        def aug(x):
            X = x.copy()
            F = X.shape[1]
            mask = np.random.rand(F) < feat_mask_ratio
            X[:, mask] = 0.0
            X += np.random.randn(*X.shape) * jitter_std
            return X
        x1 = torch.tensor(aug(x_np), dtype=torch.float)
        x2 = torch.tensor(aug(x_np), dtype=torch.float)
        edge_index = self._edge_index()
        d1 = Data(x=x1, edge_index=edge_index); d1.batch = torch.zeros(self.window, dtype=torch.long)
        d2 = Data(x=x2, edge_index=edge_index); d2.batch = torch.zeros(self.window, dtype=torch.long)
        return d1, d2
