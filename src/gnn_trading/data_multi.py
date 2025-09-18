"""Lightweight dataset for multi-company window sampling."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


class MultiCompanyDataset(Dataset):
    def __init__(
        self,
        frames: Dict[str, pd.DataFrame],
        window: int,
        horizons: Sequence[int],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        fit_end_date: Optional[pd.Timestamp] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        if not frames:
            raise ValueError("Expected at least one symbol frame")
        self.frames = frames
        self.window = int(window)
        self.horizons = sorted(int(h) for h in horizons)
        if not self.horizons:
            raise ValueError("At least one horizon is required")
        self.start_date = start_date
        self.end_date = end_date
        self.fit_end_date = fit_end_date or end_date
        self.max_h = max(self.horizons)
        self.symbols = sorted(frames.keys())
        self.symbol_to_id = {sym: idx for idx, sym in enumerate(self.symbols)}
        sample_df = frames[self.symbols[0]]
        prefixes = tuple({"ret_h", "vol_h", "breakout_h", "breakout_label"})
        self.feature_cols = [c for c in sample_df.columns if not c.startswith(prefixes)]
        self.label_cols = {
            h: {"cls": f"breakout_h{h}", "ret": f"ret_h{h}", "vol": f"vol_h{h}"} for h in self.horizons
        }
        self.scaler = scaler or StandardScaler()
        if scaler is None:
            self._fit_scaler()
        self.edge_index = self._line_graph(self.window)
        self.entries: List[Tuple[str, int]] = []
        self.meta_dates: List[pd.Timestamp] = []
        self._build_index()

    def _fit_scaler(self) -> None:
        buffers = []
        for df in self.frames.values():
            subset = df if self.fit_end_date is None else df.loc[: self.fit_end_date]
            arr = subset[self.feature_cols].dropna().values
            if arr.size:
                buffers.append(arr)
        if not buffers:
            raise ValueError("Unable to fit scaler; no numeric data available")
        self.scaler.fit(np.vstack(buffers))

    def _build_index(self) -> None:
        for sym in self.symbols:
            df = self.frames[sym]
            for idx in range(self.window, len(df) - self.max_h):
                date = df.index[idx]
                if self.start_date and date < self.start_date:
                    continue
                if self.end_date and date > self.end_date:
                    break
                window_slice = df.iloc[idx - self.window : idx][self.feature_cols]
                if window_slice.isna().any().any():
                    continue
                row = df.iloc[idx]
                if any(pd.isna(row[self.label_cols[h][key]]) for h in self.horizons for key in self.label_cols[h]):
                    continue
                self.entries.append((sym, idx))
                self.meta_dates.append(date)

    @staticmethod
    def _line_graph(window: int) -> torch.Tensor:
        src = torch.arange(0, window - 1, dtype=torch.long)
        dst = src + 1
        return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sym, pos = self.entries[idx]
        df = self.frames[sym]
        window_slice = df.iloc[pos - self.window : pos][self.feature_cols]
        features = self.scaler.transform(window_slice.values)
        row = df.iloc[pos]
        y_cls = [int(row[self.label_cols[h]["cls"]]) for h in self.horizons]
        y_ret = [float(row[self.label_cols[h]["ret"]]) for h in self.horizons]
        y_vol = [float(row[self.label_cols[h]["vol"]]) for h in self.horizons]
        return {
            "x": torch.tensor(features, dtype=torch.float32),
            "y_breakout": torch.tensor(y_cls, dtype=torch.long),
            "y_ret": torch.tensor(y_ret, dtype=torch.float32),
            "y_vol": torch.tensor(y_vol, dtype=torch.float32),
            "symbol_id": torch.tensor(self.symbol_to_id[sym], dtype=torch.long),
            "symbol": sym,
            "date": self.meta_dates[idx],
            "close": float(row["close"]),
        }


__all__ = ["MultiCompanyDataset"]
