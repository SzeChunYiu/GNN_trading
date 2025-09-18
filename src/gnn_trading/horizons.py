"""Utilities for generating multi-horizon labels and coverage reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class HorizonCoverage:
    horizon: int
    coverage_ratio: float
    available_rows: int


def compute_horizon_labels(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Adds return/volatility/breakout labels for each horizon."""

    out = df.copy()
    returns = out["close"].pct_change()
    for horizon in horizons:
        out[f"ret_h{horizon}"] = out["close"].pct_change(horizon).shift(-horizon)
        out[f"vol_h{horizon}"] = returns.rolling(horizon).std().shift(-horizon)
        window_high = out["high"].rolling(horizon, min_periods=horizon).max()
        breakout = (out["close"] > window_high.shift(1) * 1.01).astype(float)
        out[f"breakout_h{horizon}"] = breakout.shift(-horizon)
    return out


def compute_coverage(df: pd.DataFrame, horizons: Sequence[int]) -> List[HorizonCoverage]:
    coverages: List[HorizonCoverage] = []
    total = len(df)
    for horizon in horizons:
        label_col = f"ret_h{horizon}"
        if label_col not in df:
            coverage = 0.0
            available = 0
        else:
            valid = df[label_col].dropna()
            available = len(valid)
            coverage = available / max(total, 1)
        coverages.append(HorizonCoverage(horizon=horizon, coverage_ratio=coverage, available_rows=available))
    return coverages


def write_manifest(panel_df: pd.DataFrame, manifest_path: Path, horizons: Sequence[int]) -> None:
    manifest = {
        "row_count": int(len(panel_df)),
        "horizons": {
            str(h): {
                "ret_col": f"ret_h{h}",
                "vol_col": f"vol_h{h}",
                "breakout_col": f"breakout_h{h}",
            }
            for h in horizons
        },
    }
    import json

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_panel(symbol: str, database_root: Path, panel_df: pd.DataFrame) -> Path:
    out_path = database_root / "companies" / symbol / "gold" / "panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(out_path)
    return out_path


def build_coverage_heatmap(coverages: Dict[str, List[HorizonCoverage]]) -> pd.DataFrame:
    records = []
    for symbol, covs in coverages.items():
        for cov in covs:
            records.append({"ticker": symbol, "horizon": cov.horizon, "coverage": cov.coverage_ratio})
    return pd.DataFrame(records)


__all__ = [
    "HorizonCoverage",
    "compute_horizon_labels",
    "compute_coverage",
    "write_manifest",
    "save_panel",
    "build_coverage_heatmap",
]
