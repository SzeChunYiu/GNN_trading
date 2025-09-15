import pandas as pd

def make_breakout_labels(df: pd.DataFrame, lookback=20, vol_z_threshold=0.0, breakout_pct=0.01, confirm_days=0):
    rollmax = df['high'].shift(1).rolling(lookback).max()
    cond_price = df['close'] > rollmax * (1 + breakout_pct)
    cond_vol = df['vol_z20'] > vol_z_threshold
    raw = (cond_price & cond_vol)
    if confirm_days <= 0:
        return raw.astype(int)
    arr = raw.values; idx = df.index
    out = [False]*len(arr)
    for i in range(len(arr)-confirm_days):
        if arr[i] and cond_price.iloc[i:i+confirm_days+1].all():
            out[i] = True
    return pd.Series(out, index=idx).astype(int)
