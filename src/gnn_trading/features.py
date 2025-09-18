import numpy as np
import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret1'] = df['close'].pct_change()
    df['logret'] = np.log(df['close']).diff()
    tr1 = (df['high'] - df['low'])
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low']  - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['atr_pct'] = df['atr14'] / (df['close'].shift(1) + 1e-12)
    for n in [5,10,20,50]:
        df[f'roll_max_{n}'] = df['high'].rolling(n).max()
        df[f'roll_min_{n}'] = df['low'].rolling(n).min()
        df[f'roll_mean_{n}'] = df['close'].rolling(n).mean()
        df[f'roll_std_{n}']  = df['close'].rolling(n).std()
        df[f'vol_mean_{n}']  = df['volume'].rolling(n).mean()
        df[f'vol_std_{n}']   = df['volume'].rolling(n).std()
    delta = df['close'].diff()
    up = np.where(delta>0, delta, 0.0)
    dn = np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(up, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    roll_dn = pd.Series(dn, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    df['rsi14'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd'] = macd
    df['macd_sig'] = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    for n in [10,20,50]:
        df[f'vol_z{n}'] = (df['volume'] - df[f'vol_mean_{n}']) / (df[f'vol_std_{n}'] + 1e-12)
    for n in [5,10,20]:
        rng = df[f'roll_max_{n}'] - df[f'roll_min_{n}']
        df[f'compr_{n}'] = rng / (df['close'] + 1e-12)
    def slope(series, win=10):
        idx = np.arange(win)
        out = [np.nan]*(win-1)
        arr = series.values
        for i in range(win-1, len(arr)):
            y = arr[i-win+1:i+1]
            x = idx; xm, ym = x.mean(), np.nanmean(y)
            num = np.nansum((x-xm)*(y-ym)); den = np.nansum((x-xm)**2)+1e-12
            out.append(num/den)
        return pd.Series(out, index=series.index)
    df['slope_high_10'] = slope(df['high'], 10)
    df['slope_low_10']  = slope(df['low'], 10)
    df['slope_close_10']= slope(df['close'],10)
    df['slope_gap']     = df['slope_high_10'] - df['slope_low_10']

    # Chip distribution features summarise where recent volume accumulated by price.
    window = 60
    bins = 20
    close_vals = df['close'].to_numpy(dtype=float)
    vol_vals = df['volume'].to_numpy(dtype=float)
    chip_main = np.full(len(df), np.nan, dtype=float)
    chip_concentration = np.full(len(df), np.nan, dtype=float)
    chip_support = np.full(len(df), np.nan, dtype=float)
    chip_resistance = np.full(len(df), np.nan, dtype=float)

    for idx in range(window - 1, len(df)):
        price_window = close_vals[idx - window + 1 : idx + 1]
        vol_window = vol_vals[idx - window + 1 : idx + 1]
        if not np.isfinite(price_window).all() or not np.isfinite(vol_window).all():
            continue
        total_vol = vol_window.sum()
        if total_vol <= 0:
            continue
        price_min = float(price_window.min())
        price_max = float(price_window.max())
        if price_max <= price_min + 1e-9:
            centers = np.array([price_min], dtype=float)
            hist = np.array([total_vol], dtype=float)
        else:
            edges = np.linspace(price_min, price_max, bins + 1)
            hist, edges = np.histogram(price_window, bins=edges, weights=vol_window)
            centers = 0.5 * (edges[:-1] + edges[1:])
        if hist.sum() <= 0:
            continue
        max_idx = int(hist.argmax())
        chip_main[idx] = float(centers[max_idx])
        chip_concentration[idx] = float(hist[max_idx] / hist.sum())

        current_price = close_vals[idx]
        below_mask = centers < current_price - 1e-9
        above_mask = centers > current_price + 1e-9
        at_mask = ~(below_mask | above_mask)
        below_vol = hist[below_mask].sum()
        above_vol = hist[above_mask].sum()
        at_vol = hist[at_mask].sum()
        denom = hist.sum() + 1e-9
        chip_support[idx] = float((below_vol + 0.5 * at_vol) / denom)
        chip_resistance[idx] = float((above_vol + 0.5 * at_vol) / denom)

    df['chip_main_price'] = chip_main
    df['chip_concentration'] = chip_concentration
    df['chip_support_ratio'] = chip_support
    df['chip_resistance_ratio'] = chip_resistance
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
