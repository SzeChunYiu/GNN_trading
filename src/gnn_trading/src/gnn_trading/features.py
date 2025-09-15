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
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
