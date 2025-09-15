import numpy as np
import pandas as pd

def _safe_std(x, eps=1e-12):
    s = np.nanstd(x); 
    return s if s>eps else eps

def flag_score_window(close, high, low, atr_pct, lookback_run=10, lookback_cons=10):
    if len(close) < lookback_run + lookback_cons: return 0.0
    run = close[-(lookback_run+lookback_cons):-lookback_cons]
    cons_h = high[-lookback_cons:]; cons_l = low[-lookback_cons:]
    run_ret = (run[-1] - run[0]) / (run[0] + 1e-12)
    x = np.arange(len(run)); xm, ym = x.mean(), np.mean(run)
    slope_run = ((x-xm)*(run-ym)).sum()/(((x-xm)**2).sum()+1e-12)
    run_score = np.tanh(3.0*max(0.0, 0.5*run_ret + 0.5*slope_run/(abs(ym)+1e-12)))
    cons_range = (cons_h.max() - cons_l.min()) / (run[-1] + 1e-12)
    cons_range_score = np.tanh(2.0*max(0.0, 0.15 - cons_range))
    x2 = np.arange(len(cons_h))
    def slope_arr(y):
        xm, ym = x2.mean(), np.mean(y)
        return ((x2-xm)*(y-ym)).sum()/(((x2-xm)**2).sum()+1e-12)
    sh, sl = slope_arr(cons_h), slope_arr(cons_l)
    parallel_score = np.tanh(3.0*max(0.0, 0.05 - abs(sh - sl)))
    atr_run = np.nanmean(atr_pct[-(lookback_run+lookback_cons):-lookback_cons])
    atr_cons= np.nanmean(atr_pct[-lookback_cons:])
    atr_score = np.tanh(2.0*max(0.0, atr_run - atr_cons))
    return float(np.clip(0.1 + 0.4*run_score + 0.3*cons_range_score + 0.15*parallel_score + 0.05*atr_score, 0, 1))

def cup_handle_score_window(close, lookback_cup=40, handle=10):
    N = lookback_cup + handle
    if len(close) < N: return 0.0
    seg = close[-N:]
    segn = (seg - seg.min()) / (seg.max() - seg.min() + 1e-12)
    x = np.arange(lookback_cup); y = segn[:lookback_cup]
    X = np.vstack([x**2, x, np.ones_like(x)]).T
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    curv_score = np.tanh(3.0*max(0.0, a))
    left_mean, right_mean = y[:lookback_cup//2].mean(), y[lookback_cup//2:].mean()
    symm_score = np.tanh(3.0*max(0.0, 0.1 - abs(left_mean - right_mean)))
    rim_level = max(y[0], y[lookback_cup-1]); last_cup = y[-1]
    rim_score = np.tanh(3.0*max(0.0, 0.15 - abs(rim_level - last_cup)))
    h = segn[-handle:]
    handle_depth = (h.max() - h.min()); cup_depth = (y.max() - y.min()) + 1e-12
    handle_score = np.tanh(3.0*max(0.0, 0.3*cup_depth - handle_depth))
    return float(np.clip(0.05 + 0.35*curv_score + 0.2*symm_score + 0.25*rim_score + 0.15*handle_score, 0, 1))

def wedge_score_window(high, low, lookback=30):
    if len(high) < lookback: return 0.0
    h = high[-lookback:]; l = low[-lookback:]
    x = np.arange(lookback); xm = x.mean()
    def slope_fit(y):
        ym = y.mean()
        return ((x-xm)*(y-ym)).sum()/(((x-xm)**2).sum()+1e-12)
    sh, sl = slope_fit(h), slope_fit(l)
    conv = max(0.0, (0.0 - sh)) + max(0.0, sl)
    conv_score = np.tanh(2.5 * conv / (abs(h.mean()) + 1e-12))
    widths = (h - l) / (0.5*(h + l) + 1e-12)
    shrink = (widths[:lookback//2].mean() - widths[lookback//2:].mean())
    shrink_score = np.tanh(3.0 * max(0.0, shrink))
    return float(np.clip(0.1 + 0.6*conv_score + 0.3*shrink_score, 0, 1))

def add_pattern_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); N = len(df)
    flag_s, cup_s, wedge_s = [], [], []
    for i in range(N):
        sl = slice(0, i+1)
        c, h, l = df['close'].values[sl], df['high'].values[sl], df['low'].values[sl]
        atrp = df['atr_pct'].values[sl]
        flag_s.append(flag_score_window(c, h, l, atrp, 10, 10))
        cup_s.append(cup_handle_score_window(c, 40, 10))
        wedge_s.append(wedge_score_window(h, l, 30))
    df['score_flag']  = pd.Series(flag_s, index=df.index)
    df['score_cup']   = pd.Series(cup_s,  index=df.index)
    df['score_wedge'] = pd.Series(wedge_s,index=df.index)
    return df
