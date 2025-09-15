import argparse, pandas as pd, numpy as np

def simple_breakout_strategy(signals_csv, ohlcv_csv, prob_threshold=0.6, hold_days=5):
    sig = pd.read_csv(signals_csv, parse_dates=['date'])
    px = pd.read_csv(ohlcv_csv, parse_dates=['date']).set_index('date').sort_index()
    px = px[['close']].copy()
    sig = sig.set_index('date').sort_index()
    # align
    df = sig.join(px, how='inner')
    df['signal'] = (df['breakout_prob'] >= prob_threshold).astype(int) if 'breakout_prob' in df.columns else (df['prob']>=prob_threshold).astype(int)
    # simulate constant 1-unit positions
    df['ret_fwd'] = px['close'].pct_change(hold_days).shift(-hold_days).reindex(df.index)
    df['pnl'] = df['signal'] * df['ret_fwd']
    equity = (1 + df['pnl'].fillna(0)).cumprod()
    stats = {
        'final_equity': float(equity.iloc[-1]),
        'avg_trade_ret': float(df.loc[df['signal']==1, 'ret_fwd'].mean()),
        'hit_rate': float((df.loc[df['signal']==1, 'ret_fwd']>0).mean())
    }
    return equity, stats

def main(args):
    eq, stats = simple_breakout_strategy(args.signals, args.ohlcv, args.threshold, args.hold)
    print("Backtest stats:", stats)
    eq.to_frame('equity').to_csv('outputs/equity_curve.csv')
    print("Saved outputs/equity_curve.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--signals", required=True)
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--hold", type=int, default=5)
    args = p.parse_args()
    main(args)
