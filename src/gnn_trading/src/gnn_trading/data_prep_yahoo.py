import argparse, os
import pandas as pd
import yfinance as yf

def download_ticker(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return None
    df = df.rename(columns={
        'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'
    })
    df = df[['open','high','low','close','volume']].copy()
    df.reset_index(inplace=True)
    df.rename(columns={'Date':'date'}, inplace=True)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', required=True, help='Comma-separated tickers, e.g. AAPL,MSFT,SPY')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--outdir', default='data/yahoo')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]

    for t in tickers:
        df = download_ticker(t, args.start, args.end)
        if df is None or df.empty:
            print(f"[{t}] no data returned")
            continue
        out = os.path.join(args.outdir, f'{t}.csv')
        df.to_csv(out, index=False)
        print(f"Saved {out} ({len(df)} rows)")

    # also produce a long panel (ticker column)
    frames = []
    for t in tickers:
        f = os.path.join(args.outdir, f'{t}.csv')
        if os.path.exists(f):
            dd = pd.read_csv(f, parse_dates=['date'])
            dd['ticker'] = t
            frames.append(dd)
    if frames:
        panel = pd.concat(frames).sort_values(['ticker','date'])
        panel.to_csv(os.path.join(args.outdir, 'panel.csv'), index=False)
        print(f"Saved panel to {os.path.join(args.outdir, 'panel.csv')}")

if __name__ == '__main__':
    main()
