import argparse, os, pandas as pd, glob

def main():
    p = argparse.ArgumentParser(description='Merge multiple single-asset CSVs into a clean panel and per-asset splits for training.')
    p.add_argument('--indir', default='data/yahoo', help='Folder with {TICKER}.csv having date, open, high, low, close, volume')
    p.add_argument('--outdir', default='data/clean')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = glob.glob(os.path.join(args.indir, '*.csv'))
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['date'])
        if not set(['date','open','high','low','close','volume']).issubset(df.columns):
            print(f"Skipping {f}: missing columns")
            continue
        ticker = os.path.splitext(os.path.basename(f))[0]
        df['ticker'] = ticker
        frames.append(df[['date','ticker','open','high','low','close','volume']])
    if not frames:
        print("No valid files found")
        return
    panel = pd.concat(frames).sort_values(['ticker','date'])
    panel.to_csv(os.path.join(args.outdir, 'panel.csv'), index=False)
    print(f"Saved panel.csv with {len(panel)} rows for {panel['ticker'].nunique()} tickers")

    # save each ticker split as well
    for t, sub in panel.groupby('ticker'):
        out = os.path.join(args.outdir, f'{t}.csv')
        sub[['date','open','high','low','close','volume']].to_csv(out, index=False)
        print(f"Saved {out} ({len(sub)} rows)")

if __name__ == '__main__':
    main()
