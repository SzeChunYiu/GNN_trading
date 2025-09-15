import argparse, os, time, requests
import pandas as pd

def fetch_daily(ticker, api_key, outputsize='full'):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'outputsize': outputsize,
        'datatype': 'json',
        'apikey': api_key
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if 'Time Series (Daily)' not in js:
        raise ValueError(f"Unexpected response for {ticker}: {js}")
    ts = js['Time Series (Daily)']
    rows = []
    for d, v in ts.items():
        rows.append({
            'date': d,
            'open': float(v['1. open']),
            'high': float(v['2. high']),
            'low':  float(v['3. low']),
            'close':float(v['4. close']),
            'volume': int(float(v['6. volume']))
        })
    df = pd.DataFrame(rows).sort_values('date')
    return df

def fetch_news(api_key, tickers=None, from_date=None, to_date=None, limit=1000):
    url = 'https://www.alphavantage.co/query'
    params = {'function':'NEWS_SENTIMENT','apikey':api_key,'limit':1000}
    if tickers:
        params['tickers'] = ','.join(tickers)
    if from_date: params['time_from'] = from_date.replace('-','') + 'T0000'
    if to_date:   params['time_to']   = to_date.replace('-','') + 'T2359'
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    feed = js.get('feed', [])
    rows = []
    for item in feed:
        dt = item.get('time_published','')[:8]
        date = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt)==8 else None
        headline = item.get('title','')
        rows.append({'date': date, 'headline': headline})
    return pd.DataFrame(rows).dropna()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', required=True, help='Comma-separated list, e.g. AAPL,MSFT')
    p.add_argument('--api_key', required=True)
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--outdir', default='data/alphavantage')
    p.add_argument('--with_news', action='store_true')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]

    for i, t in enumerate(tickers):
        df = fetch_daily(t, args.api_key, outputsize='full')
        if args.start:
            df = df[df['date'] >= args.start]
        if args.end:
            df = df[df['date'] <= args.end]
        out = os.path.join(args.outdir, f'{t}.csv')
        df.to_csv(out, index=False)
        print(f"Saved {out} ({len(df)} rows)")
        time.sleep(12)  # free-tier rate limit

    # panel
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

    if args.with_news:
        news = fetch_news(args.api_key, tickers=tickers, from_date=args.start, to_date=args.end)
        news.to_csv(os.path.join(args.outdir, 'news.csv'), index=False)
        print(f"Saved news CSV to {os.path.join(args.outdir, 'news.csv')}")

if __name__ == '__main__':
    main()
