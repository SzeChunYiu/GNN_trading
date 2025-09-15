import argparse, os
import pandas as pd

def normalize_stooq_csv(path):
    df = pd.read_csv(path)
    # Stooq often uses: Date,Open,High,Low,Close,Volume
    cols = {c.lower(): c for c in df.columns}
    # unify
    df.columns = [c.lower() for c in df.columns]
    keep = ['date','open','high','low','close','volume']
    for k in keep:
        if k not in df.columns:
            raise ValueError(f"Missing '{k}' in {path}")
    out = df[keep].copy().sort_values('date')
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir', default='data/stooq_raw')
    p.add_argument('--outdir', default='data/stooq')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for fn in os.listdir(args.indir):
        if not fn.lower().endswith('.csv'):
            continue
        path = os.path.join(args.indir, fn)
        try:
            df = normalize_stooq_csv(path)
        except Exception as e:
            print(f"Skip {fn}: {e}")
            continue
        base = os.path.splitext(fn)[0]
        out = os.path.join(args.outdir, f"{base}.csv")
        df.to_csv(out, index=False)
        print(f"Saved {out} ({len(df)} rows)")

if __name__ == '__main__':
    main()
