import argparse, os, json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import set_seed, pick_device
from .features import compute_features
from .patterns import add_pattern_scores
from .labels import make_breakout_labels
from .dataset import OHLCVGraphDataset
from .models import GAT_GRU_Encoder

def nt_xent(z1, z2, temperature=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T)
    mask = torch.eye(2*B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9) / temperature
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    return torch.nn.CrossEntropyLoss()(sim, targets)

def main(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs('outputs', exist_ok=True)

    df = pd.read_csv(args.ohlcv, parse_dates=[args.date_col]).set_index(args.date_col).sort_index()
    df = df[['open','high','low','close','volume']].copy()

    df_feat = compute_features(df)
    df_feat = add_pattern_scores(df_feat)
    df_feat['breakout_label'] = make_breakout_labels(df_feat, args.breakout_lookback, args.vol_z_threshold, args.breakout_pct, args.confirm_days)
    df_feat = df_feat.replace([float('inf'), float('-inf')], float('nan')).dropna()

    ds = OHLCVGraphDataset(df_feat, window=args.window, pred_horizon=args.pred_horizon, use_labels=False)
    in_dim = len([c for c in df_feat.columns if c != 'breakout_label'])
    encoder = GAT_GRU_Encoder(in_dim=in_dim, hid=args.hid, heads=args.heads, gru_hid=args.gru_hid, proj_dim=64, dropout=args.dropout).to(device)

    ids = list(range(len(ds)))
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(1, args.epochs+1):
        total = 0.0
        for k in tqdm(ids, desc=f"SSL epoch {ep}"):
            d1, d2 = ds.get_ssl_pair(k, args.feat_mask_ratio, args.jitter_std)
            d1 = d1.to(device); d2 = d2.to(device)
            z1 = encoder(d1, return_proj=True)
            z2 = encoder(d2, return_proj=True)
            loss = nt_xent(z1.unsqueeze(0), z2.unsqueeze(0), args.temperature)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"[SSL] epoch {ep}: loss={total/len(ids):.4f}")

    torch.save(encoder.state_dict(), 'outputs/encoder_ssl.pt')
    print("Saved encoder to outputs/encoder_ssl.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--date_col", default="date")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--pred_horizon", type=int, default=5)
    p.add_argument("--breakout_lookback", type=int, default=20)
    p.add_argument("--breakout_pct", type=float, default=0.01)
    p.add_argument("--vol_z_threshold", type=float, default=0.0)
    p.add_argument("--confirm_days", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--feat_mask_ratio", type=float, default=0.15)
    p.add_argument("--jitter_std", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hid", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--gru_hid", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()
    main(args)
