import argparse, os, json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from tqdm import tqdm

from .utils import set_seed, pick_device
from .features import compute_features
from .patterns import add_pattern_scores
from .labels import make_breakout_labels
from .dataset import OHLCVGraphDataset
from .models import GAT_GRU_Encoder, SupervisedModel
from .news import aggregate_news_embeddings

def train_epoch(model, loader, device='cpu', lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    cls_loss = torch.nn.CrossEntropyLoss()
    reg_loss = torch.nn.SmoothL1Loss()
    vol_loss = torch.nn.SmoothL1Loss()
    total = 0.0
    for data in loader:
        data = data.to(device)
        logits, ret, vol = model(data)
        loss = cls_loss(logits, data.y) + 0.5*reg_loss(ret.squeeze(-1), data.y_reg.squeeze(-1)) + 0.5*vol_loss(vol.squeeze(-1), data.y_vol.squeeze(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device='cpu'):
    model.eval()
    probs, y_true, ret_pred, ret_true, vol_pred, vol_true = [], [], [], [], [], []
    for data in loader:
        data = data.to(device)
        logits, ret, vol = model(data)
        p = torch.softmax(logits, dim=-1)[0,1].item()
        probs.append(p); y_true.append(int(data.y.item()))
        ret_pred.append(ret.item()); ret_true.append(float(data.y_reg.item()))
        vol_pred.append(vol.item()); vol_true.append(float(data.y_vol.item()))
    auc = roc_auc_score(y_true, probs) if len(set(y_true))>1 else float('nan')
    ap  = average_precision_score(y_true, probs) if len(set(y_true))>1 else float('nan')
    mae = mean_absolute_error(ret_true, ret_pred)
    mae_vol = mean_absolute_error(vol_true, vol_pred)
    return {'AUC':auc, 'AP':ap, 'MAE_ret':mae, 'MAE_vol':mae_vol}, (np.array(probs), np.array(y_true))

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

    news_embed = None
    news_dim = None
    if args.news:
        news_embed = aggregate_news_embeddings(args.news, date_col=args.date_col, text_col=args.text_col, model_name=args.news_model, max_per_day=args.max_per_day)
        # infer dim
        news_dim = len(next(iter(news_embed.values()))) if len(news_embed)>0 else None

    # Walk-forward setup (optional)
    if args.walkforward:
        N = len(df_feat)
        splits = np.linspace(args.min_train_size, N-1, args.n_splits+1, dtype=int)
        all_signals = []
        for fold in range(args.n_splits):
            train_end = splits[fold]
            test_end  = splits[fold+1]
            df_train = df_feat.iloc[:train_end].copy()
            df_test  = df_feat.iloc[train_end:test_end].copy()
            if len(df_test) < 100: continue

            train_ds = OHLCVGraphDataset(df_train, window=args.window, pred_horizon=args.pred_horizon, use_labels=True, news_feat=news_embed)
            test_ds  = OHLCVGraphDataset(df_test,  window=args.window, pred_horizon=args.pred_horizon, use_labels=True, scaler=train_ds.scaler, news_feat=news_embed)

            in_dim = len([c for c in df_train.columns if c!='breakout_label'])
            encoder = GAT_GRU_Encoder(in_dim, args.hid, args.heads, args.gru_hid, dropout=args.dropout).to(device)
            # optional: load SSL weights if available
            if os.path.exists('outputs/encoder_ssl.pt'):
                try:
                    encoder.load_state_dict(torch.load('outputs/encoder_ssl.pt', map_location=device), strict=False)
                    print("Loaded SSL encoder weights.")
                except Exception as e:
                    print("Could not load SSL weights:", e)

            model = SupervisedModel(encoder, news_dim=news_dim, dropout=args.dropout).to(device)
            loader_tr = DataLoader(train_ds, batch_size=1, shuffle=True)
            loader_te = DataLoader(test_ds,  batch_size=1, shuffle=False)

            for ep in range(1, args.epochs+1):
                loss = train_epoch(model, loader_tr, device=device, lr=args.lr)
                if ep % max(1, args.epochs//3) == 0:
                    metrics, (probs, y_true) = evaluate(model, loader_te, device=device)
                    print(f"[Fold {fold+1}] ep {ep}: {metrics}")

            # save signals
            probs, y_true = evaluate(model, loader_te, device=device)[1]
            dates = df_test.index.values[train_ds.window: train_ds.window+len(probs)]
            fold_signals = pd.DataFrame({'date': dates, 'breakout_prob': probs, 'label': y_true})
            all_signals.append(fold_signals)

        if all_signals:
            sig = pd.concat(all_signals).sort_values('date')
            os.makedirs('outputs', exist_ok=True)
            sig.to_csv('outputs/signals.csv', index=False)
            print("Saved signals to outputs/signals.csv")
        return

    # Single train/eval on tail split
    split = int(len(df_feat)*0.8)
    df_train = df_feat.iloc[:split].copy()
    df_test  = df_feat.iloc[split:].copy()
    train_ds = OHLCVGraphDataset(df_train, window=args.window, pred_horizon=args.pred_horizon, use_labels=True, news_feat=news_embed)
    test_ds  = OHLCVGraphDataset(df_test,  window=args.window, pred_horizon=args.pred_horizon, use_labels=True, scaler=train_ds.scaler, news_feat=news_embed)

    in_dim = len([c for c in df_train.columns if c!='breakout_label'])
    encoder = GAT_GRU_Encoder(in_dim, args.hid, args.heads, args.gru_hid, dropout=args.dropout).to(device)
    if os.path.exists('outputs/encoder_ssl.pt'):
        try:
            encoder.load_state_dict(torch.load('outputs/encoder_ssl.pt', map_location=device), strict=False)
            print("Loaded SSL encoder weights.")
        except Exception as e:
            print("Could not load SSL weights:", e)

    model = SupervisedModel(encoder, news_dim=news_dim, dropout=args.dropout).to(device)
    loader_tr = DataLoader(train_ds, batch_size=1, shuffle=True)
    loader_te = DataLoader(test_ds,  batch_size=1, shuffle=False)

    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, loader_tr, device=device, lr=args.lr)
        if ep % max(1, args.epochs//3) == 0:
            metrics, _ = evaluate(model, loader_te, device=device)
            print(f"Epoch {ep}: {metrics}")

    metrics, (probs, y_true) = evaluate(model, loader_te, device=device)
    print("Final metrics:", metrics)
    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame({'prob': probs, 'y': y_true}).to_csv('outputs/signals.csv', index=False)
    torch.save(model.state_dict(), 'outputs/model_supervised.pt')
    print("Saved model to outputs/model_supervised.pt")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--news", default=None)
    p.add_argument("--date_col", default="date")
    p.add_argument("--text_col", default="headline")
    p.add_argument("--news_model", default="ProsusAI/finbert")
    p.add_argument("--max_per_day", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--pred_horizon", type=int, default=5)
    p.add_argument("--breakout_lookback", type=int, default=20)
    p.add_argument("--breakout_pct", type=float, default=0.01)
    p.add_argument("--vol_z_threshold", type=float, default=0.0)
    p.add_argument("--confirm_days", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hid", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--gru_hid", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--min_train_size", type=int, default=600)
    args = p.parse_args()
    main(args)
