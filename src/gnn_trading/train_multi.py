"""Train a shared encoder across multiple companies with walk-forward splits."""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from .config import ensure_directory, load_app_config
from .data_multi import MultiCompanyDataset
from .features import compute_features
from .labels import make_breakout_labels
from .patterns import add_pattern_scores
from .utils import pick_device, set_seed


def load_frames(root: Path, date_col: str, horizons: Sequence[int], breakout_cfg: Dict[str, float], tickers: Iterable[str] | None) -> Dict[str, pd.DataFrame]:
    keep = {t.upper() for t in tickers} if tickers else None
    frames: Dict[str, pd.DataFrame] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        symbol = sub.name.upper()
        if keep and symbol not in keep:
            continue
        csv = sub / "data.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=[date_col]).sort_values(date_col).set_index(date_col)
        df = df[["open", "high", "low", "close", "volume"]]
        feat = add_pattern_scores(compute_features(df))
        base = make_breakout_labels(feat, **breakout_cfg).astype(float)
        returns = feat["close"].pct_change()
        for h in horizons:
            feat[f"ret_h{h}"] = feat["close"].pct_change(h).shift(-h)
            feat[f"vol_h{h}"] = returns.rolling(h).std().shift(-h)
            feat[f"breakout_h{h}"] = base.shift(-1).rolling(h, min_periods=1).max().fillna(0.0)
        frames[symbol] = feat.replace([np.inf, -np.inf], np.nan)
    return frames


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    to_stack = lambda key, dtype=None: torch.stack([item[key] if dtype is None else item[key].to(dtype) for item in batch], 0)
    return {
        "x": to_stack("x"),
        "y_breakout": to_stack("y_breakout"),
        "y_ret": to_stack("y_ret"),
        "y_vol": to_stack("y_vol"),
        "symbol_id": to_stack("symbol_id"),
        "symbols": [item["symbol"] for item in batch],
        "dates": [item["date"] for item in batch],
    }


def all_dates(frames: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    uniq = set()
    for df in frames.values():
        uniq.update(df.index)
    return sorted(pd.Timestamp(d) for d in uniq)


class MultiHorizonModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        horizons: Sequence[int],
        dropout: float,
        *,
        num_symbols: int,
        symbol_embed_dim: int = 0,
        symbol_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.horizons = sorted(int(h) for h in horizons)
        from .models import MultiTaskHead  # lazy import to avoid circular

        self.heads = nn.ModuleDict({str(h): MultiTaskHead(self.encoder.output_dim, dropout=dropout) for h in self.horizons})
        self.symbol_embed = None
        self.symbol_proj = None
        self.symbol_dropout = nn.Dropout(symbol_dropout)
        if symbol_embed_dim > 0:
            if num_symbols <= 0:
                raise ValueError("num_symbols must be positive when using symbol embeddings")
            self.symbol_embed = nn.Embedding(num_symbols, symbol_embed_dim)
            self.symbol_proj = nn.Linear(self.encoder.output_dim + symbol_embed_dim, self.encoder.output_dim)

    def forward(
        self,
        x_batch: torch.Tensor,
        edge_index: torch.Tensor,
        symbol_ids: torch.Tensor | None = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        data_list = [Data(x=seq, edge_index=edge_index) for seq in x_batch]
        batch_graph = Batch.from_data_list(data_list)
        batch_graph = batch_graph.to(x_batch.device)
        hidden = self.encoder(batch_graph, return_proj=False)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        if self.symbol_embed is not None and symbol_ids is not None:
            sym_vec = self.symbol_embed(symbol_ids)
            hidden = torch.cat([hidden, sym_vec], dim=-1)
            hidden = self.symbol_proj(self.symbol_dropout(hidden))
        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for h in self.horizons:
            logits, ret, vol = self.heads[str(h)](hidden)
            out[h] = {"logits": logits, "ret": ret.squeeze(-1), "vol": vol.squeeze(-1)}
        return out


def train_epoch(
    model: MultiHorizonModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    opt: Adam,
    device: torch.device,
    horizons: Sequence[int],
) -> float:
    ce = nn.CrossEntropyLoss()
    reg = nn.SmoothL1Loss()
    total = 0.0
    model.train()
    for batch in loader:
        opt.zero_grad()
        preds = model(batch["x"].to(device), edge_index, batch["symbol_id"].to(device))
        y_break = batch["y_breakout"].to(device)
        y_ret = batch["y_ret"].to(device)
        y_vol = batch["y_vol"].to(device)
        loss = torch.zeros(1, device=device)
        for idx, horizon in enumerate(horizons):
            out = preds[horizon]
            loss = loss + ce(out["logits"], y_break[:, idx].long())
            loss = loss + 0.5 * reg(out["ret"], y_ret[:, idx])
            loss = loss + 0.5 * reg(out["vol"], y_vol[:, idx])
        loss = loss / len(horizons)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))


def evaluate(
    model: MultiHorizonModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    horizons: Sequence[int],
) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    model.eval()
    records = {h: {"prob": [], "label": [], "ret_pred": [], "ret_true": [], "vol_pred": [], "vol_true": []} for h in horizons}
    signals: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            preds = model(batch["x"].to(device), edge_index, batch["symbol_id"].to(device))
            for idx, horizon in enumerate(horizons):
                out = preds[horizon]
                probs = torch.softmax(out["logits"], dim=-1)[:, 1]
                records[horizon]["prob"].extend(probs.cpu().tolist())
                records[horizon]["label"].extend(batch["y_breakout"][:, idx].tolist())
                records[horizon]["ret_pred"].extend(out["ret"].cpu().tolist())
                records[horizon]["ret_true"].extend(batch["y_ret"][:, idx].tolist())
                records[horizon]["vol_pred"].extend(out["vol"].cpu().tolist())
                records[horizon]["vol_true"].extend(batch["y_vol"][:, idx].tolist())
            main = horizons[0]
            main_prob = torch.softmax(preds[main]["logits"], dim=-1)[:, 1].cpu().tolist()
            main_ret = preds[main]["ret"].cpu().tolist()
            main_vol = preds[main]["vol"].cpu().tolist()
            for i, sym in enumerate(batch["symbols"]):
                signals.append(
                    {
                        "date": batch["dates"][i],
                        "symbol": sym,
                        "breakout_prob": float(main_prob[i]),
                        "ret_pred": float(main_ret[i]),
                        "vol_pred": float(main_vol[i]),
                        "confidence": float(main_prob[i]),
                    }
                )
    metrics: Dict[str, Dict[str, float]] = {}
    for horizon in horizons:
        data = records[horizon]
        labels = np.array(data["label"])
        probs = np.array(data["prob"])
        ret_pred = np.array(data["ret_pred"])
        ret_true = np.array(data["ret_true"])
        vol_pred = np.array(data["vol_pred"])
        vol_true = np.array(data["vol_true"])
        auc = float("nan") if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, probs))
        metrics[str(horizon)] = {
            "auc": auc,
            "mae_return": float(mean_absolute_error(ret_true, ret_pred)) if len(ret_true) else float("nan"),
            "mae_vol": float(mean_absolute_error(vol_true, vol_pred)) if len(vol_true) else float("nan"),
        }
    return metrics, signals


def splits(dates: Sequence[pd.Timestamp], n_splits: int, min_train: int, purge: int) -> List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if len(dates) <= min_train:
        return []
    grid = np.linspace(min_train, len(dates) - 1, n_splits + 1, dtype=int)
    out: List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for idx in range(n_splits):
        train_end_idx, test_end_idx = grid[idx], grid[idx + 1]
        test_start_idx = min(train_end_idx + purge, len(dates) - 1)
        if test_start_idx >= test_end_idx:
            continue
        out.append((dates[0], dates[train_end_idx], dates[test_start_idx], dates[test_end_idx]))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ohlcv_dir", required=True)
    p.add_argument("--out_dir", default="outputs/train_multi")
    p.add_argument("--date_col", default="date")
    p.add_argument("--tickers", default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--horizons", type=int, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hid", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--gru_hid", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--symbol_embed_dim", type=int, default=16)
    p.add_argument("--breakout_lookback", type=int, default=20)
    p.add_argument("--breakout_pct", type=float, default=0.01)
    p.add_argument("--vol_z_threshold", type=float, default=0.0)
    p.add_argument("--confirm_days", type=int, default=0)
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--min_train_size", type=int, default=600)
    p.add_argument("--purge", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    cfg = load_app_config()
    window = args.window or int(cfg["features"]["lookback"])
    horizons = sorted(int(h) for h in (args.horizons or cfg["labels"]["horizons"]))
    breakout_cfg = {
        "lookback": args.breakout_lookback,
        "vol_z_threshold": args.vol_z_threshold,
        "breakout_pct": args.breakout_pct,
        "confirm_days": args.confirm_days,
    }
    root = Path(args.ohlcv_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing OHLCV directory: {root}")
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    frames = load_frames(root, args.date_col, horizons, breakout_cfg, tickers)
    if not frames:
        raise ValueError("No symbols discovered under the provided directory")
    dates = all_dates(frames)
    out_dir = ensure_directory(args.out_dir)

    from .models import GAT_GRU_Encoder  # lazy import for clarity

    def build_model(input_dim: int, num_symbols: int) -> MultiHorizonModel:
        encoder = GAT_GRU_Encoder(
            in_dim=input_dim,
            hid=args.hid,
            heads=args.heads,
            gru_hid=args.gru_hid,
            dropout=args.dropout,
        ).to(device)
        return MultiHorizonModel(
            encoder,
            horizons,
            dropout=args.dropout,
            num_symbols=num_symbols,
            symbol_embed_dim=max(0, args.symbol_embed_dim),
        ).to(device)

    def run(train_ds: MultiCompanyDataset, eval_ds: MultiCompanyDataset, tag: str) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
        if not len(train_ds) or not len(eval_ds):
            raise ValueError(f"Insufficient samples for {tag}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        model = build_model(len(train_ds.feature_cols), len(train_ds.symbols))
        opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        edge_index = train_ds.edge_index.to(device)
        for _ in range(args.epochs):
            train_epoch(model, train_loader, edge_index, opt, device, horizons)
        metrics, signals = evaluate(model, eval_loader, edge_index, device, horizons)
        tag_dir = ensure_directory(out_dir / tag)
        torch.save(model.state_dict(), tag_dir / "model.pt")
        with (tag_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics, signals

    if args.walkforward:
        fold_metrics: List[Dict[str, Dict[str, float]]] = []
        all_signals: List[Dict[str, object]] = []
        for fold, (train_start, train_end, test_start, test_end) in enumerate(splits(dates, args.n_splits, args.min_train_size, args.purge), start=1):
            train_ds = MultiCompanyDataset(frames, window, horizons, end_date=train_end, fit_end_date=train_end)
            eval_ds = MultiCompanyDataset(frames, window, horizons, start_date=test_start, end_date=test_end, scaler=train_ds.scaler)
            metrics, signals = run(train_ds, eval_ds, f"fold_{fold}")
            fold_metrics.append(metrics)
            for row in signals:
                row["fold"] = fold
                all_signals.append(row)
        summary: Dict[str, Dict[str, float]] = {}
        for h in horizons:
            aucs = [m[str(h)]["auc"] for m in fold_metrics if not math.isnan(m[str(h)]["auc"])]
            mae_r = [m[str(h)]["mae_return"] for m in fold_metrics if not math.isnan(m[str(h)]["mae_return"])]
            mae_v = [m[str(h)]["mae_vol"] for m in fold_metrics if not math.isnan(m[str(h)]["mae_vol"])]
            summary[str(h)] = {
                "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                "mae_return_mean": float(np.mean(mae_r)) if mae_r else float("nan"),
                "mae_vol_mean": float(np.mean(mae_v)) if mae_v else float("nan"),
            }
        with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if all_signals:
            df = pd.DataFrame(all_signals)
            df["date"] = df["date"].astype(str)
            df.sort_values(["date", "symbol", "fold"]).to_csv(out_dir / "signals.csv", index=False)
    else:
        split_idx = int(len(dates) * 0.8)
        if split_idx <= window + max(horizons):
            raise ValueError("Not enough history for a hold-out split; enable --walkforward instead.")
        train_end = dates[split_idx]
        train_ds = MultiCompanyDataset(frames, window, horizons, end_date=train_end, fit_end_date=train_end)
        eval_ds = MultiCompanyDataset(frames, window, horizons, start_date=train_end, end_date=dates[-1], scaler=train_ds.scaler)
        metrics, signals = run(train_ds, eval_ds, "single_run")
        with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        if signals:
            df = pd.DataFrame(signals)
            df["date"] = df["date"].astype(str)
            df.sort_values(["date", "symbol"]).to_csv(out_dir / "signals.csv", index=False)


if __name__ == "__main__":
    main()
