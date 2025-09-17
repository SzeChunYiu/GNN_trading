"""Interactive fund manager interface powered by the GNN backend."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .dataset import OHLCVGraphDataset
from .features import compute_features
from .labels import make_breakout_labels
from .models import GAT_GRU_Encoder, SupervisedModel
from .patterns import add_pattern_scores
from .utils import pick_device


@dataclass
class Holding:
    ticker: str
    shares: float
    cost_basis: float


class PortfolioDB:
    """Simple SQLite-backed storage for holdings."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._create_table()

    def _create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                ticker TEXT PRIMARY KEY,
                shares REAL NOT NULL,
                cost_basis REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, holding: Holding) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO holdings(ticker, shares, cost_basis)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                shares=excluded.shares,
                cost_basis=excluded.cost_basis
            """,
            (holding.ticker.upper(), holding.shares, holding.cost_basis),
        )
        self.conn.commit()

    def remove(self, ticker: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM holdings WHERE ticker=?", (ticker.upper(),))
        self.conn.commit()

    def list_holdings(self) -> List[Holding]:
        cur = self.conn.cursor()
        cur.execute("SELECT ticker, shares, cost_basis FROM holdings ORDER BY ticker")
        rows = cur.fetchall()
        return [Holding(ticker=row[0], shares=row[1], cost_basis=row[2]) for row in rows]


class GNNBackend:
    """Utility that wraps the GNN model for single-window inference."""

    def __init__(
        self,
        model_cfg: Optional[Dict] = None,
        label_cfg: Optional[Dict] = None,
        window: int = 60,
        pred_horizon: int = 5,
        device: str = "auto",
        checkpoint: Optional[str] = None,
        news_json: Optional[str] = None,
        date_col: str = "date",
    ):
        self.model_cfg = {
            "hid": 64,
            "heads": 2,
            "gru_hid": 64,
            "proj_dim": 64,
            "dropout": 0.2,
            "num_gru_layers": 2,
            "bidirectional": True,
            "num_gat_layers": 3,
            "use_temporal_attn": True,
            "attn_heads": 4,
            "attn_dropout": 0.1,
        }
        if model_cfg:
            self.model_cfg.update(model_cfg)

        self.label_cfg = {
            "breakout_lookback": 20,
            "breakout_pct": 0.01,
            "vol_z_threshold": 0.0,
            "confirm_days": 0,
        }
        if label_cfg:
            self.label_cfg.update(label_cfg)

        self.window = window
        self.pred_horizon = pred_horizon
        self.date_col = date_col
        self.device = pick_device(device)
        self.checkpoint = checkpoint
        self._pending_checkpoint = checkpoint
        self._news_vectors = self._load_news_vectors(news_json) if news_json else None
        self.scaler = None
        self.encoder: Optional[GAT_GRU_Encoder] = None
        self.model: Optional[SupervisedModel] = None
        self.input_dim: Optional[int] = None

    @staticmethod
    def _load_news_vectors(path: str) -> Dict[str, Sequence[float]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items()}

    def _init_model(self, in_dim: int, news_dim: Optional[int] = None) -> None:
        cfg = self.model_cfg
        encoder = GAT_GRU_Encoder(
            in_dim=in_dim,
            hid=cfg.get("hid", 64),
            heads=cfg.get("heads", 2),
            gru_hid=cfg.get("gru_hid", 64),
            proj_dim=cfg.get("proj_dim", cfg.get("hid", 64)),
            dropout=cfg.get("dropout", 0.2),
            num_gru_layers=cfg.get("num_gru_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            num_gat_layers=cfg.get("num_gat_layers", 3),
            use_temporal_attn=cfg.get("use_temporal_attn", True),
            attn_heads=cfg.get("attn_heads", 4),
            attn_dropout=cfg.get("attn_dropout", 0.1),
        ).to(self.device)
        model = SupervisedModel(encoder, news_dim=news_dim, dropout=cfg.get("dropout", 0.2)).to(self.device)
        encoder.eval()
        model.eval()
        self.encoder = encoder
        self.model = model
        self.input_dim = in_dim
        if self._pending_checkpoint:
            self.load_checkpoint(self._pending_checkpoint)
            self._pending_checkpoint = None

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        if self.model is None:
            self._pending_checkpoint = path
            return
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing parameters during load: {missing}")
        if unexpected:
            print(f"Warning: unexpected parameters during load: {unexpected}")

    def _prepare_dataframe(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=[self.date_col])
        if self.date_col not in df.columns:
            raise ValueError(f"Column '{self.date_col}' not found in {csv_path}")
        df = df.sort_values(self.date_col).set_index(self.date_col)
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        base = df[list(required)].copy()
        feat = compute_features(base)
        feat = add_pattern_scores(feat)
        feat["breakout_label"] = make_breakout_labels(
            feat,
            self.label_cfg["breakout_lookback"],
            self.label_cfg["vol_z_threshold"],
            self.label_cfg["breakout_pct"],
            self.label_cfg["confirm_days"],
        )
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        return feat

    def _prepare_news(self, df: pd.DataFrame) -> Optional[Dict[str, Sequence[float]]]:
        if not self._news_vectors:
            return None
        if isinstance(df.index, pd.DatetimeIndex):
            return {str(idx.date()): self._news_vectors.get(str(idx.date())) for idx in df.index}
        return None

    def _build_dataset(self, df_feat: pd.DataFrame) -> Optional[OHLCVGraphDataset]:
        if len(df_feat) < self.window + self.pred_horizon:
            return None
        news_vectors = self._prepare_news(df_feat)
        dataset = OHLCVGraphDataset(
            df_feat,
            window=self.window,
            pred_horizon=self.pred_horizon,
            use_labels=False,
            scaler=self.scaler,
            news_feat=news_vectors,
        )
        if self.scaler is None:
            self.scaler = dataset.scaler
        return dataset

    @torch.no_grad()
    def score_ticker(self, csv_path: str, ticker: str) -> Optional[Dict[str, float]]:
        df_feat = self._prepare_dataframe(csv_path)
        dataset = self._build_dataset(df_feat)
        if dataset is None or len(dataset) == 0:
            print(f"Not enough data to score {ticker}. Need at least {self.window} rows.")
            return None
        data = dataset[len(dataset) - 1]
        in_dim = data.x.shape[-1]
        news_dim = data.news.shape[-1] if hasattr(data, "news") else None
        if self.model is None:
            self._init_model(in_dim, news_dim=news_dim)
        elif self.input_dim is not None and in_dim != self.input_dim:
            raise ValueError("Input dimension changed; please restart the application to rebuild the model.")
        device_data = data.to(self.device)
        logits, ret, vol = self.model(device_data)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        exp_ret = ret.item()
        exp_vol = vol.item()
        last_close = float(df_feat["close"].iloc[-1]) if "close" in df_feat.columns else float("nan")
        return {
            "probability": prob,
            "expected_return": exp_ret,
            "volatility": exp_vol,
            "last_close": last_close,
        }


class FundManagerApp:
    def __init__(
        self,
        db: PortfolioDB,
        backend: GNNBackend,
        data_template: str,
        date_col: str = "date",
    ):
        self.db = db
        self.backend = backend
        self.data_template = data_template
        self.date_col = date_col

    def run(self) -> None:
        while True:
            print("\n========== Fund Manager ==========")
            print("1) Add / update holding")
            print("2) Remove holding")
            print("3) View holdings")
            print("4) Generate recommendations")
            print("5) Portfolio summary")
            print("Q) Quit")
            choice = input("Select an option: ").strip().lower()
            if choice == "1":
                self._add_or_update()
            elif choice == "2":
                self._remove()
            elif choice == "3":
                self._view()
            elif choice == "4":
                self._recommend()
            elif choice == "5":
                self._portfolio_summary()
            elif choice in {"q", "quit", "exit"}:
                print("Goodbye!")
                break
            else:
                print("Invalid selection. Please try again.")

    def _prompt_float(self, message: str) -> float:
        while True:
            raw = input(message).strip()
            try:
                return float(raw)
            except ValueError:
                print("Please enter a numeric value.")

    def _add_or_update(self) -> None:
        ticker = input("Ticker symbol: ").strip().upper()
        if not ticker:
            print("Ticker cannot be empty.")
            return
        shares = self._prompt_float("Number of shares: ")
        cost = self._prompt_float("Cost basis per share: ")
        holding = Holding(ticker=ticker, shares=shares, cost_basis=cost)
        self.db.upsert(holding)
        print(f"Saved holding for {ticker}.")

    def _remove(self) -> None:
        ticker = input("Ticker symbol to remove: ").strip().upper()
        if not ticker:
            return
        self.db.remove(ticker)
        print(f"Removed {ticker} from holdings.")

    def _view(self) -> None:
        holdings = self.db.list_holdings()
        if not holdings:
            print("No holdings recorded yet.")
            return
        print("\nTicker   Shares       Cost Basis")
        print("---------------------------------")
        for h in holdings:
            print(f"{h.ticker:<7} {h.shares:>10.2f}   {h.cost_basis:>10.2f}")

    def _recommend(self) -> None:
        holdings = {h.ticker: h for h in self.db.list_holdings()}
        tickers_input = input(
            "Enter comma-separated tickers to score (blank for holdings): "
        ).strip()
        if tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        else:
            tickers = list(holdings.keys())
        if not tickers:
            print("No tickers provided.")
            return
        results = []
        for ticker in tickers:
            csv_path = self.data_template.format(ticker=ticker)
            if not os.path.exists(csv_path):
                print(f"Data file not found for {ticker}: {csv_path}")
                continue
            try:
                score = self.backend.score_ticker(csv_path, ticker)
            except Exception as exc:  # pragma: no cover - interactive feedback
                print(f"Failed to score {ticker}: {exc}")
                continue
            if score is None:
                continue
            score["ticker"] = ticker
            holding = holdings.get(ticker)
            score["shares"] = holding.shares if holding else 0.0
            score["cost_basis"] = holding.cost_basis if holding else float("nan")
            score["action"] = self._action_for(score, holding)
            score["expected_price"] = (
                score["last_close"] * (1 + score["expected_return"])
                if np.isfinite(score["last_close"])
                else float("nan")
            )
            score["risk_adjusted"] = score["expected_return"] / (abs(score["volatility"]) + 1e-6)
            results.append(score)
        if not results:
            print("No results to display.")
            return
        results.sort(key=lambda r: r["risk_adjusted"], reverse=True)
        print(
            "\nTicker  Action  Prob    ExpRet%  Last    ExpPx   Vol%   RiskAdj  Shares"
        )
        print("-" * 74)
        for res in results:
            prob = res["probability"] * 100
            exp_ret = res["expected_return"] * 100
            vol = res["volatility"] * 100
            risk_adj = res["risk_adjusted"]
            last = res["last_close"]
            exp_price = res["expected_price"]
            print(
                f"{res['ticker']:<6} {res['action']:<6} {prob:>6.2f} {exp_ret:>8.2f}"
                f" {last:>7.2f} {exp_price:>7.2f} {vol:>7.2f} {risk_adj:>8.2f} {res['shares']:>6.2f}"
            )

    def _action_for(self, score: Dict[str, float], holding: Optional[Holding]) -> str:
        prob = score["probability"]
        exp_ret = score["expected_return"]
        if holding is None or holding.shares <= 0:
            if prob > 0.6 and exp_ret > 0.02:
                return "BUY"
            if prob < 0.4 or exp_ret < 0.0:
                return "PASS"
            return "WATCH"
        unrealized = float("nan")
        if np.isfinite(score["last_close"]):
            unrealized = (score["last_close"] - holding.cost_basis) / (holding.cost_basis + 1e-9)
        if prob < 0.35 or exp_ret < -0.02:
            return "SELL"
        if prob > 0.65 and exp_ret > 0.05:
            return "ADD"
        if np.isfinite(unrealized) and unrealized > 0.15 and exp_ret < 0.01:
            return "TRIM"
        return "HOLD"

    def _portfolio_summary(self) -> None:
        holdings = self.db.list_holdings()
        if not holdings:
            print("No holdings recorded yet.")
            return
        total_value = 0.0
        expected_value = 0.0
        actions = []
        for holding in holdings:
            csv_path = self.data_template.format(ticker=holding.ticker)
            if not os.path.exists(csv_path):
                continue
            score = self.backend.score_ticker(csv_path, holding.ticker)
            if not score:
                continue
            price = score["last_close"]
            total_value += holding.shares * price
            expected_price = price * (1 + score["expected_return"])
            expected_value += holding.shares * expected_price
            actions.append((holding.ticker, self._action_for(score, holding)))
        if total_value == 0:
            print("Portfolio value is zero or data files missing.")
            return
        exp_change = ((expected_value - total_value) / total_value) * 100
        print(f"\nCurrent portfolio value: ${total_value:,.2f}")
        print(f"Model-implied change over horizon: {exp_change:,.2f}%")
        if actions:
            print("Suggested actions:")
            for ticker, action in actions:
                print(f"  - {ticker}: {action}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive fund manager powered by the GNN model")
    parser.add_argument("--db", default="fund_manager.db", help="SQLite file to store holdings")
    parser.add_argument("--data-template", default="data/{ticker}.csv", help="Path template for OHLCV CSV files")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained supervised model state dict")
    parser.add_argument("--device", default="auto", help="Device to run inference on")
    parser.add_argument("--window", type=int, default=60, help="Sequence length window for the encoder")
    parser.add_argument("--pred-horizon", type=int, default=5, help="Prediction horizon used during training")
    parser.add_argument("--model-json", default=None, help="Optional JSON file with model hyperparameters")
    parser.add_argument("--labels-json", default=None, help="Optional JSON file with label generation params")
    parser.add_argument("--news-json", default=None, help="Optional JSON mapping YYYY-MM-DD to news embedding vectors")
    parser.add_argument("--date-col", default="date", help="Date column in the OHLCV CSV files")
    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    model_cfg = load_json(args.model_json)
    label_cfg = load_json(args.labels_json)
    backend = GNNBackend(
        model_cfg=model_cfg,
        label_cfg=label_cfg,
        window=args.window,
        pred_horizon=args.pred_horizon,
        device=args.device,
        checkpoint=args.checkpoint,
        news_json=args.news_json,
        date_col=args.date_col,
    )
    db = PortfolioDB(args.db)
    app = FundManagerApp(db=db, backend=backend, data_template=args.data_template, date_col=args.date_col)
    app.run()


if __name__ == "__main__":  # pragma: no cover - manual invocation entry point
    main()
