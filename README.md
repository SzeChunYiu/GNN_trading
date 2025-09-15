# GNN_trading

A research-grade pipeline that combines **graph neural networks** (GNN) with
technical analysis features, heuristic chart-pattern scores, **self-supervised
pretraining**, **multi-task learning** (breakout classification, return regression,
volatility forecasting), **uncertainty estimation** (MC Dropout), **probability
calibration**, **multi-asset correlation graphs**, and **news fusion** (FinBERT sentiment).

> Educational/research project — **not financial advice**.

---

## Highlights

- **Temporal GNN encoder** (GATv2 → GRU).
- **Self-supervised contrastive pretraining** (NT-Xent with feature masking + jitter).
- **Heuristic pattern features** from OHLCV only (flag, cup&handle, wedge, trendline geometry).
- **Multi-task heads**: breakout probability (classification), next-N-day return (regression), volatility (ATR%) forecast.
- **MC Dropout** for **uncertainty**; optional **temperature scaling** calibration.
- **Multi-asset graphs** with **correlation edges** (optional).
- **News fusion**: FinBERT sentiment embeddings per day, late fusion with price encoder.
- **Walk-forward evaluation** (no leakage) + basic **backtester** example.
- **Config-driven** (YAML).

---

## Quick start

```bash
# (Recommended) use Python >= 3.10
pip install -r requirements.txt

# 1) Self-supervised pretraining on OHLCV
python -m gnn_trading.ssl_pretrain --ohlcv your_ohlcv.csv --date_col date

# 2) Supervised fine-tune + walk-forward evaluation
python -m gnn_trading.train --ohlcv your_ohlcv.csv --date_col date --walkforward

# 3) (Optional) Include news
python -m gnn_trading.train --ohlcv your_ohlcv.csv --news your_news.csv --date_col date --text_col headline

# 4) Backtest a simple breakout strategy
python -m gnn_trading.backtest --signals outputs/signals.csv --ohlcv your_ohlcv.csv
```

### Input formats

- **OHLCV CSV**: columns: `date, open, high, low, close, volume` (any extra columns are ignored).
- **News CSV** (optional): columns: `date, headline` or `date, text` (configurable via `--text_col`). Multiple rows per date allowed.

---

## Configuration

Edit `config_example.yaml` and pass via `--config config_example.yaml`. Default CLI flags override configs.

Key knobs:
- `window`: history window length per graph (40–120)
- `pred_horizon`: prediction horizon (3–10 days)
- `ssl`: epochs, augmentations, temperature
- `model`: hidden sizes, heads, dropout
- `labels`: breakout lookback, threshold, volume filter
- `news`: model name (FinBERT), max articles/day, fusion type

---

## Outputs

- `outputs/`:
  - `fold_*/metrics.json`: AUC/AP and regression MAE.
  - `signals.csv`: timestamps, breakout_prob, return_pred, vol_pred, confidence.
  - `calibration.json`: temperature and reliability stats.

---

## License

MIT (see LICENSE). This code is for research & education.
