# Data & Model Paths

## Local Data Hub
- `LocalDataHub` initialises `<root>/{market,fundamentals,news,reports,metadata}` directories and maintains a `metadata/catalog.json` ledger.【F:src/gnn_trading/data_hub.py†L69-L198】
- Market data accessor: `<root>/market/<TICKER>.csv`; similar helpers store fundamentals/news/reports as CSV or JSON payloads.【F:src/gnn_trading/data_hub.py†L138-L198】

## Training & Pretraining Outputs
- SSL pretraining saves encoder weights to `outputs/encoder_ssl.pt`.【F:src/gnn_trading/ssl_pretrain.py†L55-L56】
- Supervised training writes inference signals to `outputs/signals.csv` and the model checkpoint to `outputs/model_supervised.pt`.【F:src/gnn_trading/train.py†L111-L148】
- Walk-forward mode also deposits combined signals under `outputs/signals.csv`.【F:src/gnn_trading/train.py†L102-L114】

## Backtesting
- The breakout backtester exports an equity curve to `outputs/equity_curve.csv`.【F:src/gnn_trading/backtest.py†L23-L26】

## Fund Manager & Recommendations
- Fund manager CLI defaults to OHLCV files resolved via `data/{ticker}.csv`; batch recommendations optionally persist a JSON summary when `--output` is provided.【F:src/gnn_trading/fund_manager.py†L621-L712】【F:src/gnn_trading/fund_manager_recommend.py†L117-L134】
- The data hub can act as a fallback source for recommendation runs (`LocalDataHub.market_path`).【F:src/gnn_trading/fund_manager_recommend.py†L16-L23】

## Legacy Data Scripts
- Yahoo downloader stores raw CSVs under `data/yahoo/<TICKER>.csv` and a combined `data/yahoo/panel.csv`.【F:src/gnn_trading/src/gnn_trading/data_prep_yahoo.py†L17-L48】

## Model Registry
- No dedicated on-disk registry exists; the quantamental registry only tracks estimator instances in memory.【F:src/gnn_trading/quantamental.py†L190-L211】
