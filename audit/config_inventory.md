# Configuration Inventory

## Command-Line Interfaces

### `python -m gnn_trading.train`
- `--ohlcv` (required): path to OHLCV CSV.【F:src/gnn_trading/train.py†L150-L154】
- `--news` (default `None`): optional news CSV.【F:src/gnn_trading/train.py†L154-L156】
- `--date_col` (default `"date"`), `--text_col` (default `"headline"`), `--news_model` (default `"ProsusAI/finbert"`), `--max_per_day` (default `20`): news ingestion knobs.【F:src/gnn_trading/train.py†L155-L159】
- Core training knobs: `--seed` (`42`), `--device` (`auto`), `--window` (`60`), `--pred_horizon` (`5`), `--breakout_lookback` (`20`), `--breakout_pct` (`0.01`), `--vol_z_threshold` (`0.0`), `--confirm_days` (`0`), `--epochs` (`8`), `--lr` (`1e-3`), `--hid` (`64`), `--heads` (`2`), `--gru_hid` (`64`), `--dropout` (`0.2`).【F:src/gnn_trading/train.py†L159-L172】
- Walk-forward options: `--walkforward` flag plus `--n_splits` (`4`), `--min_train_size` (`600`).【F:src/gnn_trading/train.py†L173-L175】

### `python -m gnn_trading.ssl_pretrain`
- Shared data knobs: `--ohlcv` (required), `--date_col` (`"date"`), `--window` (`60`), `--pred_horizon` (`5`).【F:src/gnn_trading/ssl_pretrain.py†L58-L66】
- Label filters: `--breakout_lookback` (`20`), `--breakout_pct` (`0.01`), `--vol_z_threshold` (`0.0`), `--confirm_days` (`0`).【F:src/gnn_trading/ssl_pretrain.py†L66-L69】
- SSL hyperparameters: `--epochs` (`6`), `--feat_mask_ratio` (`0.15`), `--jitter_std` (`0.01`), `--temperature` (`0.2`), `--lr` (`1e-3`), `--hid` (`64`), `--heads` (`2`), `--gru_hid` (`64`), `--dropout` (`0.2`).【F:src/gnn_trading/ssl_pretrain.py†L70-L78】

### `python -m gnn_trading.backtest`
- `--signals` (required), `--ohlcv` (required), `--threshold` (`0.6`), `--hold` (`5`).【F:src/gnn_trading/backtest.py†L28-L34】

### `python -m gnn_trading.fund_manager`
- Storage and data access: `--db` (`fund_manager.db`), `--data-template` (`data/{ticker}.csv`), `--data-root` (`local_data`).【F:src/gnn_trading/fund_manager.py†L621-L635】
- Model inputs: `--checkpoint`, `--device` (`auto`), `--window` (`60`), `--pred-horizon` (`5`), `--model-json`, `--labels-json`, `--news-json`, `--macro-json`, `--date-col` (`"date"`).【F:src/gnn_trading/fund_manager.py†L623-L634】
- LLM/housekeeper toggles: `--disable-housekeeper`, `--housekeeper-provider`, `--housekeeper-model`.【F:src/gnn_trading/fund_manager.py†L635-L649】

### `python -m gnn_trading.fund_manager_gui`
- Mirrors the CLI but omits LLM toggles: `--db`, `--data-template`, `--checkpoint`, `--device`, `--window`, `--pred-horizon`, `--model-json`, `--labels-json`, `--news-json`, `--macro-json`, `--date-col`.【F:src/gnn_trading/fund_manager_gui.py†L471-L483】

### `python -m gnn_trading.fund_manager_recommend`
- Batch mode arguments: `--db`, `--tickers`, `--data-template`, `--checkpoint`, `--device`, `--window`, `--pred-horizon`, `--model-json`, `--labels-json`, `--news-json`, `--date-col`, `--data-root`, `--macro-json`, `--output`.【F:src/gnn_trading/fund_manager_recommend.py†L69-L83】

## Static Configuration Examples
- `config_example.yaml` documents optional keys grouped under `seed`, `device`, `data`, `labels`, `ssl`, `model`, `supervised`, `walkforward`, and `news`; these values are not automatically loaded by current entry points.【F:config_example.yaml†L1-L34】

## Environment & Secrets
- No dedicated environment variable parsing or secrets management utilities exist in the repository; CLI defaults and optional JSON files are the only runtime configuration inputs.【F:src/gnn_trading/fund_manager.py†L621-L712】
