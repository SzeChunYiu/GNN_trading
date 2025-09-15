# Data sources (free & popular)

This project was built to work with simple CSVs. Here are common ways to get data:

## 1) Yahoo Finance via `yfinance` (free, research use)
- Python package that pulls historical OHLCV from Yahoo Finance.
- Install: `pip install yfinance`
- Script in this repo: `python -m gnn_trading.data_prep_yahoo --tickers AAPL,MSFT,SPY --start 2015-01-01 --end 2025-09-15`
- Docs: yfinance reference and docs.

## 2) Alpha Vantage (free tier, API key required)
- Get a key: https://www.alphavantage.co/support/#api-key
- Script in this repo: `python -m gnn_trading.data_prep_alphavantage --tickers AAPL,MSFT --api_key YOUR_KEY --start 2015-01-01`
- Also provides **NEWS_SENTIMENT** endpoint if you want headlines to CSV.

## 3) Stooq (free historical downloads)
- Browse https://stooq.com/db/h/ and download CSVs manually, then place them in `data/stooq/`.
- Script `data_prep_stooq` can normalize Stooq CSVs into the project's standard format.

> Professional/low-latency use usually needs licensed data. Nasdaq Data Link (formerly Quandl) and paid vendors exist if you need SLAs/completeness.
