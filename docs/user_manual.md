# GNN Trading Fund Manager User Manual

## Overview

The fund manager console is a multi-agent decision support application that
combines a graph neural network (GNN) encoder with handcrafted analytics.  It
helps portfolio managers rank opportunities, manage holdings, and understand the
rationale behind each recommendation.

### Components

1. **Price Action Agent** – wraps the supervised GNN to estimate breakout
   probability, expected return, and volatility.
2. **Technical Analysis Agent** – interprets RSI, MACD, slope, and compression
   readings from the engineered feature set.
3. **News Sentiment Agent** – consumes optional news embeddings to gauge market
   tone.
4. **Macro Agent** – adjusts aggressiveness based on macro indicators such as
   GDP growth, inflation, and interest-rate trends.
5. **Quantamental Agent** – fuses quantitative signals and fundamental factors
   with explainable scoring, portfolio construction, and backtesting support.
6. **Decision Engine** – fuses the weighted agent reports into a single action
   (BUY, ADD, HOLD, TRIM, SELL).

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your OHLCV CSV files live under a directory that matches the
   `--data-template` pattern (defaults to `data/{ticker}.csv`). Each CSV must
   include columns: `date`, `open`, `high`, `low`, `close`, and `volume`.
3. (Optional) Prepare a JSON file mapping `YYYY-MM-DD` to news embedding vectors
   and another JSON file holding macro assumptions, e.g.:
   ```json
   {
     "gdp_growth": 0.03,
     "inflation": 0.025,
     "rate_trend": 0.01
   }
   ```

## Launching the Interface

Run the interactive shell with:
```bash
PYTHONPATH=src python -m gnn_trading.fund_manager \
  --db my_portfolio.db \
  --data-template "data/{ticker}.csv" \
  --checkpoint path/to/model.pt \
  --news-json data/news_vectors.json \
  --macro-json config/macro.json
```

Key arguments:
- `--window`, `--pred-horizon` – control the lookback and forecast horizon.
- `--device` – set to `cpu`, `cuda`, or `auto`.
- `--model-json`, `--labels-json` – override encoder or label parameters.

### Graphical Fund Manager

For a point-and-click experience launch the Tkinter GUI:

```bash
PYTHONPATH=src python -m gnn_trading.fund_manager_gui \
  --db my_portfolio.db \
  --data-template "data/{ticker}.csv" \
  --checkpoint path/to/model.pt \
  --news-json data/news_vectors.json \
  --macro-json config/macro.json
```

The GUI mirrors the CLI capabilities while adding:

- Holdings table with add/edit/remove buttons.
- Interactive analysis grid sorted by risk-adjusted return.
- Detailed agent insights panel with per-agent explanations.
- Bar chart visualising expected returns for the analysed universe.
- Portfolio summary banner showing current value, model-implied change, and suggested actions.

Use the **Analyze Holdings** button to score all saved tickers or **Analyze Selected** for a subset. Selecting a row in the results table populates the insight pane and updates the chart.

## Command Reference

| Command | Description |
| --- | --- |
| `add TICKER SHARES COST` | Add or update a holding. |
| `remove TICKER` | Delete a holding. |
| `holdings` | List all stored positions. |
| `macro KEY=VALUE ...` | Update macro indicators (values expressed as decimals, e.g. `0.03` for 3%). |
| `score [TICKER ...]` | Run the agent ensemble for selected tickers or all holdings. |
| `summary` | Show aggregate portfolio valuation and implied change. |
| `agents` | Display the active agent roster. |
| `export FILE.json` | Save holdings and macro view to JSON. |
| `quit` | Exit the application. |

Commands support tab completion and history (provided by the `cmd` module).

## Example Workflow

1. **Seed the database**
   ```text
   fund-manager> add AAPL 10 165
   fund-manager> add MSFT 5 240
   ```
2. **Check holdings**
   ```text
   fund-manager> holdings
   Ticker   Shares       Cost Basis
   ---------------------------------
   AAPL        10.00         165.00
   MSFT         5.00         240.00
   ```
3. **Configure macro outlook**
   ```text
   fund-manager> macro gdp_growth=0.025 inflation=0.022 rate_trend=0.005
   Updated macro indicators.
   ```
4. **Run analysis**
   ```text
   fund-manager> score AAPL MSFT
   === Final Decision ===
   Ticker: AAPL
   Recommended action: BUY
   Composite score: 0.512
   Rationale:
     - Price Action: BUY (score=0.48, conf=0.72)
     - Technical Analysis: ACCUMULATE (score=0.35, conf=0.70)
     - News Sentiment: POSITIVE (score=0.18, conf=0.60)
     - Macro: RISK-ON (score=0.03, conf=0.60)
   ...
   ```
5. **Export the setup**
   ```text
   fund-manager> export snapshot.json
   ```

## Tips

- If the application reports that there is not enough data for a ticker, extend
  the CSV history so it contains at least `window + pred_horizon` rows.
- Use the `macro` command to quickly test different macro scenarios before
  executing trades.
- The news agent gracefully handles missing embeddings; supply them only when
  available.
- To experiment with new agents, import `default_agents` from `gnn_trading.agents`
  and append custom subclasses.

## Troubleshooting

- **Model fails to load** – confirm the checkpoint path and that it was saved
  with `torch.save(model.state_dict(), path)`.
- **CUDA errors** – switch to CPU with `--device cpu` or ensure drivers are
  installed.
- **Database locked** – close other instances using the same SQLite file.

Happy trading!
