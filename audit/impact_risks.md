# Impact & Risk Timeline

1. **Restructuring the on-disk data lake**
   - Current tooling resolves OHLCV files via `data_template.format(ticker=...)` and falls back to `LocalDataHub.market_path`; reorganising directories without compatibility shims will break fund manager and batch recommendation flows.【F:src/gnn_trading/fund_manager.py†L621-L714】【F:src/gnn_trading/fund_manager_recommend.py†L16-L65】
   - Mitigation: introduce migration helpers that mirror legacy paths into the new bronze/silver/gold layout before switching defaults.

2. **Expanding feature/label horizons**
   - The dataset assumes a single `pred_horizon` and emits one breakout label; multi-horizon labels will alter tensor shapes consumed by training, inference, and recommendation pipelines.【F:src/gnn_trading/dataset.py†L8-L55】【F:src/gnn_trading/train.py†L117-L148】
   - Mitigation: gate new features behind config flags and update loaders/heads to preserve existing behaviour when only one horizon is requested.

3. **Introducing cross-sectional graphs**
   - Present models build a linear edge index per ticker; adding correlation edges changes `edge_index` dimensions and could invalidate saved checkpoints or SSL weights.【F:src/gnn_trading/dataset.py†L28-L55】【F:src/gnn_trading/ssl_pretrain.py†L36-L55】
   - Mitigation: version encoders, provide conversion scripts, and allow opt-in cross-sectional mode.

4. **Persisting models in a registry**
   - The quantamental agent currently instantiates estimators in-memory; switching to disk-backed models requires updates to loading sites and to CLI arguments for run IDs.【F:src/gnn_trading/quantamental.py†L190-L398】【F:src/gnn_trading/agents.py†L214-L282】
   - Mitigation: keep the in-memory path as a fallback and add compatibility layers that can read legacy pickles.

5. **Embedding LLM extraction workflows**
   - The AI housekeeper presently offers optional prompts; integrating authenticated OpenAI calls introduces dependency and rate-limit failures that must be surfaced gracefully through the CLI/UI.【F:src/gnn_trading/ai_housekeeper.py†L26-L88】【F:src/gnn_trading/fund_manager.py†L635-L707】
   - Mitigation: wrap API usage with retry/backoff, allow offline fallbacks, and guard with explicit config flags.

6. **Modern UI rollout**
   - Tkinter GUI is standalone; replacing or augmenting with a web UI could duplicate business logic unless actions are exposed through a service layer.【F:src/gnn_trading/fund_manager_gui.py†L82-L506】
   - Mitigation: extract shared orchestration into reusable modules and ensure GUI/CLI tests cover both pathways before migrating users.

7. **Walk-forward enhancements**
   - Adding purged/embargoed splits and richer metrics changes expected outputs (`outputs/signals.csv`, metrics printing) and may break downstream scripts that parse the current format.【F:src/gnn_trading/train.py†L69-L148】
   - Mitigation: version output filenames or append new columns while keeping legacy headers intact by default.
