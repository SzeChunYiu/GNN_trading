# Migration: Central Configuration & Feature Flags

This guide helps existing deployments adopt the new configuration and registry
controls without disrupting current workflows.

## 1. Prepare the Config File
1. Copy `config_example.yaml` to `config/app.yaml`.
2. Review the defaults; by design, all new functionality (LLM, UI bridges,
   registry helpers) remains disabled until explicitly enabled.
3. Set `data.start_date` / `data.end_date` if timeframe validation should trim
   historical artefacts.

## 2. Update Automation Scripts
- Pass `--config config/app.yaml` to CLI entry points (`fund_manager`, registry
  commands) if a non-default location is used.
- Ensure deployment environments can read the YAML (e.g., mount the file in
  containers).

## 3. Enable Validators (Optional)
- Timeframe and leakage validators execute automatically before registry
  training when date bounds are defined.  Keep bounds `null` to skip trimming.
- Review generated reports in `reports/validation_timeframe.md` and
  `reports/validation_leakage.md`.

## 4. Model Registry Roll-Out
- Confirm `registry.path` points to a writable location.
- Run `python -m gnn_trading.registry_cli train --gold-root ...` to produce the
  first guarded run; inspect the manifest before promoting to production.
- The CLI updates a `latest` symlink; use it for downstream evaluation/predict
  flows to maintain compatibility.

## 5. Reversion Strategy
- Delete or rename `config/app.yaml` to fall back to legacy defaults.
- Remove generated `models/gnn_crosssec/<run_id>` directories to roll back to
  previous model snapshots; re-point the `latest` symlink manually if needed.

## 6. Known Limitations
- External provider integrations (Stooq, SEC, GDELT) are not yet implemented;
  placeholders exist for future development.
- UI enhancements beyond the Tkinter console remain disabled via `ui.enabled`.
