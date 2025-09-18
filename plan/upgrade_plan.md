# Minimal Risk Gap Plan

## Scope Summary
- **In scope**: Introduce configuration-driven controls, guarded data ingestion validators, disk-backed model registry commands, and reporting needed to close critical gaps identified in the audit. Focus on enabling timeframe enforcement, leakage detection, and registry lifecycle with minimal disruption to existing fund manager workflows.
- **Out of scope**: Replacing the existing UI with a Next.js/Tauri app, building full data lake connectors for external providers, implementing full LLM research pipelines, or overhauling feature engineering beyond what is required for leakage validation. These larger features remain future work.

## Tasks
| ID | Owner | Description | Acceptance Criteria |
| --- | --- | --- | --- |
| T1 | codex | Add central application configuration (`config/app.yaml`) and loader wiring guarded by flags. | Config file created with required keys defaulting to safe/off states; runtime reads values without breaking existing CLI defaults. |
| T2 | codex | Generate upgrade documentation: `docs/config_reference.md` and `docs/migrations/MIGRATE_TO_FLAGS.md`; update sample YAMLs. | Docs list new keys, defaults, and migration steps; sample configs reference new flags without enabling disruptive behavior. |
| T3 | codex | Implement timeframe validator producing `reports/validation_timeframe.md`. | Validator scans silver/gold directories, respects config start/end dates, reports counts, and exits non-zero on violations. |
| T4 | codex | Implement leakage validator with sampling and `reports/validation_leakage.md`. | Validator checks feature-label alignment across horizons, fails fast on violations, and surfaces offending columns/rows. |
| T5 | codex | Extend CLI/registry utilities to create disk-backed model registry per spec with train/evaluate/predict commands under feature flags. | Registry path configurable, run folders contain required artefacts, CLI commands operate when enabled and remain no-ops otherwise. |

## Dependency Graph
- T1 → {T3, T4, T5}: Validators and registry rely on configuration keys.
- T2 depends on completion of T1 to document accurate defaults.
- T5 depends on T1 for registry path configuration.
- T3 and T4 are independent once T1 is complete.

## Roll-out Plan
- **Feature Flags**: New behavior gated by `config/app.yaml` keys (e.g., `llm.enabled`, `ui.enabled`, validator toggles inferred from config sections). Defaults keep legacy behavior unchanged.
- **Config Keys**: Introduce required keys with conservative defaults (disabled or existing behavior). Provide migration doc and sample updates.
- **Migration Steps**: Ship migration notes instructing operators to copy `config_example.yaml` to `config/app.yaml`, review defaults, and gradually enable features in non-production before rollout.

## Test Plan
- Unit tests covering config loading and validator edge cases.
- Integration smoke: run validators against sample data directories with clean and violating cases.
- CLI tests: invoke `train`, `evaluate`, and `predict` under dry-run/small dataset settings to ensure registry artefacts are produced.

## Reversion Plan
- Retain previous CLI paths; disabling new config keys restores legacy behavior.
- Validators write reports only; operators can bypass by toggling flags or removing config file.
- Registry changes operate in separate directories; deleting the new run folder reverts to previous state. Keep backup of prior models to relink `latest` symlink if needed.

## Capabilities Matrix v2 (Post-Plan Expectations)
| Component | Expected Status | Notes |
| --- | --- | --- |
| Local database layout | Partial | Still relying on existing LocalDataHub; plan does not restructure storage. |
| Timeframe-bounded ingest | Partial | Validators enforce timeframe but ingestion connectors remain future work. |
| Leakage-safe features & labels | Partial | Validators ensure safety but new feature jobs remain pending. |
| Cross-sectional batching & lagged edges | Missing | Deferred. |
| Model registry on disk | Partial | New registry lifecycle provided behind config flag; advanced metadata may remain limited. |
| LLM extractor with provenance | Missing | Deferred. |
| Walk-forward evaluation | Partial | Evaluate CLI adds purged splitting; further enhancements future. |
| UI (multi-page) | Missing | Deferred. |
| Backtest & recommendation surface | Partial | Existing functionality maintained. |
| Config management | Partial | New central config loader with docs improves status though secrets handling remains future work. |
