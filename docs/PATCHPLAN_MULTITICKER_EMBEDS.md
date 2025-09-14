# Patch Plan — Enhancement C (Multi‑Ticker + Embeddings)

Scope (minimal, opt‑in)
- Compute regime tags in `FeaturePipeline` when `features.regime.enabled`.
- Add ticker identity one‑hots in `MultiTickerRLTrainer` when `features.ticker_identity.enabled`.
- No interface changes; reuse existing CLI and env wiring.

Files & Diffs
1) `src/features/pipeline.py`
   - Read `self.regime_config = config.get('regime', {})`.
   - After base features are computed, add causal tags:
     - `regime_vol`, `regime_vol_bucket`, `regime_trend`, `regime_trend_sign`.

2) `src/rl/multiticker_trainer.py`
   - After `X_map` is built and before env creation, inject columns `id_<TICKER>` for each ticker, one‑hot per frame.
   - Controlled by `features.ticker_identity.enabled`.

Validation
- Shapes remain consistent across tickers; env and policy need no changes.
- Disabled by default; enabling produces additional columns visible in saved feature metadata.
