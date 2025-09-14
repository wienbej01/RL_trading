# Enhancement C — Multi‑Ticker + Embeddings (Design)

Goals (opt‑in, backward‑compatible)
- Add lightweight, leak‑safe regime tags to features.
- Provide ticker identity signals to enable embedding‑like learning.
- Keep public interfaces and envs unchanged; integrate via existing pipeline/trainer.

What’s implemented
- Regime tags in `FeaturePipeline` behind `features.regime.enabled`:
  - `regime_vol` (rolling σ of returns, shifted by 1),
  - `regime_vol_bucket` (expanding terciles 0/1/2),
  - `regime_trend` (EMA of returns, shifted by 1),
  - `regime_trend_sign` (−1/0/1).
- Ticker identity one‑hots in `MultiTickerRLTrainer` behind `features.ticker_identity.enabled`:
  - Adds fixed columns `id_<TICKER>` across all per‑ticker frames with 1.0 at the active ticker, 0.0 otherwise.
  - Keeps shapes consistent across tickers and enables the policy to learn per‑ticker offsets akin to embeddings.

Future extension (not in this patch)
- Replace one‑hots with trainable embeddings in a custom SB3 policy to reduce dimensionality.
- Optional regime embeddings in model; requires policy changes only, not data.

Safety
- All tags are causal (t−1) and won’t leak future information.
- Flags default to off; no change when disabled.

Config example
```
features:
  regime:
    enabled: true
    vol_window: 60
    trend_window: 60
  ticker_identity:
    enabled: true
```
