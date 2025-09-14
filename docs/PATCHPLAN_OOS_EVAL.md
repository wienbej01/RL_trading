# Patch Plan — OOS Evaluation

Files
- `scripts/oos_eval.py` (new): CLI wrapper around `MultiTickerRLTrainer.backtest` to evaluate N disjoint windows over held-out tickers.
- `docs/DESIGN_OOS_EVAL.md` (this design).

Behavior
- Derives three windows spanning the last ~18 months of available data, evenly partitioned.
- Filters data/features to each window and requested tickers.
- Loads SB3 model and runs backtest; collects metrics and computes gates.
- Aggregates summary stats across windows and writes `oos_results.json` to the output directory.

No interface changes; defaults remain unaffected.
