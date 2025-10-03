# Baseline Snapshot — Step 10 (Branch & Validation)

- Branch: `feat/rl_refactor_bc_constraints_multiticker_oos`
- Repo root: `/home/jacobw/RL_trading/rl-intraday`
- Python/venv: used repo `venv/` for tooling

## Commands Run
- Pytest: `source venv/bin/activate && PYTHONPATH=$PWD pytest -q`
  - Result: 6 errors during collection (import/API mismatches, missing optional deps like `gym`, a couple of syntax issues in tests).
- Ruff: `source venv/bin/activate && ruff check .`
  - Result: ruff 0.12.9; reported many findings (approx 2k), with auto-fixable items available. Also noted config deprecation (move top-level to `[tool.ruff.lint]`).
- Mypy: `source venv/bin/activate && mypy src`
  - Result: mypy 1.17.1; reported package naming/config issues; needs follow-up configuration and type fixes.

## Optional Backtest (Deferred)
- Not executed to keep this baseline light. Example minimal commands:
  - Quick sanity: `python scripts/run_quick_test.py`
  - Tiny walk-forward: `python scripts/walkforward_train_eval.py --config configs/settings.yaml --data data/raw/SPY_1m.parquet --features data/features/SPY_features.parquet --output runs/wf_baseline --train-days 2 --test-days 1 --n-envs 1 --total-steps 100`

## Notes
- Tests currently fail at import stage for several modules and expectations (e.g., `ConfigError` symbol, `DataLoaderError`), plus syntax errors in a few tests. Lint surface is large; many are auto-fixable.
- This snapshot records the pre-refactor state only; no code changes made for stabilization in this step.
