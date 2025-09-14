# Step 8 — Makefile Wiring (Existing Entrypoints Only)

**Objectives**
- Create or update the Makefile to expose targets that call **existing entrypoints** only:
  - `tickers_train`: print resolved training tickers using current tooling.
  - `tickers_test`: print resolved backtest tickers using current tooling.
  - `bc_train`: run BC warm-start (short window acceptable if configured).
  - `rl_train`: run RL fine-tune from BC checkpoint with constraints enabled.
  - `backtest_oos`: run OOS evaluation across windows & held-out tickers.
  - `tune`: run the tuning orchestrator.

**Constraints**
- If a flag does not exist, do not invent it. Document manual edits with exact file/line references instead.

**Actions**
1) Inspect current CLI scripts and print exact commands each target will run (echo before execute).
2) Implement targets calling those commands.
3) Commit: `build: Makefile targets for tickers, bc_train, rl_train, backtest_oos, tune`
