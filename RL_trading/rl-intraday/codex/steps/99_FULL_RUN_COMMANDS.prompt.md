# Step 10 — Full Run Sequence (Use Existing Entrypoints)

**Objectives**
- Provide the final, end-to-end commands (Make targets or scripts) to:
  1) Print tickers for training and backtest.
  2) Run BC warm-start (full span per project settings).
  3) Run RL fine-tune with constraints.
  4) Run OOS evaluation across windows and held-out tickers.
  5) Optionally run tuning orchestrator and select top-N.

**Actions**
- Create `docs/FULL_RUNBOOK.md` listing exact commands and expected output locations consistent with this repo’s conventions.
- Do not execute heavy runs automatically here. Just output the runbook.
- Commit: `docs: full runbook`
