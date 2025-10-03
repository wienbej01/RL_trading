# Step 1 — Create Branch & Record Baseline

**Objectives**
- Create a new Git branch for this refactor & enhancement work.
- Record a baseline of tests and current backtest metrics (if a script exists).

**Actions**
1) Print repo root and current branch: `pwd`, `git status -sb`.
2) Create and switch to a new branch:
   - Branch name: `feat/rl_refactor_bc_constraints_multiticker_oos`
   - If branch exists, append a numeric suffix.
3) Baseline validations:
   - Run: `pytest -q || true`
   - Run: `ruff . || true`
   - Run: `mypy . || true`
   - If a backtest/eval entrypoint exists, run the **smallest** supported backtest (e.g., single ticker, few days) and save report as `out/reports/baseline_pre_refactor.json` (or the repo’s standard path).
4) Commit a “baseline snapshot” (no code changes yet). Include a short `BASELINE_NOTES.md` summarizing what runs and where reports were saved.

**Deliverables**
- New branch checked out.
- Tests/lint/type checks run.
- Optional baseline backtest report saved.
- Commit with message: `chore: baseline snapshot before RL refactor`.
