# Step 9 — Dry Runs & Sanity Checks (Token-Safe)

**Objectives**
- Prove wiring without heavy runs:
  - DR1: Print resolved training & testing tickers (using existing “print”/“verbose”/“dry-run” modes if available).
  - DR2: Run BC on a 2–3 day span for 1–2 symbols already cached locally; verify checkpoint creation.
  - DR3: Run OOS evaluator on ~5 trading days; verify cadence/action-mix fields appear.

**Actions**
1) Create `docs/DRY_RUNS.md` with exact commands discovered for the repo.
2) Run the dry runs and collect artifacts under existing report directories.
3) Commit: `docs: dry runs completed; wiring validated`
