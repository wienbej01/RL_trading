# Orchestrator — Execute All Steps in Order

**You must execute the following step files in this exact order**, reading and following their instructions. At each step:
- Inspect the repository to find existing file paths and entrypoints.
- Write a brief PLAN (filenames, functions, expected diffs).
- Apply minimal patches reusing existing code.
- Run tests and QA gates.
- Commit changes.

**Execution Order**
1) steps/10_BRANCH_AND_BASELINE.prompt.md
2) steps/20_AUDIT_REPO_AND_TICKERS.prompt.md
3) steps/30_ENHANCEMENT_A_BC_AUX.prompt.md
4) steps/40_ENHANCEMENT_B_RL_CONSTRAINTS.prompt.md
5) steps/50_ENHANCEMENT_C_MULTITICKER_EMBEDS.prompt.md
6) steps/60_OOS_EVAL_WINDOWS_AND_GATES.prompt.md
7) steps/70_TUNING_ORCHESTRATOR_WRAP_EXISTING_TRAIN.prompt.md
8) steps/80_MAKEFILE_WIRING_EXISTING_CLI.prompt.md
9) steps/90_DRY_RUNS_AND_SANITY.prompt.md
10) steps/99_FULL_RUN_COMMANDS.prompt.md

**Global constraints**
- Do not invent filenames; inspect and reference actual existing ones.
- Prefer config toggles over interface changes.
- Keep all improvements **optional** and **backward-compatible**.

**Begin by printing the repo root and Python version**, then proceed to Step 1.
