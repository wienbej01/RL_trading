# How to Use These Prompts with Codex CLI

1) **Open a new Codex chat** in your repo root. Paste the content of:
   - `00_PERSONA_AND_GUARDRAILS.prompt.md` (once per session)
   - `01_ORCHESTRATOR.prompt.md`

2) Then, **sequentially feed** each step file from `codex/steps/` to Codex, one at a time, waiting for Codex to:
   - Print its PLAN (files, functions, patch points),
   - Apply minimal changes reusing existing code,
   - Run tests and QA gates,
   - Commit with the specified message.

**Order:**
- `steps/10_BRANCH_AND_BASELINE.prompt.md`
- `steps/20_AUDIT_REPO_AND_TICKERS.prompt.md`
- `steps/30_ENHANCEMENT_A_BC_AUX.prompt.md`
- `steps/40_ENHANCEMENT_B_RL_CONSTRAINTS.prompt.md`
- `steps/50_ENHANCEMENT_C_MULTITICKER_EMBEDS.prompt.md`
- `steps/60_OOS_EVAL_WINDOWS_AND_GATES.prompt.md`
- `steps/70_TUNING_ORCHESTRATOR_WRAP_EXISTING_TRAIN.prompt.md`
- `steps/80_MAKEFILE_WIRING_EXISTING_CLI.prompt.md`
- `steps/90_DRY_RUNS_AND_SANITY.prompt.md`
- `steps/99_FULL_RUN_COMMANDS.prompt.md`

**Notes**
- If the repo lacks certain flags, Codex will document precise file/line references for manual edits instead of guessing interfaces.
- Minute-data start date must remain **2020-10-01** or later; Codex should show exactly how your current scripts enforce this.
- Always keep diffs small; run tests after every change.
