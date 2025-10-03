# Step 7 — Tuning Orchestrator (Wrap Existing Trainer)

**Objectives**
- Implement a lightweight orchestrator that **wraps the current training entrypoint** to run 8–16 trials over:
  - learning rate, entropy schedule, KL weight, clip, γ, λ,
  - action floor ε, aux loss weight, embedding dims (if enabled).
- Selection criteria:
  - Must pass cadence/action-mix/DD gates,
  - Rank by OOS median expectancy (after costs) and MAR.

**Constraints**
- Do not replace the trainer. Use existing CLI/config override mechanisms.

**Actions**
1) Design plan `docs/DESIGN_TUNING.md`: how to pass overrides with the repo’s current config/CLI scheme.
2) Implement the orchestrator (module or script) that writes per-trial configs and launches training + OOS evaluation.
3) Add a summarizer collecting OOS metrics, producing a top-N table.
4) Tiny dry-run test (1–2 toy trials).
5) Commit: `feat: tuning orchestrator reusing existing trainer; OOS-based selection`
