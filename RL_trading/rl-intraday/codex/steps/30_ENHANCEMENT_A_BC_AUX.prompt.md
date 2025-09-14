# Step 3 — Enhancement A: Behavior Cloning Warm-Start + Auxiliary Forecasting Head

**Objectives**
- Add an optional **BC warm-start** stage seeded by simple “teacher” strategies.
- Add an optional **Auxiliary Head** predicting short-horizon returns (r_{t+1}, r_{t+3}, r_{t+5}) to improve representation.
- **Zero assumptions**: adapt to existing model/feature/training layout.

**Constraints**
- Do not rename files. Extend existing modules where possible.
- Add **config toggles** to enable/disable BC and AuxHead.
- Ensure **no feature leakage**: labels and targets must be computed with data available at decision time (t-1 alignment).

**Actions**
1) Design & Patch Plan
   - Write `docs/DESIGN_BC_AUX.md`: exact files/functions to extend, config keys, and how to initialize RL from BC weights.
   - Write `docs/PATCHPLAN_BC_AUX.md`: granular diffs to apply (paths + line anchors).
2) Implementation
   - Add teacher strategies (e.g., Opening Range Breakout + retest; VWAP band fade; ATR swing) in the repo-consistent place.
   - Implement BC label generation that uses existing loaders and time index; save labels to the repo’s standard data format/dir.
   - Add an AuxHead to the model behind a config flag; incorporate aux loss (weighted).
   - Implement BC training loop that writes a checkpoint compatible with the existing RL trainer.
3) Tests
   - BC label smoke test (tiny date span) verifying ≥20% LONG+SHORT combined.
   - Model shape test with/without AuxHead.
   - BC training 1–2 epochs overfit sanity; checkpoint saved.
4) Run tests and commit.

**Deliverables**
- New BC and AuxHead capabilities (opt-in).
- Tests passing; minimal code changes reusing existing modules.
- Commit: `feat: BC warm-start and auxiliary forecasting head with tests and docs`
