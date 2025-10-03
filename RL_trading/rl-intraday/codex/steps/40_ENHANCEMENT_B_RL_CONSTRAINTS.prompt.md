# Step 4 — Enhancement B: RL Constraints to Prevent HOLD Collapse and Enforce Cadence

**Objectives**
- Add optional constraints to the existing RL loop:
  - Entropy schedule (start high → decay to floor).
  - KL-to-uniform prior penalty (decay to 0).
  - Early-training action floors: π(LONG), π(SHORT) ≥ ε for a warm-up fraction.
  - Trade-count regularizer towards target band; overtrade penalty.
  - Optional small HOLD tax (training only).
  - Max trades/day cap (if environment/backtester supports it).

**Constraints**
- Do not replace the RL algorithm. Extend current trainer in place.
- All features are behind config toggles; defaults preserve current behavior.

**Actions**
1) Design & Patch Plan
   - `docs/DESIGN_RL_CONSTRAINTS.md`: exact insertion points (logits, loss assembly), where to count trades/day, and how to pass configs.
   - `docs/PATCHPLAN_RL_CONSTRAINTS.md`: exact diffs.
2) Implementation
   - Add schedules and regularizers per plan; ensure back-compat.
3) Tests
   - Unit tests for schedule shapes (entropy/KL).
   - Unit test that action floors increase LONG/SHORT early without exploding volume.
   - Tiny integration test demonstrating improved action diversity on a small slice.
4) Run tests and commit.

**Deliverables**
- Optional RL constraints wired in.
- Commit: `feat: RL constraints (entropy, KL, action floors, cadence regularizer, hold tax) + tests`
