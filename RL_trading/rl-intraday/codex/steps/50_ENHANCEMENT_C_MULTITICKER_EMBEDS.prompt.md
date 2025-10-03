# Step 5 — Enhancement C: Multi-Ticker Training with Ticker/Regime Embeddings

**Objectives**
- Reuse the existing loaders to emit a `ticker_id` (or repo-equivalent) without breaking single-ticker paths.
- Add regime tags (volatility tercile, trend slope sign, liquidity tercile, time-of-day bucket) using leak-safe rolling stats.
- Add optional embedding layers for ticker_id and regime tags; concatenate with existing encoded features.

**Constraints**
- No renames; minimal edits.
- All additions are opt-in via config.

**Actions**
1) Design & Patch Plan
   - `docs/DESIGN_MULTITICKER_EMBEDS.md`
   - `docs/PATCHPLAN_MULTITICKER_EMBEDS.md`
2) Implementation
   - Compute regime tags in the current feature pipeline style.
   - Inject `ticker_id` using or extending existing mapping logic.
   - Add embedding modules and merge into model forward pass when enabled.
3) Tests
   - Loader test: shapes & null checks; verifying regime tags and ticker_id present when enabled.
   - Model test: forward shapes with/without embeddings.
4) Run tests and commit.

**Deliverables**
- Optional multiticker + embeddings capability.
- Commit: `feat: multiticker training with ticker/regime embeddings + tests`
