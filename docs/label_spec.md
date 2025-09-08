# Intraday Label Specification

## Overview
Leakage-free label for RL reward shaping or auxiliary supervised loss: Future 1-bar return, evaluated at bar_close +1 (next bar open). Economic rationale: Direct proxy for immediate trade profitability in intraday momentum/mean-reversion; 1-bar horizon balances signal strength with noise, capturing alpha from short-term predictability without multi-period leakage.

## Label Definition
- **Label**: Forward 1-bar return = (close_{t+1} / close_t - 1), where t = bar_close.
- **Window**: Exact 1 bar ahead; embargo 1 bar (label computed using only data post-close_t).
- **No Leakage**: Shift(-1) ensures label uses future data unavailable at decision time; purge for splits/dividends by adjusting close via factor (if available in data).
- **Handling**: NaN for last bar; threshold for binary (long if >0.001, short if <-0.001, hold else) if needed for classification aux loss.

## Code Sketch
```python
# Assume df has 'close' at 1-min bars, timezone-aware index
df['fwd_return'] = df['close'].shift(-1) / df['close'] - 1  # 1-bar forward
df['fwd_return'] = df['fwd_return'].fillna(0)  # or dropna()
# Purge splits: if split_factor col exists
df['close_adj'] = df['close'] / df['split_factor'].cumprod()
df['label'] = df['close_adj'].shift(-1) / df['close_adj'] - 1
df['label'] = df['label'].where(df['volume'] > 0, np.nan)  # filter low-liq
# For RL reward: clip [-0.05, 0.05] to bound; binary for aux loss
df['binary_label'] = np.where(df['label'] > 0.001, 1, np.where(df['label'] < -0.001, -1, 0))
df = df.dropna(subset=['label'])  # embargo last bar
```

## Guardrails
- **Evaluation**: At bar_close +1 bar open; no intra-bar leakage.
- **Purged/Embargoed**: Adjust for splits (factor multiply); embargo 1 bar post-event.
- **Costs Parameterized**: Subtract modeled slippage/commission from label for net return.

## Mermaid Computation Flow
```mermaid
graph TD
    A[Bar Close Data: OHLCV_t] --> B[Compute Adjusted Close: close_adj_t = close_t / cumprod(split_factor)]
    C[Next Bar: close_adj_{t+1}] --> D[Label = close_adj_{t+1} / close_adj_t - 1]
    B --> D
    D --> E{Volume_t > Threshold?}
    E -->|No| F[NaN: Skip Low-Liq]
    E -->|Yes| G[Clip [-0.05, 0.05]; Binary if |label| > 0.001]
    F --> G
    G --> H[Shift(-1) Embargo: Drop Last Bar NaNs]
    H --> I[Leakage-Free Label for RL Reward/Aux Loss]
    style I fill:#90EE90