# Trading Rulebook for RL Intraday System

## Overview
Precise, testable rules for RL policy outputs, ensuring symmetric long/short, leakage-free signals evaluated at bar_close +1 bar. Economic rationale: Fixed RR 1:2 captures asymmetric risk-reward in intraday mean-reversion/momentum; filters avoid low-liquidity/high-vol regimes; management limits drawdown via sizing/EOD flat. Rules guide policy training (e.g., reward for compliance) and serve as baseline comparator.

## Entry Rules
- **Long Entry**: Policy action=1 (post-softmax signal >0.5 threshold) and flat position; size 1% equity risk.
- **Short Entry**: Policy action=-1 (signal <-0.5) and flat; symmetric sizing.
- **No Entry**: Hold (action=0) if |signal| <=0.5 or filters fail; no pyramiding (max 1 position).
- **Rationale**: Thresholds ensure conviction (e.g., strong momentum via features like MACD>0 + RSI<70 for long); symmetry prevents bias.

## Exit Rules
- **Stop Loss**: Exit at 1R loss (entry - 1*ATR for long; entry +1*ATR for short), or triple-barrier touch.
- **Take Profit**: Exit at 2R profit (entry +2*ATR long; entry -2*ATR short).
- **Time Stop**: Exit after 30 min holding (intraday horizon limit).
- **EOD Flatten**: Force close at 16:00 ET with inventory penalty in reward.
- **Rationale**: 1:2 RR targets positive expectancy (win rate >33% breakeven); time-stop prevents overnight risk in intraday system.

## Filters & Sessions
- **Sessions**: Regular Trading Hours (RTH) 09:30-16:00 ET only; skip first/last 5 min (no-trade window).
- **Vol Filter**: Skip if VIX >30 (high regime, increased noise/costs).
- **Liquidity Filter**: Skip if RVOL <0.5 or volume < daily avg (avoid illiquid bars).
- **Rationale**: RTH captures 80% volume/alpha; VIX filter avoids fat tails; liquidity ensures executable signals without excessive slippage.

## Risk Management
- **Position Sizing**: 1% equity risk per trade (contracts = risk / (stop_distance * point_value)); max 3 concurrent (portfolio risk 3%).
- **Daily Limits**: Kill-switch if daily drawdown >3R (flatten all); max trades/day=8 (via activity shaping).
- **Costs**: Parameterized (commission 0.35/share, slippage 0.5bps, impact scaled by size); included in reward.
- **Rationale**: Kelly-inspired sizing maximizes growth under vol; caps prevent blowups, aligning with hedge fund prudence.

## Code Sketch for Rule Compliance Check
```python
def apply_rules(df, policy_signals, atr_col='atr', vix_col='vix'):
    df = df.copy()
    df['signal'] = policy_signals  # from RL model
    df['action'] = 0
    df['position'] = 0.0
    df['entry_price'] = np.nan
    current_pos = 0
    for i in range(1, len(df)):
        ts = df.index[i]
        if not is_rth(ts): continue  # session filter
        if df['vix'].iloc[i] > 30 or df['rvol'].iloc[i] < 0.5: continue  # vol/liquidity
        signal = df['signal'].iloc[i]
        atr = df['atr'].iloc[i]
        if current_pos == 0:
            if signal > 0.5:
                current_pos = 1  # long
                df.loc[df.index[i], 'action'] = 1
                df.loc[df.index[i], 'entry_price'] = df['close'].iloc[i]
            elif signal < -0.5:
                current_pos = -1  # short
                df.loc[df.index[i], 'action'] = -1
                df.loc[df.index[i], 'entry_price'] = df['close'].iloc[i]
        else:
            # check exits
            entry = df['entry_price'].iloc[i-1]
            r_realized = (df['close'].iloc[i] - entry) * current_pos / atr
            if (current_pos > 0 and (r_realized <= -1 or r_realized >= 2)) or \
               (current_pos < 0 and (r_realized >= 1 or r_realized <= -2)) or \
               (i - df[df['entry_price'].notna()].index.get_loc(ts) >= 30):  # time-stop
                current_pos = 0
                df.loc[df.index[i], 'action'] = 0  # close
        df.loc[df.index[i], 'position'] = current_pos
        if is_eod(ts): current_pos = 0  # EOD flat
    # Size: post-facto, but in sim: contracts = 0.01 * equity / (atr * point_value)
    return df

def is_rth(ts): return '09:30' <= ts.time() <= '16:00'
def is_eod(ts): return ts.time() >= '16:00'
```

## Mermaid Decision Flow
```mermaid
graph TD
    A[Bar Close Signal] --> B{Is RTH & VIX<30 & RVOL>0.5?}
    B -->|No| C[Hold: action=0]
    B -->|Yes| D{Signal >0.5?}
    D -->|Yes| E[Long Entry: action=1, size=1% risk]
    D -->|No| F{Signal <-0.5?}
    F -->|Yes| G[Short Entry: action=-1, size=1% risk]
    F -->|No| C
    E --> H{Check Exits: Stop 1R / TP 2R / Time 30m / EOD?}
    G --> H
    H -->|Yes| I[Close: action=0]
    H -->|No| J[Hold Position]
    I --> K[Flatten if EOD]
    style C fill:#ffcccc
    style E fill:#ccffcc
    style G fill:#ccffcc