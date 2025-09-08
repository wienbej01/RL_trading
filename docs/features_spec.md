# Intraday US Equities Feature Specification

## Overview
Parsimonious set of 8 leakage-free features for RL state space, computed at bar_close using only past data (lookback 120 bars ~2h). Economic rationale: Capture momentum persistence (RSI/MACD), liquidity/microstructure imbalances (vol_imbalance/FVG), intraday patterns (tod), regime/volatility (VIX/RVOL), structure (swing_dist) to predict 1-min returns without overfitting.

## Selected Features
1. **RSI(14)**: Relative Strength Index over 14 bars. Rationale: Measures overbought/oversold for mean-reversion alpha in intraday ranges.
2. **MACD(12,26,9)**: Moving Average Convergence Divergence. Rationale: Trend momentum signal for persistence in short-term drifts.
3. **Vol Imbalance**: (buy_volume - sell_volume) / total_volume (proxied via tick rule). Rationale: Order flow imbalance predicts near-term price direction via liquidity provision.
4. **ToD Sin/Cos**: Sinusoidal encoding of time-of-day (hour*60 + minute). Rationale: Captures intraday seasonality (e.g., open volatility, lunch lull) for pattern alpha.
5. **VIX Level**: Spot VIX value. Rationale: Volatility regime proxy; high VIX signals risk-off, adjusts expectations for larger moves.
6. **RVOL**: Relative volume vs 20-day avg. Rationale: Activity surge indicates news/interest, precedes breakouts.
7. **FVG Dist**: Distance to nearest Fair Value Gap (unfilled OHLC gap). Rationale: Price inefficiency attracts fills, mean-reversion to gaps.
8. **Swing Dist**: Normalized distance to last swing high/low. Rationale: Structure breaks signal trend changes, proximity to levels for support/resistance.

## Parsimony & Guardrails
- Total: 8 features (dim-reduced via PCA if VIF>5).
- No leakage: All computed using data up to bar_close; embargo next bar.
- Normalization: Z-score per feature, clip [-3,3].

## Ablations Planned
- **Univariate IC**: Pearson corr with 1-bar forward return >0.05 threshold; drop if <0.01.
- **Multicollinearity**: VIF <5; orthogonalize via Gram-Schmidt if violated.
- **SHAP Importance**: Post-training attribution; retain top-8 by avg |SHAP|; ablate subsets (e.g., remove microstructure: expect IC drop 0.02).
- **Regime Split**: Test IC in low/high VIX; ensure stability (std<0.01).

## Code Sketch
```python
import ta  # or custom indicators
df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
df['macd'] = ta.trend.MACD(df['close']).macd()
df['vol_imbalance'] = (df['buy_vol'] - df['sell_vol']) / df['volume']  # proxy
df['tod_sin'] = np.sin(2 * np.pi * (df.index.hour * 60 + df.index.minute) / 1440)
df['tod_cos'] = np.cos(2 * np.pi * (df.index.hour * 60 + df.index.minute) / 1440)
df['vix'] = vix_df.reindex(df.index, method='ffill')['vix']
df['rvol'] = df['volume'] / df['volume'].rolling(390*20).mean()  # 20-day avg
df['fvg_dist'] = compute_fvg_distance(df)  # custom: min dist to unfilled gaps
df['swing_dist'] = (df['close'] - df['last_swing_low']) / df['atr']  # normalized
features = df[['rsi', 'macd', 'vol_imbalance', 'tod_sin', 'tod_cos', 'vix', 'rvol', 'fvg_dist', 'swing_dist']].shift(1).dropna()  # embargo
```

## Mermaid Pipeline Flow
```mermaid
graph TD
    A[OHLCV Data] --> B[Technical: RSI/MACD/ATR]
    A --> C[Micro: Vol Imbalance/FVG]
    A --> D[Time: ToD Sin/Cos]
    E[VIX Data] --> F[Regime: VIX/RVOL]
    A --> G[Structure: Swing Dist]
    B --> H[Normalize & Z-Score]
    C --> H
    D --> H
    F --> H
    G --> H
    H --> I[Feature Vector for RL State]
    style I fill:#90EE90