# Trend or Regime Identification

> “What kind of market am I in: uptrend, downtrend, or chop?”

Getting this right improves **signal filtering**, **risk allocation**, and **entry/exit timing**. Let’s walk through **how to detect and capture** trend regimes systematically.

---

## 🧠 What Is a Trend Regime?

A **trend regime** is a label or numeric score that reflects the market’s current **macro behavior**:

- 📈 Uptrend
- 📉 Downtrend
- 🔁 Sideways (Range/Consolidation)
- 🧯 Volatile vs Stable (Volatility Regime)
- 💪 Weak vs Strong (Trend Strength)

You can capture these as:
- Categorical labels: `"up"`, `"down"`, `"range"`
- Numeric indicators: slope, trend strength, volatility

---

## 🔧 How to Detect Trend Regimes (Core Methods)

---

### 🔹 1. **Slope of Moving Average (or Price)**

```python
window = 20
ma = df['close'].rolling(window).mean()
df['trend_slope'] = ma.diff() / ma.shift(1)
```

- Positive slope → uptrend
- Negative slope → downtrend
- Near zero → chop

🔁 Can convert into regime:
```python
df['trend_regime'] = np.select(
    [df['trend_slope'] > 0.001, df['trend_slope'] < -0.001],
    ['up', 'down'],
    default='range'
)
```

---

### 🔹 2. **ADX (Average Directional Index)**

```python
import ta
df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
```

- **ADX > 20 or 25** → trending
- **ADX < 20** → choppy
- Combine with MA slope or price structure to determine trend direction

---

### 🔹 3. **Cumulative Return / Price Change Over N**

```python
df['trend_strength'] = df['close'].pct_change(periods=20)
```

- Large positive → strong uptrend
- Large negative → strong downtrend
- Close to zero → ranging

🔁 Convert to regime with thresholds

---

### 🔹 4. **Price vs Long-Term MA**

```python
ma200 = df['close'].rolling(200).mean()
df['above_ma200'] = df['close'] > ma200
```

- Above = bullish bias
- Below = bearish bias
- Near = neutral bias

🧠 Combine with short-term trend logic to define higher time frame filter.

---

### 🔹 5. **Trend Oscillator (Custom)**

You can build your own:

```python
# Momentum + direction + volatility normalization
trend_osc = (ma.diff() / df['close'].rolling(window).std()) * 100
df['trend_osc'] = trend_osc
```

---

## 🧯 Volatility Regime Detection

Use ATR, standard deviation, or Bollinger Band width:

```python
atr = df['high'] - df['low']  # or use ta.volatility.AverageTrueRange
df['volatility_regime'] = pd.qcut(atr, 3, labels=['low', 'medium', 'high'])
```

---

## 💡 Combine Trend + Volatility for Full Regime

| Trend | Volatility | Regime |
|-------|------------|--------|
| Up    | Low        | "grind-up" |
| Up    | High       | "breakout-up" |
| Down  | Low        | "grind-down" |
| Down  | High       | "panic-sell" |
| Range | Low        | "chop" |
| Range | High       | "volatile chop" |

---

## 🧠 Additional Trend Regime Feature Ideas

| Feature | Description | Why It's Useful |
|--------|-------------|-----------------|
| `trend_duration_bars` | How long the current regime has lasted | Tracks aging trends |
| `trend_slope_zscore` | Normalize slope vs historical std | Detects abnormally strong moves |
| `trend_slope_diff` | Change in slope over time | Acceleration/deceleration detection |
| `trend_regime_onehot` | One-hot encode the regime label | ML-compatible categorical input |
| `cross_tf_regime_conflict` | Compare H1 vs H4 trend regime | Filters false signals |

---

## 🧪 Labeling Regimes for Supervised ML

You can also **pre-label regimes** for training a classifier or classifier-switching model:
- Rule-based labeler (e.g. 5-bar structure + ADX + volatility filter)
- Then train ML models on specific regimes

---

## ✅ TL;DR: Best Trend Regime Capture Stack

| Technique | Captures |
|----------|----------|
| Slope of MA or Price | Basic directional bias |
| ADX | Trend presence/strength |
| MA crossover (short vs long) | Regime transitions |
| Price vs long-term MA | Higher timeframe bias |
| Volatility via ATR or BBs | Volatility regime |
| Composite labels | Combined trend + volatility regimes |
