
# Financial Feature Engineering

<br>
<br>

## MA Related Features
- Slope of the MA
- Fast, Medium, Slow MA
- Distance between candle and MA

## Divergence Features
- Divergence happens when price moves in one direction, but a technical indicator moves in the opposite direction.
- This often signals weakness in the trend — a possible reversal or slowdown.
- Used with RSI, MACD (Histogram), Stochastic Oscillator, Momentum or ROC
- has_bullish_divergence // Boolean flag // Can trigger long entry filters
- divergence_strength // RSI diff × price diff // Quantifies strength of divergence
- divergence_with_zone_overlap // Divergence + proximity to S/R or S/D // Enhances reliability
- divergence_recent // Has divergence occurred in last N bars? // Feature for ML windowed model
- divergence_count_window // # of divergences in last N candles // Captures choppiness or inflection potential

<br>

## Relative Position to Multiple MAs
- Is price above all three (Fast, Medium, Slow MA)? → Strong uptrend.
- Is price below all three? → Strong downtrend.

<br>

## ATR-Based Volatility Features (Average True Range)
- Measure current market turbulence.
- ATR measures market volatility — not direction, just how much price is moving.
- True Range (TR) = max(high - low, abs(high - previous close), abs(low - previous close))
- ATR = Exponential or Simple Moving Average of TR over N periods (e.g. 14)
- Normalized ATR (ATR divided by price) = % volatility.
- 14-period ATR vs 50-period ATR → Short-term volatility relative to long-term.
- High ATR → Breakouts more likely.
- Low ATR → Mean reversion more likely.

<br>

## Price Acceleration (2nd Derivative of MA)
- Take the second derivative of your MA.
- 1st Derivative: Slope of MA = diff(MA)
- 2nd Derivative: Acceleration = diff(diff(MA))

<br>

## Bollinger Band Width
- Distance between upper and lower Bollinger Bands.

<br>

## RSI Slope or RSI Divergence Detection
- Slope of RSI over N periods.
- Price making new highs while RSI making lower highs → Bearish divergence.
- Divergences often precede reversals or at least strong pullbacks.
- Normal RSI tells you overbought/oversold
- RSI Slope tells you momentum pressure
    ```python
    # Slope = change over time window
    df['ma_slope'] = df['ma'].diff(periods=5) / 5
    from scipy.stats import linregress
    def compute_slope(series):
        y = series.values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return slope
    df['slope'] = df['ma'].rolling(window=10).apply(compute_slope, raw=False)
    ```

<br>

## Volume-Weighted Moving Averages (VWMA)
- If VWMA > SMA → recent volume favors buyers.
- VWMA slope gives a more conviction-based trend signal.
- MA that weights prices based on volume — i.e., gives more importance to high-volume candles.
    ```python
    n = 20  # lookback
    price_volume = df['close'] * df['volume']
    vwma = price_volume.rolling(n).sum() / df['volume'].rolling(n).sum()
    df['vwma'] = vwma
    ```

<br>

## Candle Body Size vs Total Range
- Body Size = Close - Open.
- Total Range = High - Low.
- Ratio = Body Size / Total Range.
- Big bodies relative to range = strong conviction candles.
- Small bodies with long wicks = indecision, likely chop.

<br>

## Local Volatility Spike Detection
- Capture "volatility clusters" common before moves.
- Short-term rolling standard deviation of returns.

<br>

## Cumulative Return Over Lookback Windows.
- Summarize directional bias.
- Cumulative return over 10, 20, 50 periods.

<br>

## More Feature Ideas
- Skewness of recent returns : Detect bias toward sudden down moves vs slow grinds up (or vice versa).
- Kurtosis of returns : Detect "fat tails" risk — unstable markets.
- Days Since Last New High/Low : Measure breakout freshness.
- Rate of Change of Volume : Capture how fast participation is changing.
- BoS vs CHoCH and S/R vs S/D

<br>

## Function Transformer
```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Your feature function
def compute_log_return(X):
    return np.log(X / X.shift(1)).fillna(0)

# Wrap it
log_return_transformer = FunctionTransformer(compute_log_return)
```

<br>

---
---

## BoS - CHoCH - S/R - S/D
### 🔹 1. BoS (Break of Structure) vs CHoCH (Change of Character)

### 📌 Key Conceptual Difference

| Term | What It Means | When It Happens | Trading Implication |
|------|----------------|------------------|----------------------|
| **BoS** (Break of Structure) | Continuation of current trend | Breaks prior swing high (uptrend) or low (downtrend) | Trend-following entry trigger |
| **CHoCH** (Change of Character) | Possible reversal of trend | First time price breaks the **opposite** structure level | Early signal of trend regime change |

---

### 🧠 Example:

- Price makes **HH → HL → HH → HL**, then:
  - If price breaks the latest **HH** → **BoS** (uptrend continuation).
  - If price breaks the last **HL** → **CHoCH** (reversal warning → possible downtrend).

---

### 🛠️ How to Detect Primary BoS/CHoCH Features

Step-by-step (pseudocode style):

```python
# Parameters
swing_lookback = 5  # how many candles to look back to define local highs/lows

# Detect swings
swing_highs = df['high'][::-1].rolling(swing_lookback).max()[::-1]
swing_lows = df['low'][::-1].rolling(swing_lookback).min()[::-1]

# Maintain structure history (you'd store last confirmed HH and HL)

# BoS (continuation):
bos_long = df['close'] > swing_highs.shift(1)  # breaking previous high
bos_short = df['close'] < swing_lows.shift(1)  # breaking previous low

# CHoCH (reversal):
choch_long = df['close'] > last_lower_high  # break of prior lower high
choch_short = df['close'] < last_higher_low  # break of prior higher low
```

Note: You may want to define `last_higher_low` and `last_lower_high` manually using swing tracking logic.

---

### 🌟 Derived Features from BoS and CHoCH

| Feature Name | How to Compute | Why It’s Useful |
|--------------|----------------|-----------------|
| `bos_count_lookback_n` | Number of BoS in past N bars | Measures trend *momentum* via structure confirmation |
| `choch_recent` | 1 if CHoCH occurred within last N bars | Flags reversal zones |
| `bos_confirmed_by_volume` | BoS + above-average volume | Filters valid structure breaks |
| `bos_distance` | Price difference between break and previous swing | Captures breakout strength |
| `bos_to_choch_ratio` | BoS count / CHoCH count over window | Measures trend stability vs instability |

---

### 🔹 2. Supply/Resistance vs Supply/Demand

### 📌 Conceptual Clarification

| Term | Meaning | Based On | Key Insight |
|------|--------|----------|-------------|
| **Support/Resistance** | Horizontal price levels where reversals occurred | Previous highs/lows, range edges | Reflects **price memory** (retail + algo anchored) |
| **Supply/Demand Zones** | Zones where price **left quickly**, suggesting strong buying/selling | Candlestick clusters, imbalance zones | Reflects **institutional order flow** (Smart Money behavior) |

---

### 🧠 Example:

- **Resistance**: Multiple past price rejections at 21500.
- **Supply Zone**: Price spent 3 candles consolidating at 21700, then fell aggressively — suggests **unfilled sell orders** exist there.

---

### 🛠 How to Detect Primary Features

#### A. **Support/Resistance**
- Use **local max/min over N periods** (same as swing high/low detection)
- Store **recent levels with multiple touches**

```python
sr_highs = df['high'][::-1].rolling(window=N).max()[::-1]
sr_lows = df['low'][::-1].rolling(window=N).min()[::-1]
```

#### B. **Supply/Demand Zones**
- Detect **consolidation (tight-range clusters)** followed by **impulsive move**
- Save:
  - **Base candles** (zone)
  - **Departure direction** (up/down)
  - **Zone width**, **volume** in base
  - **Departure candle size**

```python
# Basic supply zone detection (pseudo-code logic)
consolidation = (df['high'] - df['low']).rolling(window=3).mean() < threshold
impulse = (df['close'] - df['open']).abs() > 1.5 * rolling_atr
zone_detected = consolidation.shift(1) & impulse
```

---

### 🌟 Derived Features from S/R and Supply/Demand

| Feature Name | How to Compute | Why It’s Useful |
|--------------|----------------|-----------------|
| `distance_to_sr` | Abs(price - nearest support or resistance) | Mean-reversion or breakout readiness |
| `zone_retest_success` | Whether price retested zone and bounced | Confirm validity of zone |
| `zone_strength_score` | Weight of volume + rejections + times tested | Quantifies importance of level |
| `current_in_zone` | Boolean if price is inside zone | Alerts “do not trade” or prepare for breakout |
| `zone_magnetism` | Price visits per bar near zone center | Shows institutional anchoring |
| `zone_freshness` | Time since last visit to the zone | Older zones weaken over time |
| `zone_type_transition` | Demand → supply or vice versa | Captures polarity shift (structure flip) |

---

### 🧠 Bonus Strategy: Combine Structure + Zone Logic

Use **CHoCH inside demand zone** = high probability long setup.
Use **BoS + break from supply** = continuation short setup.

Add filters:
- **High volume** at zone = more conviction.
- **Volatility squeeze** = more explosive breakout.
- **Wick rejections** = confirm rejection intention.
