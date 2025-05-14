# Multiple Time Frames (MTF)

## 🧠 Multi-Timeframe Features

### 🔹 High-Level Principles

- **Use Higher Timeframes (HTF)** for:
  - Trend context
  - Volatility regime
  - Supply/Demand zones
- **Use Lower Timeframes (LTF)** for:
  - Entry triggers
  - Rejections, breakouts, execution logic
- **Combine timeframes** by:
  - Computing features separately
  - Merging (aligning) them into your base timeframe (usually the LTF)

---

### 🔁 Integration Options

| Approach | Use Case |
|----------|----------|
| HTF → LTF | Most common. Create slow-moving context features (trend, zone) on HTF, inject into LTF model. |
| LTF → HTF | Aggregate short-term behavior (rejections, spikes) into HTF metrics (e.g., "3 CHoCHs last 1H"). |
| Cross-Timeframe Interaction | Build meta-features like `rsi_15m / rsi_1h` or `price / ma_4h` to detect alignment/divergence. |

---

## ✅ Practical Example (15m Model with 1h and 4h Context)

| Feature | Timeframe | Purpose |
|--------|-----------|---------|
| `trend_label_1h` | 1h | Only trade long if 1h trend is up |
| `zone_retest_4h` | 4h | Institutional demand confirmation |
| `rsi_15m` | 15m | Entry trigger (oversold bounce) |
| `atr_15m` | 15m | Risk adjustment |
| `rsi_ratio_15m_to_1h` | Cross-TF | Momentum alignment |

---

## 🔧 Code Snippets

### 🧪 Resample + Compute HTF Features

```python
# Code chunk
df_1h = df.resample('1H').agg({
    'open': 'first', 'high': 'max',
    'low': 'min', 'close': 'last', 'volume': 'sum'
})
df_1h['trend_label'] = compute_trend_regime(df_1h)
```

```python
# Function
def resample_to_higher_timeframe(df, timeframe='1H'):
    """
    Resample OHLCV data to a higher timeframe.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        timeframe (str): Target timeframe (e.g., '1H', '4H').

    Returns:
        pd.DataFrame: Resampled dataframe with computed features.
    """
    df_htf = df.resample(timeframe).agg({
        'open': 'first', 'high': 'max',
        'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    df_htf['trend_label'] = compute_trend_regime(df_htf)
    return df_htf
```

```python
# Convert Function into a Transformer Class
from sklearn.base import BaseEstimator, TransformerMixin

class ResampleTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer to resample OHLCV data
    and compute higher timeframe features.
    """
    def __init__(self, timeframe='1H'):
        self.timeframe = timeframe

    def fit(self, X, y=None):
        # No fitting required for resampling
        return self

    def transform(self, X):
        """
        Transform input dataframe by resampling and computing features.

        Args:
            X (pd.DataFrame): Input dataframe with OHLCV data.

        Returns:
            pd.DataFrame: Resampled dataframe with computed features.
        """
        return resample_to_higher_timeframe(X, self.timeframe)
```

```python
# Function Transformer
from sklearn.preprocessing import FunctionTransformer

# Define the transformation function
def resample_transformer(X, timeframe='1H'):
    return resample_to_higher_timeframe(X, timeframe)

# Create a FunctionTransformer
resample_function_transformer = FunctionTransformer(
    func=lambda X: resample_transformer(X, timeframe='1H')
)
```

```python
# Partial Function
from sklearn.preprocessing import FunctionTransformer
from functools import partial

# Create a partial function with a fixed timeframe
resample_partial = partial(resample_to_higher_timeframe, timeframe='1H')

# Wrap it in a FunctionTransformer
resample_function_transformer = FunctionTransformer(func=resample_partial)
```

### 🔗 Merge HTF into LTF

```python
# Forward-fill HTF features into base timeframe
df['trend_label_1h'] = df_1h['trend_label'].reindex(df.index, method='ffill')
```

```python
# Merge HTF features into LTF
def merge_htf_to_ltf(df_ltf, df_htf, method='ffill'):
    """
    Merge higher timeframe (HTF) features into lower timeframe (LTF) data.

    Args:
        df_ltf (pd.DataFrame): Lower timeframe dataframe.
        df_htf (pd.DataFrame): Higher timeframe dataframe with features.
        method (str): Method to align HTF features to LTF (e.g., 'ffill').

    Returns:
        pd.DataFrame: LTF dataframe enriched with HTF features.
    """
    df_merged = df_ltf.copy()
    for col in df_htf.columns:
        df_merged[col] = df_htf[col].reindex(df_ltf.index, method=method)
    return df_merged

# Aggregate LTF features into HTF
def aggregate_ltf_to_htf(df_ltf, timeframe='1H', agg_funcs=None):
    """
    Aggregate lower timeframe (LTF) features into higher timeframe (HTF).

    Args:
        df_ltf (pd.DataFrame): Lower timeframe dataframe.
        timeframe (str): Target higher timeframe (e.g., '1H', '4H').
        agg_funcs (dict): Aggregation functions for each column.

    Returns:
        pd.DataFrame: Higher timeframe dataframe with aggregated features.
    """
    if agg_funcs is None:
        agg_funcs = {
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }
    df_htf = df_ltf.resample(timeframe).agg(agg_funcs)
    return df_htf
```


---

## 🏭 Production Setup Strategy

### 📂 Options:
1. **Single resampled source**
   - Resample the base OHLCV into multiple TFs in code
   - Pros: Simple, no external sync issues
   - Cons: More compute

2. **Multiple data sources**
   - Store precomputed HTF features or load from database/API
   - Join on timestamp into LTF
   - Pros: Scalable, modular
   - Cons: Needs careful timestamp alignment

---

## 📁 Recommended Setup

```
data/
├── base_loader.py          # Load 1m/5m OHLCV
├── resampler.py            # Convert to 1h, 4h
├── feature_merger.py       # Merge HTF into LTF
```
