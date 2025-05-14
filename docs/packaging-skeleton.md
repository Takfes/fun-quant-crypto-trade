# Quants Packaging Guidelines

✅ **include both functions *and* transformers.**

---

#### 🔹 build each feature as a **pure function**:
```python
def compute_rsi(df, period=14):
    ...
    return pd.Series(...)  # or add column to df
```

🔁 **wrap it** into a **pipeline-compatible transformer**:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class RSIFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, period=14):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rsi = compute_rsi(X, self.period)
        return rsi.to_frame(name=f"rsi_{self.period}")
```

🔁 **prebuild composite transformers** later, e.g.:
```python
class TrendFeatureSet(BaseEstimator, TransformerMixin):
    def transform(self, X):
        return pd.concat([
            compute_slope(X),
            compute_adx(X),
            compute_trend_label(X),
        ], axis=1)
```

---

#### 🚀 Bonus: Auto-Wrapping Utility
build a utility that **turns any function into a transformer**:

```python
def function_to_transformer(fn, name=None, **kwargs):
    class WrappedTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return fn(X, **kwargs)
    return WrappedTransformer()
```

---

## Example structure, categorization

```python
my_trading_features/
│
├── features/
│   ├── trend_regimes.py
│   │   ├── detect_ma_slope()
│   │   ├── compute_adx()
│   │   ├── label_trend_regime()
│   │   ├── compute_cumulative_return()
│   │   ├── trend_strength_oscillator()
│   │   └── price_vs_ma_bias()
│
│   ├── volatility_regimes.py
│   │   ├── compute_atr()
│   │   ├── compute_bb_width()
│   │   ├── label_volatility_regime()
│   │   ├── atr_zscore()
│   │   └── detect_volatility_squeeze()
│
│   ├── price_structure.py
│   │   ├── detect_swing_highs_lows()
│   │   ├── detect_bos()
│   │   ├── detect_choch()
│   │   ├── rejection_counter_near_sr()
│   │   └── polarity_shift_detector()
│
│   ├── supply_demand_zones.py
│   │   ├── detect_base_zone()
│   │   ├── detect_impulse_away()
│   │   ├── label_supply_demand_zones()
│   │   ├── zone_freshness()
│   │   ├── zone_magnetism_score()
│   │   └── retest_success_tracker()
│
│   ├── volume_features.py
│   │   ├── compute_vwma()
│   │   ├── detect_volume_spikes()
│   │   ├── compute_poc()                     ← (if using volume profile)
│   │   ├── volume_density_around_price()
│   │   └── volume_std_dev_ratio()
│
│   ├── divergence_features.py
│   │   ├── detect_rsi_divergence()
│   │   ├── detect_macd_divergence()
│   │   ├── detect_hidden_divergence()
│   │   ├── divergence_strength_score()
│   │   └── label_divergence_zone_overlap()
│
│   ├── microstructure_features.py
│   │   ├── get_bid_ask_spread()
│   │   ├── compute_orderbook_imbalance()
│   │   ├── track_quote_change_rate()
│   │   ├── compute_mid_price()
│   │   └── tick_to_trade_ratio()
│
│   ├── sentiment_features.py     (optional)
│   │   ├── compute_twitter_sentiment()
│   │   ├── compute_news_sentiment()
│   │   ├── funding_rate_bias()
│   │   └── social_volume_intensity()
```

## 🔧 Recommended Folder Structure

```python
my_trading_features/
│
├── __init__.py
│
├── data/
│   ├── loaders.py                ← load OHLCV, tick, L2, sentiment data
│   ├── preprocessing.py          ← missing values, resampling, deduplication
│
├── features/
│
│   ├── trend_regimes/
│   │   ├── functions.py          ← compute_ma_slope(), compute_adx(), etc.
│   │   ├── transformers.py       ← RSISlopeTransformer, TrendLabelTransformer
│   │   ├── group.py              ← TrendFeatureGroup (returns all trend features)
│   │   └── __init__.py
│
│   ├── volatility_regimes/
│   │   ├── functions.py          ← compute_atr(), compute_bb_width()
│   │   ├── transformers.py       ← ATRTransformer, BBWidthTransformer
│   │   ├── group.py              ← VolatilityFeatureGroup
│   │   └── __init__.py
│
│   ├── price_structure/
│   │   ├── functions.py          ← detect_swing_points(), detect_bos(), etc.
│   │   ├── transformers.py       ← BoSTransformer, CHoCHTransformer
│   │   ├── group.py              ← StructureFeatureGroup
│   │   └── __init__.py
│
│   ├── supply_demand_zones/
│   │   ├── functions.py          ← detect_zones(), base_impulse_logic(), etc.
│   │   ├── transformers.py       ← SupplyZoneTransformer, DemandRetestTransformer
│   │   ├── group.py              ← SupplyDemandFeatureGroup
│   │   └── __init__.py
│
│   ├── volume_features/
│   │   ├── functions.py          ← compute_vwma(), detect_volume_spike()
│   │   ├── transformers.py       ← VWMA_Transformer, POCTransformer
│   │   ├── group.py              ← VolumeFeatureGroup
│   │   └── __init__.py
│
│   ├── divergence_features/
│   │   ├── functions.py          ← detect_rsi_divergence(), divergence_strength()
│   │   ├── transformers.py       ← RSIDivergenceTransformer
│   │   ├── group.py              ← DivergenceFeatureGroup
│   │   └── __init__.py
│
│   ├── microstructure_features/
│   │   ├── functions.py          ← compute_bid_ask_spread(), imbalance()
│   │   ├── transformers.py       ← BidAskSpreadTransformer, OrderFlowTransformer
│   │   ├── group.py              ← MicrostructureFeatureGroup
│   │   └── __init__.py
│
│   ├── sentiment_features/       ← Optional if you ingest NLP or news
│   │   ├── functions.py
│   │   ├── transformers.py
│   │   ├── group.py
│   │   └── __init__.py
│
│   ├── global_feature_factory.py ← Assembles features from all groups
│
├── engineering/
│   ├── feature_pipeline.py       ← Combines transformers into full scikit-learn pipeline
│   ├── auto_wrapper.py           ← function_to_transformer(fn)
│   ├── base_transformer.py       ← Base class for your custom transformers
│   ├── cross_feature_engineering.py ← Interactions like rsi × atr, etc.
│
├── labels/
│   ├── triple_barrier.py
│   ├── event_labels.py           ← CHoCH, BoS, zone touch labels
│
├── utils/
│   ├── math_utils.py             ← zscore, slope, std, normalize
│   ├── plotting.py               ← Overlay features on price charts
│   ├── rolling.py                ← Rolling max/min, rolling regression
│
├── config/
│   ├── feature_configs.py        ← Default parameters, e.g. RSI period, ATR length
│   ├── constants.py              ← Thresholds, lookbacks, regime cutoffs
│
├── tests/
│   ├── test_trend_regimes.py
│   ├── test_volume_features.py
│   ├── ...
│
├── notebooks/
│   ├── feature_visual_debug.ipynb
│   ├── full_feature_matrix_demo.ipynb
│
└── README.md
```
