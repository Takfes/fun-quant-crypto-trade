import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import argrelextrema, find_peaks, peak_prominences

from pytrade.loaders import fetch_crypto_ohlcv_data
from pytrade.plotting import add_chart_to_figure, create_base_candlestick_chart

dfc = fetch_crypto_ohlcv_data(symbol="BTC/USDT", timeframe="1d", lookback_period="3m")

df = dfc.copy()

# Example: Using the 'close' column for analysis
data = dfc["close"].values

# =============================================================
# 1. Testing argrelextrema
# =============================================================

order = 5
local_maxima_indices = argrelextrema(data, np.greater, order=order)[0]
local_minima_indices = argrelextrema(data, np.less, order=order)[0]

df["local_maxima"] = df["high"].iloc[local_maxima_indices]
df["local_minima"] = df["low"].iloc[local_minima_indices]

fig1 = create_base_candlestick_chart(
    df, title=f"local_maxima:{len(local_maxima_indices)}, local_minima:{len(local_minima_indices)}"
)
fig1 = add_chart_to_figure(
    fig1,
    go.Scatter(
        x=df.index, y=df["local_maxima"], mode="markers", name="Local Maxima", marker={"color": "green", "size": 10}
    ),
)

fig1 = add_chart_to_figure(
    fig1,
    go.Scatter(
        x=df.index, y=df["local_minima"], mode="markers", name="Local Minima", marker={"color": "red", "size": 10}
    ),
)
fig1.show()

# =============================================================
# 2. Testing find_peaks
# =============================================================

change_heuristic = np.percentile(df["close"].diff(1).abs().dropna(), 75)
distance_heuristic = int(df.shape[0] * 0.05)

distance = distance_heuristic
threshold = None
prominence = change_heuristic
width = None
wlen = None
plateau_size = None

peaks_indices, peak_info = find_peaks(
    data,
    prominence=prominence,
    width=width,
    wlen=wlen,
    threshold=threshold,
    plateau_size=plateau_size,
    distance=distance,
)
dips_indices, dip_info = find_peaks(
    -data,
    prominence=prominence,
    width=width,
    wlen=wlen,
    threshold=threshold,
    plateau_size=plateau_size,
    distance=distance,
)

df["local_peaks"] = df["high"].iloc[peaks_indices]
df["local_dips"] = df["low"].iloc[dips_indices]

fig2 = create_base_candlestick_chart(df, title=f"peaks:{len(peaks_indices)}, dips:{len(dips_indices)}")
fig2 = add_chart_to_figure(
    fig2,
    go.Scatter(
        x=df.index, y=df["local_peaks"], mode="markers", name="local_peaks", marker={"color": "green", "size": 10}
    ),
)
fig2 = add_chart_to_figure(
    fig2,
    go.Scatter(x=df.index, y=df["local_dips"], mode="markers", name="local_dips", marker={"color": "red", "size": 10}),
)
fig2.show()


# 3. Testing peak_prominences
prominences_peaks = peak_prominences(data, peaks_indices)[0]
prominences_dips = peak_prominences(-data, dips_indices)[0]

# print(f"{prominences_peaks=}")
print(f"{np.mean(prominences_peaks):=.2f}")
# print(f"{prominences_dips=}")
print(f"{np.mean(prominences_dips):=.2f}")
