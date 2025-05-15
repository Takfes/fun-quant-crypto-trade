import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pytrade.loaders import fetch_crypto_ohlcv_data
from pytrade.plotting import add_chart_to_figure, create_base_candlestick_chart

dfc = fetch_crypto_ohlcv_data(symbol="BTC/USDT", timeframe="1d", lookback_period="3m")

df = dfc.copy()
