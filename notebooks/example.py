import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from ta import add_all_ta_features

from pytrade.loaders import fetch_crypto_ohlcv_data

raw = fetch_crypto_ohlcv_data(symbol="BTC/USDT", timeframe="1d", lookback_period="3m")

# # OUT-OF-THE-BOX
# xxx = add_all_ta_features(raw, open="open", high="high", low="low", close="close", volume="volume")
