import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def get_time_series_daily(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={os.getenv('API_KEY_ALPHAVANTAGE')}"
    response = requests.get(url)
    data = response.json()
    return data


def get_etf_profile(symbol):
    url = f"https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={symbol}&apikey={os.getenv('API_KEY_ALPHAVANTAGE')}"
    response = requests.get(url)
    data = response.json()
    return data


etf_profile = get_etf_profile("IEFA")
holdings = pd.DataFrame(etf_profile.get("holdings"))


import itertools

import numpy as np

N = 3

for k in range(2, N + 1):
    print(k)
    for i, j in itertools.combinations(range(N), k):
        print(i, j)
        w = np.zeros(N)
        w[i] = 0.5
        w[j] = 0.5
        print(w)
