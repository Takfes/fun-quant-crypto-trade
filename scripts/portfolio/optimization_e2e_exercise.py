import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

START = "2020-01-01"
END = "2025-11-30"
RISK_FREE_RATE = 0.0417
MIN_ALLOCATION_WEIGHT = 0.0
MAX_ALLOCATION_WEIGHT = 0.5
NUM_PORTFOLIOS = 10_000

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options=headers)[0]

tickers = [
    "AAPL",
    "AMGN",
    "CAT",
    "FDX",
    "GE",
    "GS",
    "INTC",
    "IRM",
    "JNJ",
    "JPM",
    "LMT",
    "MRK",
    "MRNA",
    "MSFT",
    "MTCH",
    "PLG",
    "RTX",
    "WELL",
    "WMB",
    "XOM",
]

rawdata = yf.download(tickers, start=START, end=END)
closedata = rawdata["Close"].copy()
returnsdata = closedata.pct_change().dropna()

statsdata = pd.DataFrame({
    "min": returnsdata.min(),
    "q99": returnsdata.quantile(0.01),
    "q95": returnsdata.quantile(0.05),
    "mean": returnsdata.mean(),
    "median": returnsdata.median(),
    "max": returnsdata.max(),
    "std": returnsdata.std(),
    "skew": returnsdata.skew(),
    "kurtosis": returnsdata.kurtosis(),
    "annual_returns": returnsdata.mean() * 252,
    "annual_std": returnsdata.std() * np.sqrt(252),
    "coefficient_of_variation": returnsdata.std() * np.sqrt(252) / (returnsdata.mean() * 252),
}).T

covmatrix = returnsdata.cov()
cormatrix = returnsdata.corr()

# Generate random portfolios
weights_equal = np.ones(len(tickers)) / len(tickers)
weights_dirichlet = np.random.dirichlet(np.ones(len(tickers)), size=NUM_PORTFOLIOS)
weights_power = np.random.rand(NUM_PORTFOLIOS, len(tickers))
weights_power /= weights_power.sum(axis=1)[:, np.newaxis]
weights = np.vstack([weights_equal, weights_dirichlet, weights_power])

# Calculate annualized portfolio returns & risk for all portfolios
portfolio_returns = weights @ statsdata.loc["annual_returns"].values
portfolio_variance = (weights @ covmatrix * weights).sum(axis=1)
portfolio_risk = (np.sqrt(portfolio_variance) * np.sqrt(252)).values

# Plot the portfolio risk-return scatter
plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns / portfolio_risk, marker="o")

# np.sqrt((weights_equal @ covmatrix) @ weights_equal.T) * np.sqrt(252)
