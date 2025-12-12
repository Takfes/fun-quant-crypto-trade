"""Utility functions for data retrieval and statistical calculations.

This module provides functions to fetch market data and compute
statistical metrics for portfolio analysis.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def get_sp500_tickers() -> pd.DataFrame:
    """
    Fetch S&P 500 tickers from Wikipedia.

    Returns:
        DataFrame with S&P 500 company information.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options=headers)[0]
    return sp500


def get_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download historical price data and compute returns.

    Args:
        tickers: List of ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        DataFrame of daily returns.
    """
    rawdata = yf.download(tickers, start=start, end=end)
    closedata = rawdata["Close"].copy()
    returnsdata = closedata.pct_change().dropna()
    return returnsdata


def compute_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for returns data.

    Args:
        returns: DataFrame of asset returns.

    Returns:
        Transposed DataFrame with descriptive statistics.
    """
    # Calculate VaR values for CVaR computation
    var99 = returns.quantile(0.01)
    var95 = returns.quantile(0.05)

    # Reason: CVaR is the expected value of returns below the VaR threshold
    cvar99 = returns[returns <= var99].mean()
    cvar95 = returns[returns <= var95].mean()

    return pd.DataFrame({
        "min": returns.min(),
        "var99": var99,
        "cvar99": cvar99,
        "var95": var95,
        "cvar95": cvar95,
        "mean": returns.mean(),
        "median": returns.median(),
        "max": returns.max(),
        "var": returns.var(),
        "std": returns.std(),
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "annual_returns": returns.mean() * 252,
        "annual_std": returns.std() * np.sqrt(252),
        "coefficient_of_variation": returns.std() * np.sqrt(252) / (returns.mean() * 252),
    }).T
