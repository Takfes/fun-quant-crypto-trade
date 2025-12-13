"""Utility functions for data retrieval and statistical calculations.

This module provides functions to fetch market data and compute
statistical metrics for portfolio analysis.
"""

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


def get_ticker_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
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


def get_sim_daily() -> pd.DataFrame:
    """
    Fetch Single-Index-Model daily data from Ken French's data library.
    Returns:
        pd.DataFrame: Daily Single-Index-Model data with date index.
    """
    # Fama-French-3 Factor Model Daily Dataset
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    df = pd.read_csv(
        url,
        compression="zip",
        header="infer",
        skiprows=4,
        skipfooter=2,
        parse_dates=[0],
        index_col=0,
        engine="python",
    ).rename(columns={"Mkt-RF": "eMKT"})[["eMKT", "RF"]]
    return df / 100


def get_ff3_daily() -> pd.DataFrame:
    """
    Fetch Fama-French 3-factor daily data from Ken French's data library.
    Returns:
        pd.DataFrame: Daily Fama-French 3-factor data with date index.
    """
    # Fama-French-3 Factor Model Daily Dataset
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    df = pd.read_csv(
        url,
        compression="zip",
        header="infer",
        skiprows=4,
        skipfooter=2,
        parse_dates=[0],
        index_col=0,
        engine="python",
    ).rename(columns={"Mkt-RF": "eMKT"})
    return df / 100


def get_ch4_daily() -> pd.DataFrame:
    """
    Fetch Carhart 4-factor daily data from Ken French's data library.
    Returns:
        pd.DataFrame: Daily Carhart 4-factor data with date index.
    """
    # Fama-French-3 Factor Model Daily Dataset
    url_ff3 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    # Carhart-4 Factor Model Daily Dataset
    url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    ff3 = pd.read_csv(
        url_ff3,
        compression="zip",
        header="infer",
        skiprows=4,
        skipfooter=2,
        parse_dates=[0],
        index_col=0,
        engine="python",
    )
    mom = pd.read_csv(
        url_mom,
        compression="zip",
        header="infer",
        skiprows=13,
        skipfooter=2,
        parse_dates=[0],
        index_col=0,
        engine="python",
    )
    df = ff3.join(mom, how="inner")
    df = df[["Mkt-RF", "SMB", "HML", "Mom", "RF"]].rename(columns={"Mkt-RF": "eMKT", "Mom": "MOM"})
    return df / 100


def get_ff5_daily() -> pd.DataFrame:
    """
    Fetch Fama-French 5-factor daily data from Ken French's data library.
    Returns:
        pd.DataFrame: Daily Fama-French 5-factor data with date index.
    """
    # Fama-French-5 Factor Model Daily Dataset
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    df = pd.read_csv(
        url,
        compression="zip",
        header="infer",
        skiprows=4,
        skipfooter=2,
        parse_dates=[0],
        index_col=0,
        engine="python",
    ).rename(columns={"Mkt-RF": "eMKT"})
    return df / 100
