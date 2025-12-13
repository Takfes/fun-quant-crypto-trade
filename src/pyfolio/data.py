"""Utility functions for data retrieval and statistical calculations.

This module provides functions to fetch market data and compute
statistical metrics for portfolio analysis.
"""

import pandas as pd
import yfinance as yf

# S&P 500 Sector Indices
SECTOR_INDICES = {
    "Energy": "^SP500-10",
    "Materials": "^SP500-15",
    "Industrials": "^SP500-20",
    "Consumer Discretionary": "^SP500-25",
    "Consumer Staples": "^SP500-30",
    "Health Care": "^SP500-35",
    "Financials": "^SP500-40",
    "Information Technology": "^SP500-45",
    "Communication Services": "^SP500-50",
    "Utilities": "^SP500-55",
    "Real Estate": "^SP500-60",
}

# S&P 500 Sector ETFs
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Energy": "XLE",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


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


def get_sector_indices_returns(start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical returns for S&P 500 Sector Indices.

    Args:
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        DataFrame of daily returns with sector names as columns.
    """
    tickers = list(SECTOR_INDICES.values())
    # Invert dictionary to map tickers back to sector names
    ticker_to_sector = {v: k for k, v in SECTOR_INDICES.items()}

    rawdata = yf.download(tickers, start=start, end=end, progress=False)

    # Handle case where only one ticker is returned (series vs dataframe)
    if len(tickers) == 1:
        closedata = rawdata["Close"].to_frame()
        closedata.columns = tickers
    else:
        closedata = rawdata["Close"].copy()

    # Rename columns from tickers to sector names
    closedata.rename(columns=ticker_to_sector, inplace=True)
    closedata.dropna(inplace=True, axis=1)

    returnsdata = closedata.pct_change().dropna()
    return returnsdata


def get_sector_etf_returns(start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical returns for S&P 500 Sector ETFs.

    Args:
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        DataFrame of daily returns with sector names as columns.
    """
    tickers = list(SECTOR_ETFS.values())
    # Invert dictionary to map tickers back to sector names
    ticker_to_sector = {v: k for k, v in SECTOR_ETFS.items()}

    rawdata = yf.download(tickers, start=start, end=end, progress=False)

    # Handle case where only one ticker is returned
    if len(tickers) == 1:
        closedata = rawdata["Close"].to_frame()
        closedata.columns = tickers
    else:
        closedata = rawdata["Close"].copy()

    # Rename columns from tickers to sector names
    closedata.rename(columns=ticker_to_sector, inplace=True)

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
