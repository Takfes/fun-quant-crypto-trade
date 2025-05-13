import datetime
from typing import Optional

import ccxt
import pandas as pd
import yfinance as yf

COLUMNS = ["open", "high", "low", "close", "volume"]
INDEX = "timestamp"


def validate_columns(dataframe, required_columns):
    if not all(col in dataframe.columns.str.lower() for col in required_columns):
        raise ValueError


def parse_date(date_str: str) -> datetime.datetime:
    """
    Parses a date string in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
    If time is omitted, defaults to 00:00:00.
    """
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")


def resolve_time_range(
    from_date: Optional[str],
    to_date: Optional[str],
    lookback_period: Optional[str],
) -> tuple[datetime.datetime, datetime.datetime]:
    """
    Resolves the from_date and to_date based on provided parameters.
    If both from_date and lookback_period are provided, uses the earlier of the two.
    If neither is provided, defaults to maximum allowable range.
    """
    # Determine to_date
    if to_date:
        to_dt = parse_date(to_date)
    else:
        to_dt = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Determine from_date
    if from_date:
        from_dt = parse_date(from_date)
    elif lookback_period:
        # Parse lookback_period (e.g., '1d', '1mo', '1y')
        unit = lookback_period[-1]
        value = int(lookback_period[:-1])
        if unit == "d":
            delta = datetime.timedelta(days=value)
        elif unit == "w":
            delta = datetime.timedelta(weeks=value)
        elif unit == "m":
            delta = datetime.timedelta(days=30 * value)
        elif unit == "y":
            delta = datetime.timedelta(days=365 * value)
        else:
            raise ValueError(f"Invalid lookback_period unit: {unit}")
        from_dt = to_dt - delta
    else:
        from_dt = None

    # If both from_date and lookback_period are provided, use the earlier date
    if from_date and lookback_period:
        from_dt = max(parse_date(from_date), from_dt)

    return from_dt, to_dt


def fetch_stock_ohlcv_data(
    symbol: str,
    timeframe: str = "1d",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    lookback_period: Optional[str] = None,
    include_symbol: bool = True,
) -> pd.DataFrame:
    """
    Fetches OHLCV data for a stock symbol using yfinance.
    """
    from_dt, to_dt = resolve_time_range(from_date, to_date, lookback_period)

    if from_dt:
        data = yf.download(symbol, interval=timeframe, start=from_dt, end=to_dt)
    else:
        # If from_dt is None, yfinance will fetch maximum data
        data = yf.download(symbol, interval=timeframe, end=to_dt)

    if data.empty:
        return pd.DataFrame(columns=COLUMNS, index=[INDEX])

    if data.columns.nlevels > 1:
        data = data.droplevel(axis=1, level=1)

    # Fix index name
    data.index.name = INDEX
    data.reset_index(inplace=True)
    data[INDEX] = pd.to_datetime(data[INDEX])
    data.set_index(INDEX, inplace=True)

    # Fix column names and index name
    validate_columns(data, COLUMNS)
    data.columns = data.columns.str.lower()
    data = data[COLUMNS]

    if include_symbol:
        data.insert(0, "symbol", symbol)

    return data


def fetch_crypto_ohlcv_data(
    symbol: str,
    timeframe: str = "1d",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    lookback_period: Optional[str] = None,
    exchange_id: str = "binance",
    include_symbol: bool = True,
) -> pd.DataFrame:
    """
    Fetches OHLCV data for a cryptocurrency symbol using ccxt.
    """
    from_dt, to_dt = resolve_time_range(from_date, to_date, lookback_period)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()

    # check if from_dt is None, if so, set it to a default value
    if from_dt is None:
        from_dt = datetime.datetime(2010, 1, 1)

    since = int(from_dt.timestamp() * 1000)
    end_ts = int(to_dt.timestamp() * 1000)

    all_ohlcv = []
    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # Move to the next timestamp

    if not all_ohlcv:
        return pd.DataFrame(columns=COLUMNS, index=[INDEX])

    data = pd.DataFrame(all_ohlcv, columns=[INDEX] + COLUMNS)
    data[INDEX] = pd.to_datetime(data[INDEX], unit="ms")
    data.set_index(INDEX, inplace=True)

    if include_symbol:
        data.insert(0, "symbol", symbol)

    return data
