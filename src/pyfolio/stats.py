"""Portfolio risk and return metrics.

This module provides core functions for calculating portfolio-level
risk and return characteristics.
"""

import numpy as np
import pandas as pd


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


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Calculate annualized portfolio return.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.

    Returns:
        Annualized portfolio return.
    """
    return np.dot(weights, returns.mean()) * 252


def portfolio_risk(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Calculate annualized portfolio volatility.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.

    Returns:
        Annualized portfolio volatility (standard deviation).
    """
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))


def portfolio_downside_risk(weights: np.ndarray, returns: pd.DataFrame, target: float = 0.0) -> float:
    """Calculate annualized downside deviation below target.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.
        target: Target return threshold (default: 0.0).

    Returns:
        Annualized downside deviation.
    """
    port_returns = returns @ weights
    downside_diff = np.minimum(0, port_returns - target)
    downside_var = (downside_diff**2).mean()
    return np.sqrt(downside_var) * np.sqrt(252)
