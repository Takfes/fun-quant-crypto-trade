"""Portfolio risk and return metrics.

This module provides core functions for calculating portfolio-level
risk and return characteristics.
"""

import numpy as np
import pandas as pd


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
