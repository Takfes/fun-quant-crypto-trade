"""Portfolio optimization objective functions.

This module implements various objective functions for portfolio optimization,
including risk-adjusted returns, downside protection, and diversification metrics.
"""

import numpy as np
import pandas as pd

from .stats import portfolio_downside_risk, portfolio_return, portfolio_risk


def sharpe_ratio_objective(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Maximize Sharpe Ratio (negative for minimization).

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.
        risk_free_rate: Annual risk-free rate (default: 0.0).

    Returns:
        Negative Sharpe ratio for minimization.
    """
    port_rtrn = portfolio_return(weights, returns)
    port_risk = portfolio_risk(weights, returns)
    return -(port_rtrn - risk_free_rate) / port_risk


def sortino_ratio_objective(
    weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.0, target: float = 0.0
) -> float:
    """Maximize Sortino Ratio (negative for minimization).

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.
        risk_free_rate: Annual risk-free rate (default: 0.0).
        target: Target return threshold (default: 0.0).

    Returns:
        Negative Sortino ratio for minimization.
    """
    port_return = portfolio_return(weights, returns)
    downside_risk = portfolio_downside_risk(weights, returns, target)
    if downside_risk == 0:
        return -np.inf
    return -(port_return - risk_free_rate) / downside_risk


def var_objective(weights: np.ndarray, returns: pd.DataFrame, alpha: float = 0.05) -> float:
    """Minimize VaR at confidence level alpha.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.
        alpha: Confidence level (default: 0.05 for 95% VaR).

    Returns:
        Negative VaR for minimization.
    """
    port_returns = returns @ weights
    return -np.percentile(port_returns, alpha * 100)


def cvar_objective(weights: np.ndarray, returns: pd.DataFrame, alpha: float = 0.05) -> float:
    """Minimize CVaR (average loss beyond VaR) at confidence level alpha.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.
        alpha: Confidence level (default: 0.05 for 95% CVaR).

    Returns:
        Negative CVaR for minimization.
    """
    port_returns = returns @ weights
    var_threshold = np.percentile(port_returns, alpha * 100)
    tail_losses = port_returns[port_returns <= var_threshold]
    return -tail_losses.mean()


def omega_ratio_objective(weights: np.ndarray, returns: pd.DataFrame, target_return: float = 0.0) -> float:
    """Maximize Omega Ratio (probability weighted gains over losses).

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.
        target_return: Target return threshold (default: 0.0).

    Returns:
        Negative Omega ratio for minimization.
    """
    port_returns = returns @ weights
    excess = port_returns - target_return
    positive_sum = excess[excess > 0].sum()
    negative_sum = np.abs(excess[excess < 0].sum())
    if negative_sum == 0:
        return -np.inf
    return -(positive_sum / negative_sum)


def max_drawdown_objective(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Minimize maximum peak-to-trough drawdown.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.

    Returns:
        Absolute value of maximum drawdown.
    """
    port_returns = returns @ weights
    wealth_index = (1 + port_returns).cumprod()
    peaks = wealth_index.cummax()
    drawdowns = (wealth_index - peaks) / peaks
    return np.abs(drawdowns.min())


def max_diversification_objective(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Maximize diversification ratio (weighted average vol over portfolio vol).

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.

    Returns:
        Negative diversification ratio for minimization.
    """
    cov_matrix = returns.cov() * 252
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, asset_vols)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    div_ratio = weighted_avg_vol / port_vol
    return -div_ratio


def risk_parity_objective(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Minimize dispersion of risk contributions to achieve equal risk allocation.

    Args:
        weights: Array of portfolio weights.
        returns: DataFrame of asset returns.

    Returns:
        Sum of squared deviations from target risk contribution.
    """
    cov_matrix = returns.cov() * 252
    port_var = weights @ cov_matrix @ weights
    # Reason: Marginal risk contribution is the derivative of portfolio variance w.r.t. weights
    mrc = (cov_matrix @ weights) / np.sqrt(port_var)
    risk_contribs = weights * mrc
    target_risk = np.sqrt(port_var) / len(weights)
    return np.sum((risk_contribs - target_risk) ** 2)
