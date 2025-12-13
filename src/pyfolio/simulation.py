"""Stochastic portfolio generation for Monte Carlo analysis.

This module provides functions to generate random portfolio allocations
for efficient frontier visualization and comparison.
"""

import numpy as np
import pandas as pd


def generate_random_portfolios(returns: pd.DataFrame, num_portfolios: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random portfolio weights and calculate their risk-return characteristics.

    Args:
        returns: DataFrame of asset returns (rows: time periods, columns: assets).
        num_portfolios: Number of random portfolios to generate.

    Returns:
        Tuple of (portfolio_risk, portfolio_returns) arrays with annualized values.
    """
    noa = returns.shape[1]

    # Generate equal-weighted portfolio
    weights_equal = np.ones(noa) / noa

    # Generate portfolios using Dirichlet distribution (naturally sum to 1)
    weights_dirichlet = np.random.dirichlet(np.ones(noa), size=num_portfolios)

    # Generate portfolios using uniform random weights (normalized)
    weights_power = np.random.rand(num_portfolios, noa)
    weights_power /= weights_power.sum(axis=1)[:, np.newaxis]

    # Combine all weight matrices
    weights = np.vstack([weights_equal, weights_dirichlet, weights_power])

    # Reason: Portfolio statistics computed for efficient frontier visualization
    covmatrix = returns.cov()
    portfolio_returns = weights @ returns.mean().values * 252  # Annualized returns
    portfolio_variance = (weights @ covmatrix * weights).sum(axis=1)
    portfolio_risk = (np.sqrt(portfolio_variance) * np.sqrt(252)).values  # Annualized volatility

    return portfolio_risk, portfolio_returns
