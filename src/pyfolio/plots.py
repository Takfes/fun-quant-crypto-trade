"""Visualization functions for portfolio analysis.

This module provides plotting utilities for visualizing portfolio characteristics
and optimization results.
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_risk_return_scatter(results_df: pd.DataFrame, random_portfolios: Optional[tuple] = None) -> plt.Figure:
    """Plot risk vs return for optimized portfolios and optionally random portfolios.

    Args:
        results_df: DataFrame containing optimization results with 'risk', 'return',
        'sharpe', and 'objective' columns.
        random_portfolios: Optional tuple of (risks, returns) arrays for random
        portfolios to display as background.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if random_portfolios is not None:
        risks, returns = random_portfolios
        ax.scatter(
            risks,
            returns,
            c=returns / risks,
            marker="o",
            alpha=0.3,
            s=10,
            cmap="viridis",
            label="Random Portfolios",
        )

    # Plot optimized portfolios
    scatter_opt = ax.scatter(  # noqa : F841
        results_df["risk"],
        results_df["return"],
        c=results_df["sharpe"],
        marker="*",
        s=500,
        cmap="plasma",
        edgecolors="black",
        linewidths=1.5,
        label="Optimized Portfolios",
        zorder=5,
    )

    # Annotate
    for _, row in results_df.iterrows():
        ax.annotate(
            row["objective"],
            (row["risk"], row["return"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )

    ax.set_xlabel("Annualized Risk (Volatility)", fontsize=12)
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.set_title("Portfolio Optimization: Random vs Optimized Portfolios", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
