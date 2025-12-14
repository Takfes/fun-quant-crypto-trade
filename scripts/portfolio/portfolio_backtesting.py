"""
Example script to demonstrate the Backtester class.
"""

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from pyfolio.optimizer import PortfolioOptimizer
from pyfolio.simulation import Backtester


def get_data():
    """Fetch some sample data for backtesting."""
    tickers = ["SPY", "TLT", "GLD"]
    print(f"Fetching data for {tickers}...")
    # yfinance returns a MultiIndex by default for multiple tickers, we want just the Close prices
    data = yf.download(tickers, start="2020-01-01", end="2023-12-31", progress=False)["Close"]
    return data


def max_sharpe_optimizer(returns_df):
    """
    Simple wrapper around PortfolioOptimizer to return weights for Max Sharpe.
    """
    # Initialize optimizer with the historical slice
    opt = PortfolioOptimizer(returns_df, risk_free_rate=0.0)

    # Run optimization
    result = opt.optimize("sharpe")

    if result.success:
        # Return dictionary of weights
        return dict(zip(returns_df.columns, result.weights))
    else:
        # Fallback to equal weights if optimization fails
        n = len(returns_df.columns)
        return dict(zip(returns_df.columns, [1 / n] * n))


def main():
    # 1. Get Data
    prices = get_data()

    # 2. Initialize Backtester
    backtester = Backtester(initial_capital=10000.0)

    # 3. Define Benchmarks
    # Equal weights for 3 assets
    initial_weights = dict.fromkeys(prices.columns, 1 / 3)

    print("\n--- Running Buy and Hold ---")
    bnh_result = backtester.run_buy_and_hold(prices, initial_weights)
    print("Buy & Hold Metrics:", bnh_result.metrics)

    print("\n--- Running Fixed Allocation (Rebalanced Monthly) ---")
    fixed_result = backtester.run_fixed_allocation(prices, initial_weights, rebalance_freq="ME")
    print("Fixed Allocation Metrics:", fixed_result.metrics)

    print("\n--- Running Walk-Forward Optimization (Max Sharpe) ---")
    # We use a shorter lookback for this example to ensure we have enough data
    wf_result = backtester.run_walk_forward(
        prices,
        optimizer_func=max_sharpe_optimizer,
        lookback_window=126,  # ~6 months
        rebalance_freq="ME",
    )
    print("Walk-Forward Metrics:", wf_result.metrics)

    # 4. Compare Results
    results = pd.DataFrame({
        "Buy & Hold": bnh_result.metrics,
        "Fixed Rebal": fixed_result.metrics,
        "Walk-Forward": wf_result.metrics,
    })

    print("\n--- Comparative Results ---")
    print(results.T)

    # 5. Plotting
    print("\nGenerating plots...")

    # Combined Equity Curve
    combined = pd.concat(
        [
            bnh_result.equity_curve.rename(columns={"Portfolio Value": "Buy & Hold"}),
            fixed_result.equity_curve.rename(columns={"Portfolio Value": "Fixed Rebal"}),
            wf_result.equity_curve.rename(columns={"Portfolio Value": "Walk-Forward"}),
        ],
        axis=1,
    )

    plt.figure(figsize=(12, 6))
    combined.plot(ax=plt.gca())
    plt.title("Portfolio Value Comparison")
    plt.ylabel("Value ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot Weights for each strategy
    # Buy & Hold (Drift visualization)
    bnh_result.plot_weights(title="Buy & Hold: Weight Drift Over Time")
    plt.show()

    # Fixed Allocation (Sawtooth pattern visualization)
    fixed_result.plot_weights(title="Fixed Allocation: Rebalancing Effects")
    plt.show()

    # Walk-Forward (Regime change visualization)
    wf_result.plot_weights(title="Walk-Forward Optimization: Dynamic Allocation")
    plt.show()


if __name__ == "__main__":
    main()
