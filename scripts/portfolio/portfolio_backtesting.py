"""
Example script to demonstrate the Backtester class.
"""

import pandas as pd
import yfinance as yf

from pyfolio.optimizer import PortfolioOptimizer
from pyfolio.simulation import Backtester


def get_data():
    """Fetch some sample data for backtesting."""
    tickers = ["SPY", "TLT", "GLD"]
    print(f"Fetching data for {tickers}...")
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
    bnh_curve = backtester.run_buy_and_hold(prices, initial_weights)
    bnh_metrics = backtester.calculate_metrics(bnh_curve)
    print("Buy & Hold Metrics:", bnh_metrics)

    print("\n--- Running Fixed Allocation (Rebalanced Monthly) ---")
    fixed_curve = backtester.run_fixed_allocation(prices, initial_weights, rebalance_freq="ME")
    fixed_metrics = backtester.calculate_metrics(fixed_curve)
    print("Fixed Allocation Metrics:", fixed_metrics)

    print("\n--- Running Walk-Forward Optimization (Max Sharpe) ---")
    # We use a shorter lookback for this example to ensure we have enough data
    wf_curve = backtester.run_walk_forward(
        prices,
        optimizer_func=max_sharpe_optimizer,
        lookback_window=126,  # ~6 months
        rebalance_freq="ME",
    )
    wf_metrics = backtester.calculate_metrics(wf_curve)
    print("Walk-Forward Metrics:", wf_metrics)

    # 4. Compare Results
    results = pd.DataFrame({"Buy & Hold": bnh_metrics, "Fixed Rebal": fixed_metrics, "Walk-Forward": wf_metrics})

    print("\n--- Comparative Results ---")
    print(results.T)

    # Optional: Plotting (if running interactively)
    # combined = pd.concat([
    #     bnh_curve.rename(columns={"Portfolio Value": "Buy & Hold"}),
    #     fixed_curve.rename(columns={"Portfolio Value": "Fixed Rebal"}),
    #     wf_curve.rename(columns={"Portfolio Value": "Walk-Forward"})
    # ], axis=1)
    # combined.plot()
    # plt.show()


if __name__ == "__main__":
    main()
