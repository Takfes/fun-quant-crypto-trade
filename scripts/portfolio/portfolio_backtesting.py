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
    # Fetch data starting from 2020 to allow for warm-up period
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

    # Define a common start date for the simulation to ensure apples-to-apples comparison
    # We start simulation in 2021, using 2020 data for the initial lookback of Walk-Forward
    sim_start_date = "2021-01-01"
    print(f"\nStarting all simulations from: {sim_start_date}")

    print("\n--- Running Buy and Hold ---")
    bnh_result = backtester.run_buy_and_hold(prices, initial_weights, start_date=sim_start_date)
    print("Buy & Hold Metrics:", bnh_result.metrics)

    print("\n--- Running Fixed Allocation (Rebalanced Monthly) ---")
    fixed_result = backtester.run_fixed_allocation(
        prices, initial_weights, rebalance_freq="ME", start_date=sim_start_date
    )
    print("Fixed Allocation Metrics:", fixed_result.metrics)

    print("\n--- Running Walk-Forward Optimization (Max Sharpe) ---")
    wf_result = backtester.run_walk_forward(
        prices,
        optimizer_func=max_sharpe_optimizer,
        lookback_window=126,  # ~6 months
        rebalance_freq="ME",
        start_date=sim_start_date,
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

    # Drawdown Comparison
    plt.figure(figsize=(12, 6))
    # We can manually plot drawdowns or use the helper method for individual ones
    # Here we plot combined drawdowns for comparison
    for name, res in [("Buy & Hold", bnh_result), ("Fixed", fixed_result), ("Walk-Forward", wf_result)]:
        series = res.equity_curve["Portfolio Value"]
        dd = (series - series.cummax()) / series.cummax()
        plt.plot(dd.index, dd, label=name)

    plt.title("Drawdown Comparison")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.fill_between(dd.index, 0, -1, color="gray", alpha=0.05)  # Visual baseline
    plt.tight_layout()
    plt.show()

    # Individual Strategy Analysis (Walk-Forward)
    print("\nDisplaying detailed analysis for Walk-Forward Strategy...")
    wf_result.plot_weights(title="Walk-Forward: Dynamic Allocation")
    plt.show()

    wf_result.plot_drawdown(title="Walk-Forward: Underwater Plot")
    plt.show()

    wf_result.plot_rolling_metrics(title="Walk-Forward: Rolling Sharpe (6-Month)")
    plt.show()


if __name__ == "__main__":
    main()
