"""Stochastic portfolio generation for Monte Carlo analysis.

This module provides functions to generate random portfolio allocations
for efficient frontier visualization and comparison.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
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


@dataclass
class BacktestResult:
    """
    Container for backtest results including equity curve, weights, and metrics.
    """

    equity_curve: pd.DataFrame
    weights_history: pd.DataFrame
    metrics: dict

    def plot_weights(self, title: str = "Portfolio Weights Allocation"):
        """Plot the historical weights as a stacked area chart."""
        if self.weights_history.empty:
            print("No weights history to plot.")
            return

        # Plotting
        ax = self.weights_history.plot.area(figsize=(12, 6), title=title, alpha=0.8, stacked=True)
        plt.ylabel("Allocation")
        plt.xlabel("Date")
        plt.margins(0, 0)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        plt.tight_layout()
        return ax


class Backtester:
    """
    A simple event-driven backtester for portfolio strategies.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def calculate_metrics(self, portfolio_value: pd.DataFrame) -> dict:
        """
        Calculate standard performance metrics.

        Args:
            portfolio_value: DataFrame with 'Portfolio Value' column.

        Returns:
            Dictionary of metrics.
        """
        series = portfolio_value["Portfolio Value"]
        returns = series.pct_change().dropna()

        if len(series) < 2:
            return {}

        total_return = (series.iloc[-1] / series.iloc[0]) - 1

        # Annualization factor
        days = (series.index[-1] - series.index[0]).days
        years = days / 365.25
        if years > 0:
            cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
        else:
            cagr = 0.0

        volatility = returns.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0.0

        # Max Drawdown
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        max_drawdown = drawdown.min()

        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility (Ann.)": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
        }

    def run_buy_and_hold(self, prices: pd.DataFrame, weights: dict[str, float]) -> BacktestResult:
        """
        Backtest a Buy and Hold strategy (no rebalancing).

        Args:
            prices: DataFrame of asset prices (index is Datetime).
            weights: Dictionary of initial target weights {ticker: weight}.

        Returns:
            BacktestResult object.
        """
        # Validation
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Align data
        assets = list(weights.keys())
        missing_assets = [a for a in assets if a not in prices.columns]
        if missing_assets:
            raise ValueError(f"Prices DataFrame missing assets: {missing_assets}")

        prices = prices[assets].dropna()
        if prices.empty:
            raise ValueError("No price data available for selected assets.")

        # Initial Allocation
        start_date = prices.index[0]
        initial_prices = prices.loc[start_date]
        holdings = {}

        for asset, weight in weights.items():
            holdings[asset] = (self.initial_capital * weight) / initial_prices[asset]

        # Calculate daily value for the entire period
        asset_values = prices.mul(pd.Series(holdings))
        portfolio_value = asset_values.sum(axis=1).to_frame(name="Portfolio Value")

        # Calculate weights over time
        weights_history = asset_values.div(portfolio_value["Portfolio Value"], axis=0)

        metrics = self.calculate_metrics(portfolio_value)

        return BacktestResult(portfolio_value, weights_history, metrics)

    def run_fixed_allocation(
        self, prices: pd.DataFrame, weights: dict[str, float], rebalance_freq: str = "ME"
    ) -> BacktestResult:
        """
        Backtest a fixed allocation strategy with periodic rebalancing.

        Args:
            prices: DataFrame of asset prices (index is Datetime).
            weights: Dictionary of target weights {ticker: weight}.
            rebalance_freq: Pandas frequency string (e.g., 'ME' for Month End).

        Returns:
            BacktestResult object.
        """
        # Validation
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Align data
        assets = list(weights.keys())
        missing_assets = [a for a in assets if a not in prices.columns]
        if missing_assets:
            raise ValueError(f"Prices DataFrame missing assets: {missing_assets}")

        prices = prices[assets].dropna()

        if prices.empty:
            raise ValueError("No price data available for selected assets.")

        # Identify rebalance dates
        rebalance_dates = prices.groupby(pd.Grouper(freq=rebalance_freq)).apply(lambda x: x.index[-1])

        # Ensure start date is included
        start_date = prices.index[0]
        if start_date not in rebalance_dates.values:
            all_dates = [start_date, *rebalance_dates.tolist()]
            all_dates = sorted(set(all_dates))
        else:
            all_dates = rebalance_dates.tolist()

        # Simulation State
        current_capital = self.initial_capital
        holdings = dict.fromkeys(assets, 0.0)
        portfolio_history = []
        weights_history_list = []

        # Iterate through periods
        for i in range(len(all_dates) - 1):
            start_period = all_dates[i]
            end_period = all_dates[i + 1]

            # 1. Rebalance at start_period
            current_prices = prices.loc[start_period]
            for asset, weight in weights.items():
                holdings[asset] = (current_capital * weight) / current_prices[asset]

            # Capture weights at rebalance (start of period)
            # These are exactly the target weights
            w_series = pd.Series(weights, name=start_period)
            weights_history_list.append(w_series.to_frame().T)

            # 2. Evolve value through the period
            period_prices = prices.loc[start_period:end_period].iloc[1:]

            if period_prices.empty:
                continue

            daily_asset_values = period_prices.mul(pd.Series(holdings))
            daily_total = daily_asset_values.sum(axis=1)

            portfolio_history.append(daily_total)
            weights_history_list.append(daily_asset_values.div(daily_total, axis=0))

            # Update capital for next rebalance
            current_capital = daily_total.iloc[-1]

        # Combine history
        if portfolio_history:
            full_curve = pd.concat(portfolio_history)
            full_curve.loc[start_date] = self.initial_capital
            full_curve = full_curve.sort_index().to_frame(name="Portfolio Value")

            full_weights = pd.concat(weights_history_list).sort_index()
            full_weights = full_weights.fillna(0.0)
        else:
            full_curve = pd.Series([self.initial_capital], index=[start_date]).to_frame(name="Portfolio Value")
            full_weights = pd.DataFrame([weights], index=[start_date])

        metrics = self.calculate_metrics(full_curve)
        return BacktestResult(full_curve, full_weights, metrics)

    def run_walk_forward(  # noqa: C901
        self,
        prices: pd.DataFrame,
        optimizer_func: callable,
        lookback_window: int = 252,
        rebalance_freq: str = "ME",
    ) -> BacktestResult:
        """
        Backtest a strategy using walk-forward optimization.

        Args:
            prices: DataFrame of asset prices (index is Datetime).
            optimizer_func: Function that takes (returns_df) and returns weights dict/array.
            lookback_window: Number of trading days to look back for optimization.
            rebalance_freq: Pandas frequency string (e.g., 'ME' for Month End).

        Returns:
            BacktestResult object.
        """
        # Align data
        prices = prices.dropna()
        if prices.empty:
            raise ValueError("No price data available.")

        assets = prices.columns.tolist()

        # Identify rebalance dates
        rebalance_dates = prices.groupby(pd.Grouper(freq=rebalance_freq)).apply(lambda x: x.index[-1])

        # Filter rebalance dates
        valid_rebalance_dates = [d for d in rebalance_dates if prices.index.get_loc(d) >= lookback_window]

        if not valid_rebalance_dates:
            raise ValueError(f"Not enough history for lookback window of {lookback_window} days.")

        # Simulation State
        start_date = valid_rebalance_dates[0]
        current_capital = self.initial_capital
        holdings = dict.fromkeys(assets, 0.0)
        portfolio_history = []
        weights_history_list = []

        # Iterate through periods
        for i in range(len(valid_rebalance_dates) - 1):
            rebalance_date = valid_rebalance_dates[i]
            next_rebalance_date = valid_rebalance_dates[i + 1]

            # 1. Get Historical Data
            loc_idx = prices.index.get_loc(rebalance_date)
            start_idx = loc_idx - lookback_window + 1

            if start_idx < 0:
                continue

            hist_prices = prices.iloc[start_idx : loc_idx + 1]
            hist_returns = hist_prices.pct_change().dropna()

            # 2. Run Optimizer
            try:
                weights_result = optimizer_func(hist_returns)

                if isinstance(weights_result, (np.ndarray, list)):
                    weights = dict(zip(assets, weights_result))
                elif isinstance(weights_result, dict):
                    weights = weights_result
                else:
                    raise TypeError("Optimizer must return dict or array/list of weights")  # noqa: TRY301

            except Exception as e:
                print(f"Optimization failed at {rebalance_date}: {e}. Keeping previous weights.")
                pass
            else:
                # 3. Rebalance
                current_prices = prices.loc[rebalance_date]
                for asset, weight in weights.items():
                    holdings[asset] = (current_capital * weight) / current_prices[asset]

            # Capture weights at rebalance_date (start of period)
            # We calculate actual weights based on holdings and current prices
            # This handles both rebalanced and drifted (failed opt) states
            current_vals = pd.Series(holdings) * prices.loc[rebalance_date]
            total_val = current_vals.sum()
            if total_val > 0:
                w = current_vals / total_val
                weights_history_list.append(w.to_frame(name=rebalance_date).T)

            # 4. Evolve value through the period
            period_prices = prices.loc[rebalance_date:next_rebalance_date].iloc[1:]

            if period_prices.empty:
                continue

            daily_asset_values = period_prices.mul(pd.Series(holdings))
            daily_total = daily_asset_values.sum(axis=1)

            portfolio_history.append(daily_total)
            weights_history_list.append(daily_asset_values.div(daily_total, axis=0))

            current_capital = daily_total.iloc[-1]

        # Combine history
        if portfolio_history:
            full_curve = pd.concat(portfolio_history)
            full_curve.loc[start_date] = self.initial_capital
            full_curve = full_curve.sort_index().to_frame(name="Portfolio Value")

            full_weights = pd.concat(weights_history_list).sort_index()
            full_weights = full_weights.fillna(0.0)
        else:
            full_curve = pd.Series([self.initial_capital], index=[start_date]).to_frame(name="Portfolio Value")
            # Try to capture initial weights if loop ran at least once but no period prices
            if weights_history_list:
                full_weights = pd.concat(weights_history_list).sort_index()
            else:
                full_weights = pd.DataFrame()

        metrics = self.calculate_metrics(full_curve)
        return BacktestResult(full_curve, full_weights, metrics)
