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


class Backtester:
    """
    A simple event-driven backtester for portfolio strategies.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital

    def run_fixed_allocation(
        self, prices: pd.DataFrame, weights: dict[str, float], rebalance_freq: str = "ME"
    ) -> pd.DataFrame:
        """
        Backtest a fixed allocation strategy with periodic rebalancing.

        Args:
            prices: DataFrame of asset prices (index is Datetime).
            weights: Dictionary of target weights {ticker: weight}.
            rebalance_freq: Pandas frequency string (e.g., 'ME' for Month End).

        Returns:
            DataFrame with 'Portfolio Value' and daily returns.
        """
        # Validation
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Align data
        assets = list(weights.keys())
        # Ensure we have all assets in prices
        missing_assets = [a for a in assets if a not in prices.columns]
        if missing_assets:
            raise ValueError(f"Prices DataFrame missing assets: {missing_assets}")

        prices = prices[assets].dropna()

        if prices.empty:
            raise ValueError("No price data available for selected assets.")

        # Identify rebalance dates (last trading day of each period)
        # We use a robust grouper to find the last valid index in each period
        rebalance_dates = prices.groupby(pd.Grouper(freq=rebalance_freq)).apply(lambda x: x.index[-1])

        # Ensure start date is included for initial allocation
        start_date = prices.index[0]
        if start_date not in rebalance_dates.values:
            # Prepend start date
            all_dates = [start_date, *rebalance_dates.tolist()]
            all_dates = sorted(set(all_dates))  # Remove duplicates and sort
        else:
            all_dates = rebalance_dates.tolist()

        # Simulation State
        current_capital = self.initial_capital
        holdings = dict.fromkeys(assets, 0.0)
        portfolio_history = []

        # Iterate through periods
        for i in range(len(all_dates) - 1):
            start_period = all_dates[i]
            end_period = all_dates[i + 1]

            # 1. Rebalance at start_period (using prices at start_period)
            current_prices = prices.loc[start_period]
            for asset, weight in weights.items():
                holdings[asset] = (current_capital * weight) / current_prices[asset]

            # 2. Evolve value through the period
            # Get prices from start_period (exclusive) to end_period (inclusive)
            period_prices = prices.loc[start_period:end_period].iloc[1:]

            if period_prices.empty:
                continue

            # Calculate daily value: sum(holdings * price)
            daily_values = period_prices.mul(pd.Series(holdings)).sum(axis=1)
            portfolio_history.append(daily_values)

            # Update capital for next rebalance
            current_capital = daily_values.iloc[-1]

        # Combine history
        if portfolio_history:
            full_curve = pd.concat(portfolio_history)
            # Add initial point
            full_curve.loc[start_date] = self.initial_capital
            full_curve = full_curve.sort_index()
        else:
            # Single day or no data
            full_curve = pd.Series([self.initial_capital], index=[start_date])

        return full_curve.to_frame(name="Portfolio Value")

    def run_walk_forward(  # noqa: C901
        self,
        prices: pd.DataFrame,
        optimizer_func: callable,
        lookback_window: int = 252,
        rebalance_freq: str = "ME",
    ) -> pd.DataFrame:
        """
        Backtest a strategy using walk-forward optimization.

        Args:
            prices: DataFrame of asset prices (index is Datetime).
            optimizer_func: Function that takes (returns_df) and returns weights dict/array.
            lookback_window: Number of trading days to look back for optimization.
            rebalance_freq: Pandas frequency string (e.g., 'ME' for Month End).

        Returns:
            DataFrame with 'Portfolio Value'.
        """
        # Align data
        prices = prices.dropna()
        if prices.empty:
            raise ValueError("No price data available.")

        assets = prices.columns.tolist()

        # Identify rebalance dates
        rebalance_dates = prices.groupby(pd.Grouper(freq=rebalance_freq)).apply(lambda x: x.index[-1])

        # Filter rebalance dates to ensure we have enough history for the first rebalance
        valid_rebalance_dates = [d for d in rebalance_dates if prices.index.get_loc(d) >= lookback_window]

        if not valid_rebalance_dates:
            raise ValueError(f"Not enough history for lookback window of {lookback_window} days.")

        # Simulation State
        start_date = valid_rebalance_dates[0]
        current_capital = self.initial_capital
        holdings = dict.fromkeys(assets, 0.0)
        portfolio_history = []

        # Iterate through periods
        for i in range(len(valid_rebalance_dates) - 1):
            rebalance_date = valid_rebalance_dates[i]
            next_rebalance_date = valid_rebalance_dates[i + 1]

            # 1. Get Historical Data for Optimization
            # Slice from (rebalance_date - lookback) to rebalance_date
            # We use iloc to get the exact window size
            loc_idx = prices.index.get_loc(rebalance_date)
            start_idx = loc_idx - lookback_window + 1  # +1 to include rebalance_date in count

            if start_idx < 0:
                continue  # Should be handled by valid_rebalance_dates check, but safety first

            hist_prices = prices.iloc[start_idx : loc_idx + 1]
            hist_returns = hist_prices.pct_change().dropna()

            # 2. Run Optimizer
            try:
                # optimizer_func is expected to return a dictionary {ticker: weight} or array
                weights_result = optimizer_func(hist_returns)

                # Handle different return types from optimizer
                if isinstance(weights_result, (np.ndarray, list)):
                    weights = dict(zip(assets, weights_result))
                elif isinstance(weights_result, dict):
                    weights = weights_result
                else:
                    raise TypeError("Optimizer must return dict or array/list of weights")  # noqa: TRY301

            except Exception as e:
                print(f"Optimization failed at {rebalance_date}: {e}. Keeping previous weights.")
                # If optimization fails, we keep previous holdings (drift) or could rebalance to previous weights
                # Here we choose to drift (do nothing), effectively skipping rebalance logic but still evolving value
                # To do that, we just skip the rebalancing part below
                pass
            else:
                # 3. Rebalance
                current_prices = prices.loc[rebalance_date]
                for asset, weight in weights.items():
                    holdings[asset] = (current_capital * weight) / current_prices[asset]

            # 4. Evolve value through the period
            period_prices = prices.loc[rebalance_date:next_rebalance_date].iloc[1:]

            if period_prices.empty:
                continue

            daily_values = period_prices.mul(pd.Series(holdings)).sum(axis=1)
            portfolio_history.append(daily_values)

            current_capital = daily_values.iloc[-1]

        # Combine history
        if portfolio_history:
            full_curve = pd.concat(portfolio_history)
            # Add initial point
            full_curve.loc[start_date] = self.initial_capital
            full_curve = full_curve.sort_index()
        else:
            full_curve = pd.Series([self.initial_capital], index=[start_date])

        return full_curve.to_frame(name="Portfolio Value")

    def run_buy_and_hold(self, prices: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
        """
        Backtest a Buy and Hold strategy (no rebalancing).

        Args:
            prices: DataFrame of asset prices (index is Datetime).
            weights: Dictionary of initial target weights {ticker: weight}.

        Returns:
            DataFrame with 'Portfolio Value'.
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
        # Value = sum(shares * price) for each day
        portfolio_value = prices.mul(pd.Series(holdings)).sum(axis=1)

        return portfolio_value.to_frame(name="Portfolio Value")

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
