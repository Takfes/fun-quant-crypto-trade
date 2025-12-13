"""Portfolio optimization engine and result containers.

This module provides the main portfolio optimization class that supports
multiple objective functions and constraint specifications.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import objectives
from .stats import portfolio_return, portfolio_risk


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    objective_name: str
    success: bool
    weights: np.ndarray
    objective_value: float
    portfolio_return: float
    portfolio_risk: float
    sharpe_ratio: float
    sortino_ratio: float
    var: float
    cvar: float
    omega_ratio: float
    max_drawdown: float
    diversification_ratio: float
    risk_parity_dispersion: float
    message: str = ""


class PortfolioOptimizer:
    """
    Portfolio optimization engine supporting multiple objective functions.

    Args:
        returns: DataFrame of asset returns.
        risk_free_rate: Annual risk-free rate (default: 0.0).
        min_weight: Minimum weight per asset (default: 0.0).
        max_weight: Maximum weight per asset (default: 1.0).
    """

    def __init__(
        self, returns: pd.DataFrame, risk_free_rate: float = 0.0, min_weight: float = 0.0, max_weight: float = 1.0
    ):
        """Initialize the portfolio optimizer.

        Args:
            returns: DataFrame of asset returns.
            risk_free_rate: Annual risk-free rate (default: 0.0).
            min_weight: Minimum weight per asset (default: 0.0).
            max_weight: Maximum weight per asset (default: 1.0).
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Auto-compute optimization parameters
        self.n_assets = returns.shape[1]
        self.initial_weights = np.ones(self.n_assets) / self.n_assets
        self.bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        self.constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # Pre-bind risk_free_rate to objectives that need it
        self.objectives_map = {
            "sharpe": partial(objectives.sharpe_ratio_objective, risk_free_rate=risk_free_rate),
            "sortino": partial(objectives.sortino_ratio_objective, risk_free_rate=risk_free_rate),
            "var": objectives.var_objective,
            "cvar": objectives.cvar_objective,
            "omega": objectives.omega_ratio_objective,
            "max_drawdown": objectives.max_drawdown_objective,
            "max_diversification": objectives.max_diversification_objective,
            "risk_parity": objectives.risk_parity_objective,
        }

    def _compute_all_metrics(self, weights: np.ndarray) -> dict:
        """Compute all objective metrics for given weights.

        Args:
            weights: Array of portfolio weights.

        Returns:
            Dictionary of all computed metrics.
        """
        return {
            "sharpe_ratio": -self.objectives_map["sharpe"](weights, self.returns),
            "sortino_ratio": -self.objectives_map["sortino"](weights, self.returns),
            "var": self.objectives_map["var"](weights, self.returns),
            "cvar": self.objectives_map["cvar"](weights, self.returns),
            "omega_ratio": -self.objectives_map["omega"](weights, self.returns),
            "max_drawdown": self.objectives_map["max_drawdown"](weights, self.returns),
            "diversification_ratio": -self.objectives_map["max_diversification"](weights, self.returns),
            "risk_parity_dispersion": self.objectives_map["risk_parity"](weights, self.returns),
        }

    def optimize(self, objective_name: str) -> OptimizationResult:
        """
        Run optimization for a single objective.

        Args:
            objective_name: Name of objective function to optimize.

        Returns:
            OptimizationResult object with weights and metrics.
        """
        if objective_name not in self.objectives_map:
            raise ValueError(f"Unknown objective: {objective_name}. Available: {list(self.objectives_map.keys())}")

        obj_func = self.objectives_map[objective_name]

        # Unified interface - thanks to partial in objectives_map
        opt_result = minimize(
            obj_func,
            self.initial_weights,
            args=(self.returns,),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints,
        )

        if not opt_result.success:
            return OptimizationResult(
                objective_name=objective_name,
                success=False,
                weights=np.array([]),
                objective_value=np.nan,
                portfolio_return=np.nan,
                portfolio_risk=np.nan,
                sharpe_ratio=np.nan,
                sortino_ratio=np.nan,
                var=np.nan,
                cvar=np.nan,
                omega_ratio=np.nan,
                max_drawdown=np.nan,
                diversification_ratio=np.nan,
                risk_parity_dispersion=np.nan,
                message=opt_result.message,
            )

        w_opt = opt_result.x

        # Compute all metrics - regardless of optimization objective
        all_metrics = self._compute_all_metrics(w_opt)

        return OptimizationResult(
            objective_name=objective_name,
            success=True,
            weights=w_opt,
            objective_value=opt_result.fun,
            portfolio_return=portfolio_return(w_opt, self.returns),
            portfolio_risk=portfolio_risk(w_opt, self.returns),
            sharpe_ratio=all_metrics["sharpe_ratio"],
            sortino_ratio=all_metrics["sortino_ratio"],
            var=all_metrics["var"],
            cvar=all_metrics["cvar"],
            omega_ratio=all_metrics["omega_ratio"],
            max_drawdown=all_metrics["max_drawdown"],
            diversification_ratio=all_metrics["diversification_ratio"],
            risk_parity_dispersion=all_metrics["risk_parity_dispersion"],
            message="Optimization successful",
        )

    def optimize_all(self) -> pd.DataFrame:
        """
        Run optimization for all available objectives.

        Returns:
            DataFrame with results from all optimizations.
        """
        results = []

        for objective_name in self.objectives_map:
            result = self.optimize(objective_name)

            if not result.success:
                continue

            results.append({
                "objective": result.objective_name,
                "fun_value": result.objective_value,
                "return": result.portfolio_return,
                "risk": result.portfolio_risk,
                "sharpe": result.sharpe_ratio,
                "sortino": result.sortino_ratio,
                "var": result.var,
                "cvar": result.cvar,
                "omega": result.omega_ratio,
                "max_drawdown": result.max_drawdown,
                "diversification": result.diversification_ratio,
                "risk_parity": result.risk_parity_dispersion,
                "weights": result.weights,
            })

        return pd.DataFrame(results)

    def get_available_objectives(self) -> list[str]:
        """Return list of available objective functions.

        Returns:
            List of objective function names.
        """
        return list(self.objectives_map.keys())
