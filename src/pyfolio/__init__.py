"""Portfolio optimization and analysis package.

This package provides tools for portfolio optimization, risk metrics calculation,
and visualization of portfolio characteristics.
"""

from .metrics import portfolio_downside_risk, portfolio_return, portfolio_risk  # noqa : F401
from .optimization import OptimizationResult, PortfolioOptimizer  # noqa : F401
from .plotting import plot_risk_return_scatter  # noqa : F401
