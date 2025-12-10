import numpy as np
import pandas as pd
import quantstats as qs
import yfinance as yf
from matplotlib import pyplot as plt
from scipy.optimize import minimize

START = "2020-01-01"
END = "2025-11-30"
RISK_FREE_RATE = 0.0417
MIN_ALLOCATION_WEIGHT = 0.0
MAX_ALLOCATION_WEIGHT = 0.5
NUM_PORTFOLIOS = 10_000

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# }

# sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options=headers)[0]

tickers = [
    "AAPL",
    "AMGN",
    "CAT",
    "FDX",
    "GE",
    "GS",
    "INTC",
    "IRM",
    "JNJ",
    "JPM",
    "LMT",
    "MRK",
    "MRNA",
    "MSFT",
    "MTCH",
    "PLG",
    "RTX",
    "WELL",
    "WMB",
    "XOM",
    "^GSPC",
]

rawdata = yf.download(tickers, start=START, end=END)
closedata = rawdata["Close"].copy()
returnsdata = closedata.pct_change().dropna()

statsdata = pd.DataFrame({
    "min": returnsdata.min(),
    "q99": returnsdata.quantile(0.01),
    "q95": returnsdata.quantile(0.05),
    "mean": returnsdata.mean(),
    "median": returnsdata.median(),
    "max": returnsdata.max(),
    "std": returnsdata.std(),
    "skew": returnsdata.skew(),
    "kurtosis": returnsdata.kurtosis(),
    "annual_returns": returnsdata.mean() * 252,
    "annual_std": returnsdata.std() * np.sqrt(252),
    "coefficient_of_variation": returnsdata.std() * np.sqrt(252) / (returnsdata.mean() * 252),
}).T

stats_ax = statsdata.T.plot.scatter(
    x="annual_std",
    y="annual_returns",
    c="coefficient_of_variation",
    cmap="viridis",
    colorbar=True,
    title="Risk vs Return Scatter Plot",
)

# Annotate each point with its ticker
for ticker, row in statsdata.T.iterrows():
    stats_ax.annotate(
        ticker,
        (row["annual_std"], row["annual_returns"]),
        xytext=(3, 3),  # offset in points
        textcoords="offset points",
        fontsize=8,
    )

covmatrix = returnsdata.cov()
cormatrix = returnsdata.corr()

# Generate random portfolios
weights_equal = np.ones(len(tickers)) / len(tickers)
weights_dirichlet = np.random.dirichlet(np.ones(len(tickers)), size=NUM_PORTFOLIOS)
weights_power = np.random.rand(NUM_PORTFOLIOS, len(tickers))
weights_power /= weights_power.sum(axis=1)[:, np.newaxis]
weights = np.vstack([weights_equal, weights_dirichlet, weights_power])

# Calculate annualized portfolio returns & risk for all portfolios
portfolio_returns = weights @ statsdata.loc["annual_returns"].values
portfolio_variance = (weights @ covmatrix * weights).sum(axis=1)
portfolio_risk = (np.sqrt(portfolio_variance) * np.sqrt(252)).values

# Plot the portfolio risk-return scatter
plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns / portfolio_risk, marker="o")


def portfolio_return(weights, returns):
    """Calculate annualized portfolio return."""
    return np.dot(weights, returns.mean()) * 252


def portfolio_risk(weights, returns):
    """Calculate annualized portfolio volatility."""
    # this would yield same results port_returns.std() * np.sqrt(252)
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))


def portfolio_downside_risk(weights, returns, target=0.0):
    """Calculate annualized downside deviation below target."""
    port_returns = returns @ weights
    downside_diff = np.minimum(0, port_returns - target)
    downside_var = (downside_diff**2).mean()
    return np.sqrt(downside_var) * np.sqrt(252)


def sharpe_ratio_objective(weights, returns, risk_free_rate=RISK_FREE_RATE):
    """Maximize Sharpe Ratio (negative for minimization)."""
    port_rtrn = portfolio_return(weights, returns)
    port_risk = portfolio_risk(weights, returns)
    return -(port_rtrn - risk_free_rate) / port_risk


def sortino_ratio_objective(weights, returns, risk_free_rate=RISK_FREE_RATE, target=0.0):
    """Maximize Sortino Ratio (negative for minimization)."""
    port_return = portfolio_return(weights, returns)
    downside_risk = portfolio_downside_risk(weights, returns, target)
    if downside_risk == 0:
        return -np.inf
    return -(port_return - risk_free_rate) / downside_risk


def var_objective(weights, returns, alpha=0.05):
    """Minimize VaR at confidence level alpha (e.g., 5%)."""
    port_returns = returns @ weights
    # The alpha-th percentile return (e.g., the return at the 5% worst cutoff)
    # We negate it because optimization minimizes, and losses are negative numbers.
    return -np.percentile(port_returns, alpha * 100)


def cvar_objective(weights, returns, alpha=0.05):
    """Minimize CVaR (average loss beyond VaR) at confidence level alpha."""
    port_returns = returns @ weights
    var_threshold = np.percentile(port_returns, alpha * 100)
    # Average of all returns worse than the threshold
    tail_losses = port_returns[port_returns <= var_threshold]
    return -tail_losses.mean()


def omega_ratio_objective(weights, returns, target_return=0.0):
    """Maximize Omega Ratio (Probability weighted gains / losses)."""
    port_returns = returns @ weights
    excess = port_returns - target_return
    positive_sum = excess[excess > 0].sum()
    negative_sum = np.abs(excess[excess < 0].sum())
    if negative_sum == 0:
        return -np.inf
    return -(positive_sum / negative_sum)


def max_drawdown_objective(weights, returns):
    """Minimize Maximum Peak-to-Trough Drawdown."""
    port_returns = returns @ weights
    wealth_index = (1 + port_returns).cumprod()
    peaks = wealth_index.cummax()
    drawdowns = (wealth_index - peaks) / peaks
    return np.abs(drawdowns.min())


def max_diversification_objective(weights, returns):
    """Maximize Diversification Ratio (Weighted Avg Vol / Portfolio Vol)."""
    cov_matrix = returns.cov() * 252
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, asset_vols)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    div_ratio = weighted_avg_vol / port_vol
    return -div_ratio


def risk_parity_objective(weights, returns):
    """Minimize dispersion of risk contributions (make them equal)."""
    cov_matrix = returns.cov() * 252
    port_var = weights @ cov_matrix @ weights
    mrc = (cov_matrix @ weights) / np.sqrt(port_var)
    risk_contribs = weights * mrc
    target_risk = np.sqrt(port_var) / len(weights)
    return np.sum((risk_contribs - target_risk) ** 2)


# Optimization
# Number of Assets
noa = returnsdata.shape[1]
# Constraints
constraints = {
    "type": "eq",
    "fun": lambda x: np.sum(x) - 1,  # Weights must sum to 1
}
# Bounds for each stock's weight
bounds = tuple((MIN_ALLOCATION_WEIGHT, MAX_ALLOCATION_WEIGHT) for _ in range(noa))
# Initialization
weights_init = np.ones(noa) / noa
# Minimization
result = minimize(
    var_objective, weights_init, args=(returnsdata), method="SLSQP", bounds=bounds, constraints=constraints
)

# result.fun
# portfolio_return(result.x, returnsdata)
# portfolio_risk(result.x, returnsdata)
# sharpe_ratio_objective(result.x, returnsdata)
# sortino_ratio_objective(result.x, returnsdata)
# var_objective(result.x, returnsdata)
# cvar_objective(result.x, returnsdata)
# omega_ratio_objective(result.x, returnsdata)
# max_drawdown_objective(result.x, returnsdata)
# max_diversification_objective(result.x, returnsdata)
# risk_parity_objective(result.x, returnsdata)

# qs.stats.sharpe(returnsdata @ result.x, rf=RISK_FREE_RATE, periods=252)
# qs.stats.sortino(returnsdata @ result.x, rf=RISK_FREE_RATE, periods=252)
# qs.stats.var(returnsdata @ result.x)
# qs.stats.cvar(returnsdata @ result.x)
# omega_ratio = qs.stats.omega(returnsdata @ result.x, required_return=0, rf=RISK_FREE_RATE, periods=252)
# qs.stats.max_drawdown(returnsdata @ result.x)

# qs.reports.basic(returnsdata @ result.x, rf=RISK_FREE_RATE, title="Optimized Portfolio Report")

# Minimization Loop for all objectives
objectives = {
    "sharpe": sharpe_ratio_objective,
    "sortino": sortino_ratio_objective,
    "var": var_objective,
    "cvar": cvar_objective,
    "omega": omega_ratio_objective,
    "max_drawdown": max_drawdown_objective,
    "max_diversification": max_diversification_objective,
    "risk_parity": risk_parity_objective,
}

results = []

for name, obj_fn in objectives.items():
    opt_result = minimize(
        obj_fn,
        weights_init,
        args=(returnsdata,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not opt_result.success:
        continue

    w_opt = opt_result.x
    results.append({
        "objective": name,
        "fun_value": opt_result.fun,
        "return": portfolio_return(w_opt, returnsdata),
        "risk": portfolio_risk(w_opt, returnsdata),
        "sharpe": -sharpe_ratio_objective(w_opt, returnsdata),
        "sortino": -sortino_ratio_objective(w_opt, returnsdata),
        "var": var_objective(w_opt, returnsdata),
        "cvar": cvar_objective(w_opt, returnsdata),
        "omega": -omega_ratio_objective(w_opt, returnsdata),
        "max_drawdown": max_drawdown_objective(w_opt, returnsdata),
        "max_diversification": -max_diversification_objective(w_opt, returnsdata),
        "risk_parity_dispersion": risk_parity_objective(w_opt, returnsdata),
        "weights": w_opt,
    })

results_df = pd.DataFrame(results)


stats_ax = results_df.plot.scatter(
    x="risk",
    y="return",
    c="sharpe",
    cmap="viridis",
    colorbar=True,
    title="Optimized Portfolios Risk vs Return Scatter Plot",
)


# Create comprehensive visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Plot random portfolios as background
scatter_bg = ax.scatter(
    portfolio_risk,
    portfolio_returns,
    c=portfolio_returns / portfolio_risk,
    marker="o",
    alpha=0.3,
    s=10,
    cmap="viridis",
    label="Random Portfolios",
)

# Plot optimized portfolios on top
scatter_opt = ax.scatter(
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

# Annotate optimized portfolios
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

# # Draw efficient frontier
# sorted_portfolios = results_df.sort_values("risk")
# ax.plot(
#     sorted_portfolios["risk"],
#     sorted_portfolios["return"],
#     "r--",
#     linewidth=2,
#     label="Efficient Frontier",
#     zorder=4,
# )

# # Plot risk-free point
# ax.scatter(
#     0,
#     RISK_FREE_RATE,
#     color="green",
#     marker="D",
#     s=200,
#     edgecolors="black",
#     linewidths=1.5,
#     label="Risk-Free Rate",
#     zorder=5,
# )

# # Draw Capital Allocation Line to max Sharpe portfolio
# max_sharpe_portfolio = results_df.loc[results_df["sharpe"].idxmax()]
# ax.plot(
#     [0, max_sharpe_portfolio["risk"]],
#     [RISK_FREE_RATE, max_sharpe_portfolio["return"]],
#     "g-",
#     linewidth=2,
#     label=f"CAL (Max Sharpe: {max_sharpe_portfolio['objective']})",
#     zorder=4,
# )

# Formatting
ax.set_xlabel("Annualized Risk (Volatility)", fontsize=12)
ax.set_ylabel("Annualized Return", fontsize=12)
ax.set_title("Portfolio Optimization: Random vs Optimized Portfolios", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
# plt.colorbar(scatter_opt, ax=ax, label="Sharpe Ratio")
plt.tight_layout()
plt.show()
