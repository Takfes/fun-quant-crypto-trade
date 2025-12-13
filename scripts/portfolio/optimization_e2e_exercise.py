import matplotlib.pyplot as plt

from pyfolio.metrics import compute_statistics
from pyfolio.optimization import PortfolioOptimizer
from pyfolio.plotting import plot_risk_return_scatter
from pyfolio.stochastic import generate_random_portfolios
from pyfolio.utils import get_tickers_data

START = "2020-01-01"
END = "2025-11-30"
RISK_FREE_RATE = 0.0417
NUM_PORTFOLIOS = 10_000

tickers = [
    "AAPL",
    "GE",
    "MSFT",
    "WMB",
    "TSLA",
    "GOOGL",
]


def main():
    print("Fetching data...")
    returns = get_tickers_data(tickers, START, END)

    print("Computing statistics...")
    stats_df = compute_statistics(returns)
    print(stats_df)

    print("Running optimization suite...")
    popt = PortfolioOptimizer(returns, risk_free_rate=RISK_FREE_RATE)
    optres = popt.optimize_all()

    print("Generating random portfolios for comparison...")
    random_risk, random_ret = generate_random_portfolios(returns, NUM_PORTFOLIOS)

    print("Plotting results...")
    fig = plot_risk_return_scatter(optres, random_portfolios=(random_risk, random_ret))  # noqa : F841
    plt.show()


if __name__ == "__main__":
    main()
