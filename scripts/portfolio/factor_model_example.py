from pyfolio.data import get_ch4_daily, get_ff3_daily, get_ff5_daily, get_sim_daily, get_ticker_returns
from pyfolio.factors import LinearFactorModel

START = "2020-01-01"
END = "2025-11-30"
tickers = [
    "AMZN",
    "MSFT",
    "GOOGL",
]

returns = get_ticker_returns(tickers, START, END)

dsim = get_sim_daily()
msim = LinearFactorModel()
msim.fit(returns, dsim, fit_intercept=False)
mu_sim = msim.expected_returns()
sigma_sim = msim.covariance()

dff3 = get_ff3_daily()
mff3 = LinearFactorModel()
mff3.fit(returns, dff3, fit_intercept=False)
mu_ff3 = mff3.expected_returns()
sigma_ff3 = mff3.covariance()

dch4 = get_ch4_daily()
mch4 = LinearFactorModel()
mch4.fit(returns, dch4, fit_intercept=False)
mu_ch4 = mch4.expected_returns()
sigma_ch4 = mch4.covariance()

dff5 = get_ff5_daily()
mff5 = LinearFactorModel()
mff5.fit(returns, dff5, fit_intercept=False)
mu_ff5 = mff5.expected_returns()
sigma_ff5 = mff5.covariance()
