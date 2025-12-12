import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()
# fetch the daily returns for a stock
stock = qs.utils.download_returns("META")
# show sharpe ratio
qs.stats.sharpe(stock)
# or using extend_pandas() :)
stock.sharpe()

# plot a snapshot
qs.plots.snapshot(stock, title="Facebook Performance", show=True)
# generate a full report
qs.reports.full(stock, "SPY", output="facebook_report.html")
# or an HTML report
qs.reports.html(stock, "SPY")
