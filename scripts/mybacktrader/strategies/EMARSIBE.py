import datetime

import backtrader as bt


class BullishEngulfingStrategy(bt.Strategy):
    """
    Bullish Engulfing Strategy with EMA and RSI

    This strategy combines trend filtering, momentum analysis, and price action:
    - Directional bias using customizable EMA (default 200 EMA).
    - Momentum filter using customizable RSI (default 9) above 50.
    - Bullish Engulfing pattern triggers entries.
    - Stop Loss = 2x previous candle range.
    - Take Profit = 2x Stop Loss distance (Risk:Reward = 1:2).
    - Optional timeout exit after X bars if TP/SL not hit.
    - Adjustable session window (trading only between specific hours).
    - Supports optional multiple simultaneous trades.
    """

    params = (
        ("ema_length", 200),
        ("rsi_length", 9),
        ("rsi_threshold", 50),
        ("timeout_bars", 100),
        ("enable_timeout_exit", False),
        ("allow_multiple_trades", False),
        ("start_date", datetime.datetime(2024, 4, 1)),
        ("end_date", datetime.datetime(2025, 4, 1)),
        ("percentage_of_equity", 5),  # Default position size as percentage of equity
    )

    def __init__(self):
        # Keep track of pending orders and positions
        self.orders = {}
        self.trades = {}
        self.trade_entry_bar = {}

        # Daily close data (equivalent to request.security in Pine)
        self.daily_data = bt.ind.Resample(self.data, timeframe=bt.TimeFrame.Days, compression=1)

        # EMA indicator using daily close
        self.ema = bt.ind.EMA(self.daily_data.close, period=self.p.ema_length)

        # RSI indicator
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_length)

        # Additional indicators for visualization
        self.bullish_engulfing = BullishEngulfingIndicator(self.data)
        self.purple_circle = PurpleCircleIndicator(self.data, self.rsi, self.p.rsi_threshold)

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def is_in_date_range(self):
        """Check if current bar is within specified date range"""
        current_date = self.data.datetime.datetime(0)
        return self.p.start_date <= current_date <= self.p.end_date

    def next(self):
        # Check if we're within the specified date range
        if not self.is_in_date_range():
            return

        # Check for timeout exits for current trades
        if self.p.enable_timeout_exit:
            for trade_id, entry_bar in list(self.trade_entry_bar.items()):
                # If timeout period has elapsed and the trade is still open
                if self.position and (len(self) - entry_bar) >= self.p.timeout_bars:
                    self.close(trade=self.trades[trade_id], comment="Timeout Exit")
                    del self.trade_entry_bar[trade_id]

        # Check if we can enter a new trade
        can_trade = self.p.allow_multiple_trades or not self.position

        if can_trade:
            # Check for bullish engulfing pattern
            bullish_engulfing = self.check_bullish_engulfing()

            # Check if price is above EMA
            price_above_ema = self.data.close[0] > self.ema[0]

            # Check if RSI is above threshold
            rsi_above_threshold = self.rsi[0] > self.p.rsi_threshold

            # Generate buy signal
            buy_signal = bullish_engulfing and price_above_ema and rsi_above_threshold

            if buy_signal:
                # Calculate candle range
                candle_range = self.data.high[0] - self.data.low[0]

                # Calculate entry price, stop loss, and take profit
                entry_price = self.data.close[0]
                stop_loss = entry_price - (2 * candle_range)
                take_profit = entry_price + (4 * candle_range)  # 2x the stop distance

                # Calculate position size (percentage of equity)
                cash = self.broker.getcash()
                value = self.broker.getvalue()
                size = (value * self.p.percentage_of_equity / 100) / entry_price

                # Create bracket order (entry, stop loss, take profit)
                self.log(f"BUY CREATE, {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

                # Place the bracket order
                bracket_order = self.buy_bracket(
                    size=size, price=entry_price, stopprice=stop_loss, limitprice=take_profit, exectype=bt.Order.Market
                )

                # Store order and entry bar info for timeout tracking
                parent_order = bracket_order[0]
                trade_id = parent_order.ref
                self.orders[trade_id] = bracket_order
                self.trade_entry_bar[trade_id] = len(self)

    def check_bullish_engulfing(self):
        """Check for bullish engulfing pattern"""
        if len(self.data) < 2:  # Need at least 2 bars
            return False

        # Previous candle is bearish (close < open)
        prev_bearish = self.data.close[-1] < self.data.open[-1]

        # Current candle is bullish (close > open)
        curr_bullish = self.data.close[0] > self.data.open[0]

        # Current close > previous open
        close_above_prev_open = self.data.close[0] > self.data.open[-1]

        # Current open <= previous close
        open_below_prev_close = self.data.open[0] <= self.data.close[-1]

        return prev_bearish and curr_bullish and close_above_prev_open and open_below_prev_close

    def notify_trade(self, trade):
        """Keep track of completed trades"""
        if trade.isclosed:
            self.log(f"TRADE CLOSED, Ref: {trade.ref}, PNL: {trade.pnl:.2f}, Comm: {trade.commission:.2f}")
            if trade.ref in self.trade_entry_bar:
                del self.trade_entry_bar[trade.ref]
        elif trade.isopen:
            self.log(f"TRADE OPENED, Ref: {trade.ref}, Size: {trade.size}")
            self.trades[trade.ref] = trade

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.6f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.6f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.Status[order.status]}")


class BullishEngulfingIndicator(bt.Indicator):
    """Indicator to detect Bullish Engulfing patterns"""

    lines = ("bullish_engulfing",)
    plotinfo = dict(plot=True, plotymargin=0.05, subplot=False, plotlinelabels=True)
    plotlines = dict(bullish_engulfing=dict(marker="^", markersize=8, color="green", fillstyle="full"))

    def next(self):
        if len(self.data) < 2:  # Need at least 2 bars
            self.lines.bullish_engulfing[0] = 0
            return

        # Previous candle is bearish (close < open)
        prev_bearish = self.data.close[-1] < self.data.open[-1]

        # Current candle is bullish (close > open)
        curr_bullish = self.data.close[0] > self.data.open[0]

        # Current close > previous open
        close_above_prev_open = self.data.close[0] > self.data.open[-1]

        # Current open <= previous close
        open_below_prev_close = self.data.open[0] <= self.data.close[-1]

        # Set indicator value (0 for False, 1 for True)
        if prev_bearish and curr_bullish and close_above_prev_open and open_below_prev_close:
            self.lines.bullish_engulfing[0] = self.data.low[0]  # Plot at the bottom of candle
        else:
            self.lines.bullish_engulfing[0] = 0


class PurpleCircleIndicator(bt.Indicator):
    """Indicator for RSI condition with Bullish Engulfing (purple circle)"""

    lines = ("purple_circle",)
    params = (("rsi_threshold", 50),)
    plotinfo = dict(plot=True, plotymargin=0.05, subplot=False, plotlinelabels=True)
    plotlines = dict(purple_circle=dict(marker="o", markersize=8, color="purple", fillstyle="full"))

    def __init__(self):
        self.bullish_engulfing = BullishEngulfingIndicator(self.data)

    def next(self):
        if len(self.data) < 2:  # Need at least 2 bars
            self.lines.purple_circle[0] = 0
            return

        # Previous candle is bearish (close < open)
        prev_bearish = self.data.close[-1] < self.data.open[-1]

        # Current candle is bullish (close > open)
        curr_bullish = self.data.close[0] > self.data.open[0]

        # Current close > previous open
        close_above_prev_open = self.data.close[0] > self.data.open[-1]

        # Current open <= previous close
        open_below_prev_close = self.data.open[0] <= self.data.close[-1]

        bullish_engulfing = prev_bearish and curr_bullish and close_above_prev_open and open_below_prev_close
        rsi_above_threshold = self.data1[0] > self.p.rsi_threshold

        # Set indicator value (0 for False, high price for True)
        if bullish_engulfing and rsi_above_threshold:
            self.lines.purple_circle[0] = self.data.high[0]  # Plot at the top of candle
        else:
            self.lines.purple_circle[0] = 0


def run_backtest(data_path=None):
    """Run the backtest"""
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(BullishEngulfingStrategy)

    # Set initial capital
    cerebro.broker.setcash(10000.0)

    # Set commission (0.1%)
    cerebro.broker.setcommission(commission=0.001)

    # Add data
    if data_path:
        # Load data from CSV file
        data = bt.feeds.YahooFinanceCSVData(
            dataname=data_path,
            # Yahoo data format
            fromdate=datetime.datetime(2024, 4, 1),
            todate=datetime.datetime(2025, 4, 1),
            reverse=False,
        )
        cerebro.adddata(data)
    else:
        # Use sample data
        data = bt.feeds.YahooFinanceData(
            dataname="AAPL", fromdate=datetime.datetime(2024, 4, 1), todate=datetime.datetime(2025, 4, 1), reverse=False
        )
        cerebro.adddata(data)

    # Print starting portfolio value
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run the backtest
    cerebro.run()

    # Print final portfolio value
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Plot the results
    cerebro.plot(style="candlestick", barup="green", bardown="red")


if __name__ == "__main__":
    # Example usage
    run_backtest()
