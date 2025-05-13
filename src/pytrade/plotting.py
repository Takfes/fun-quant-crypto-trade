import pandas as pd
import plotly.graph_objects as go


def plot_candlestick_chart(df, title="Candlestick Chart"):
    """
    Plots a candlestick chart using Plotly.

    Parameters:
    - df: DataFrame with 'open', 'high', 'low', 'close', and 'time' columns.
    - title: Title of the chart.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candlesticks",
            )
        ]
    )

    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False)

    fig.show()


def create_base_candlestick_chart(df, title="Candlestick Chart"):
    """
    Creates a base candlestick chart using Plotly.

    Parameters:
    - df: DataFrame with 'open', 'high', 'low', 'close', and 'time' columns.
    - title: Title of the chart.

    Returns:
    - A Plotly Figure object.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candlesticks",
            )
        ]
    )

    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False)
    return fig


def add_chart_to_figure(fig, chart):
    """
    Adds a new chart to an existing Plotly Figure.

    Parameters:
    - fig: The existing Plotly Figure object.
    - chart: A Plotly trace (e.g., Scatter, Bar) to add to the figure.

    Returns:
    - The updated Plotly Figure object.
    """
    fig.add_trace(chart)
    return fig
