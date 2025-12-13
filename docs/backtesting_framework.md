# Backtesting Framework Guide

This document outlines the three core backtesting strategies available in the `Backtester` class.

## 1. Buy and Hold (`run_buy_and_hold`)

**Philosophy:** Passive investing.
**Mechanism:**
1.  **Initial Allocation:** Capital is allocated to assets based on initial weights on Day 1.
2.  **No Rebalancing:** The number of shares held for each asset remains constant throughout the simulation.
3.  **Drift:** Portfolio weights will drift over time as asset prices change (e.g., winners become a larger % of the portfolio).

**Use Case:** Benchmarking. This represents the "do nothing" alternative.

## 2. Fixed Allocation (`run_fixed_allocation`)

**Philosophy:** Systematic rebalancing to maintain a target risk profile.
**Mechanism:**
1.  **Periodic Rebalancing:** At specific intervals (e.g., Monthly), the portfolio is reset to the target weights.
2.  **Selling Winners / Buying Losers:** To return to target weights, the algorithm sells assets that have appreciated relative to the portfolio and buys those that have underperformed.
3.  **Constant Risk:** Keeps the portfolio aligned with the intended allocation strategy (e.g., 60/40).

**Use Case:** Benchmarking against standard rebalanced portfolios (e.g., S&P 500 / Bonds).

## 3. Walk-Forward Optimization (`run_walk_forward`)

**Philosophy:** Dynamic, adaptive strategy based on recent market conditions.
**Mechanism:**
1.  **Rolling Lookback:** At each rebalance date, the algorithm looks at a specific window of past data (e.g., past 252 days).
2.  **Optimization:** An optimizer function is called on this historical slice to determine the *optimal* weights for the *next* period.
3.  **Execution:** The portfolio is rebalanced to these new, dynamically generated weights.

**Use Case:** Testing active strategies (e.g., "Maximize Sharpe Ratio", "Minimum Variance") to see if they outperform static benchmarks out-of-sample.
