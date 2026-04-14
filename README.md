# Streamlit Portfolio Analysis App

This app analyzes selected stocks and the S&P 500 benchmark using adjusted close prices from yfinance. It includes exploratory statistics, risk analytics, constrained portfolio optimization (no short selling), risk contribution, efficient frontier construction, custom portfolios, and lookback sensitivity analysis.

## What the App Does

- Validates 3 to 10 tickers and a minimum 2-year date range.
- Downloads adjusted close prices for selected stocks and `^GSPC` benchmark.
- Computes simple daily returns and annualized metrics.
- Builds summary statistics, distribution plots, rolling risk plots, drawdown, and risk-adjusted metrics.
- Produces correlation heatmaps, rolling correlations, and covariance tables.
- Constructs equal-weight, GMV, tangency, and custom portfolios.
- Computes and displays risk contribution for GMV and tangency portfolios.
- Computes the true constrained efficient frontier and plots the Capital Allocation Line.
- Compares portfolio wealth paths and metrics versus the benchmark.
- Tests optimization sensitivity across feasible lookback windows.
