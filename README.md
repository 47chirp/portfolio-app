# Streamlit Portfolio Analysis App

This app analyzes selected stocks and the S&P 500 benchmark using adjusted close prices from yfinance. It includes exploratory statistics, risk analytics, constrained portfolio optimization (no short selling), risk contribution, efficient frontier construction, custom portfolios, and lookback sensitivity analysis.

## Run Locally

1. Install dependencies:

```bash
uv add streamlit yfinance pandas plotly scipy numpy
```

2. Start the app:

```bash
uv run streamlit run app.py
```

3. Open the local URL shown in the terminal, typically `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from your GitHub repo.
3. Set the main file path to `app.py`.
4. Ensure `requirements.txt` is in the project root.
5. Deploy.

Streamlit Cloud installs dependencies from `requirements.txt` automatically.

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
