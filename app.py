from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize


# ==============================
# Constants
# ==============================
TRADING_DAYS = 252
BENCHMARK_TICKER = "^GSPC"
MIN_TICKERS = 3
MAX_TICKERS = 10
MAX_ALLOWED_MISSING = 0.05


st.set_page_config(page_title="Portfolio Analysis App", layout="wide")


# ==============================
# Basic Helpers
# ==============================
def parse_tickers(raw_text: str) -> List[str]:
    cleaned = [t.strip().upper() for t in raw_text.split(",") if t.strip()]
    out: List[str] = []
    seen = set()
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _as_series(values: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected a one-dimensional series-like input.")
        return values.iloc[:, 0]
    return values


def annualized_return(series: pd.Series | pd.DataFrame) -> float:
    series = _as_series(series)
    return float(series.mean() * TRADING_DAYS)


def annualized_volatility(series: pd.Series | pd.DataFrame) -> float:
    series = _as_series(series)
    return float(series.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(series: pd.Series | pd.DataFrame, rf_annual: float) -> float:
    series = _as_series(series)
    daily_rf = rf_annual / TRADING_DAYS
    excess = series - daily_rf
    denom = series.std()
    if denom <= 0 or np.isnan(denom):
        return np.nan
    return float((excess.mean() / denom) * np.sqrt(TRADING_DAYS))


def sortino_ratio(series: pd.Series | pd.DataFrame, rf_annual: float) -> float:
    series = _as_series(series)
    daily_rf = rf_annual / TRADING_DAYS
    downside = np.minimum(0.0, series - daily_rf)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev <= 0 or np.isnan(downside_dev):
        return np.nan
    excess_mean = float((series - daily_rf).mean())
    return float((excess_mean / downside_dev) * np.sqrt(TRADING_DAYS))


def drawdown_dataframe(series: pd.Series | pd.DataFrame) -> pd.DataFrame:
    series = _as_series(series)
    wealth = (1 + series).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return pd.DataFrame({"wealth": wealth, "peak": peak, "drawdown": drawdown})


def maximum_drawdown(series: pd.Series | pd.DataFrame) -> float:
    series = _as_series(series)
    return float(drawdown_dataframe(series)["drawdown"].min())


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return pd.Series(returns.values @ weights, index=returns.index, name="portfolio")


def portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray, rf_annual: float) -> Dict[str, float]:
    p = portfolio_daily_returns(returns, weights)
    return {
        "Annualized Return": annualized_return(p),
        "Annualized Volatility": annualized_volatility(p),
        "Sharpe Ratio": sharpe_ratio(p, rf_annual),
        "Sortino Ratio": sortino_ratio(p, rf_annual),
        "Maximum Drawdown": maximum_drawdown(p),
    }


# ==============================
# Data Download and Validation
# ==============================
@st.cache_data(ttl=3600)
def download_single_adjusted_close(ticker: str, start_date: date, end_date: date) -> pd.Series:
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"{ticker}: download error: {exc}") from exc

    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: no data returned")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise ValueError(f"{ticker}: adjusted close data unavailable")
        extracted = raw["Adj Close"]
        if isinstance(extracted, pd.DataFrame):
            if ticker in extracted.columns:
                series = extracted[ticker].copy()
            elif extracted.shape[1] == 1:
                series = extracted.iloc[:, 0].copy()
            else:
                raise ValueError(f"{ticker}: adjusted close data unavailable")
        else:
            series = extracted.copy()
    elif "Adj Close" not in raw.columns:
        raise ValueError(f"{ticker}: adjusted close data unavailable")
    else:
        series = raw["Adj Close"].copy()

    series.name = ticker
    if series.dropna().empty:
        raise ValueError(f"{ticker}: adjusted close series is empty")
    return series


def download_adjusted_close(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
) -> Dict[str, object]:
    diagnostics = {
        "invalid_or_missing": [],
        "insufficient_data": [],
        "dropped_missing": [],
        "warnings": [],
        "notes": [],
    }

    price_frames: List[pd.Series] = []
    for t in tickers:
        try:
            series = download_single_adjusted_close(t, start_date, end_date)
        except Exception as exc:
            diagnostics["invalid_or_missing"].append(t)
            diagnostics["warnings"].append(str(exc))
            continue

        price_frames.append(series)

    try:
        benchmark_prices = download_single_adjusted_close(BENCHMARK_TICKER, start_date, end_date)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Benchmark (^GSPC) data is unavailable for the selected window: {exc}",
            "diagnostics": diagnostics,
        }

    if not price_frames:
        return {
            "success": False,
            "error": "None of the submitted stock tickers returned valid adjusted close prices.",
            "diagnostics": diagnostics,
        }

    stock_prices = pd.concat(price_frames, axis=1)

    if stock_prices.empty:
        return {
            "success": False,
            "error": "No valid stock price data could be assembled from the submitted tickers.",
            "diagnostics": diagnostics,
        }

    # Drop any duplicated columns defensively if a symbol was repeated in user input.
    stock_prices = stock_prices.loc[:, ~stock_prices.columns.duplicated()].copy()

    expected_rows = len(stock_prices)
    if expected_rows < 2:
        return {
            "success": False,
            "error": "Insufficient observations for the selected period.",
            "diagnostics": diagnostics,
        }

    valid_tickers: List[str] = list(stock_prices.columns)

    if stock_prices.shape[1] == 0:
        return {
            "success": False,
            "error": "No valid stock tickers remained after download.",
            "diagnostics": diagnostics,
        }

    insufficient = [
        t
        for t in stock_prices.columns
        if stock_prices[t].notna().sum() < int(0.8 * expected_rows)
    ]
    if insufficient:
        diagnostics["insufficient_data"].extend(insufficient)

    missing_ratio = stock_prices.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MAX_ALLOWED_MISSING].index.tolist()
    if drop_cols:
        diagnostics["dropped_missing"].extend(drop_cols)
        stock_prices = stock_prices.drop(columns=drop_cols)
        diagnostics["warnings"].append(
            "Dropped tickers with more than 5% missing adjusted close values: " + ", ".join(drop_cols)
        )

    if stock_prices.shape[1] < MIN_TICKERS:
        return {
            "success": False,
            "error": "Fewer than 3 valid tickers remain after data-quality filtering.",
            "diagnostics": diagnostics,
        }

    rows_before_overlap = len(stock_prices)
    stock_prices = stock_prices.dropna(how="any")
    if len(stock_prices) == 0:
        return {
            "success": False,
            "error": "No overlapping stock dates remain after alignment.",
            "diagnostics": diagnostics,
        }
    if len(stock_prices) < rows_before_overlap:
        diagnostics["notes"].append(
            "Stock prices were truncated to overlapping dates across selected tickers."
        )

    common_index = stock_prices.index.intersection(benchmark_prices.dropna().index)
    if len(common_index) < 2:
        return {
            "success": False,
            "error": "Insufficient overlapping dates between stock prices and benchmark.",
            "diagnostics": diagnostics,
        }

    if len(common_index) < len(stock_prices):
        diagnostics["notes"].append(
            "Data was aligned to dates shared by stocks and benchmark."
        )

    stock_prices = stock_prices.loc[common_index].sort_index()
    benchmark_prices = benchmark_prices.loc[common_index].sort_index()

    if len(stock_prices) < 400:
        diagnostics["warnings"].append(
            "Aligned sample has limited observations. Optimization outputs can be unstable."
        )

    return {
        "success": True,
        "stock_prices": stock_prices,
        "benchmark_prices": benchmark_prices,
        "diagnostics": diagnostics,
    }


@st.cache_data(ttl=3600)
def compute_returns(stock_prices: pd.DataFrame, benchmark_prices: pd.Series) -> Dict[str, pd.DataFrame | pd.Series]:
    stock_returns = stock_prices.pct_change().dropna(how="any")
    benchmark_prices = _as_series(benchmark_prices)
    benchmark_returns = benchmark_prices.pct_change().dropna()
    idx = stock_returns.index.intersection(benchmark_returns.index)
    stock_returns = stock_returns.loc[idx]
    benchmark_returns = benchmark_returns.loc[idx]
    combined_returns = stock_returns.copy()
    combined_returns[BENCHMARK_TICKER] = benchmark_returns
    return {
        "stock_returns": stock_returns,
        "benchmark_returns": benchmark_returns,
        "combined_returns": combined_returns,
    }


# ==============================
# Summary and Risk Tables
# ==============================
@st.cache_data(ttl=3600)
def summary_statistics(returns_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=returns_df.columns)
    out["Annualized Mean Return"] = returns_df.mean() * TRADING_DAYS
    out["Annualized Volatility"] = returns_df.std() * np.sqrt(TRADING_DAYS)
    out["Skewness"] = returns_df.skew()
    out["Kurtosis"] = returns_df.kurtosis()
    out["Minimum Daily Return"] = returns_df.min()
    out["Maximum Daily Return"] = returns_df.max()
    return out


def risk_adjusted_table(returns_df: pd.DataFrame, rf_annual: float) -> pd.DataFrame:
    rows = []
    for col in returns_df.columns:
        s = returns_df[col].dropna()
        rows.append(
            {
                "Asset": col,
                "Sharpe Ratio": sharpe_ratio(s, rf_annual),
                "Sortino Ratio": sortino_ratio(s, rf_annual),
            }
        )
    return pd.DataFrame(rows).set_index("Asset")


# ==============================
# Optimization Helpers
# ==============================
def optimize_gmv(returns: pd.DataFrame) -> Optional[np.ndarray]:
    n = returns.shape[1]
    cov = returns.cov().values
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w: np.ndarray) -> float:
        # Portfolio variance: w^T Sigma w
        return float(w.T @ cov @ w)

    result = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return None
    return result.x


def optimize_tangency(returns: pd.DataFrame, rf_annual: float) -> Optional[np.ndarray]:
    n = returns.shape[1]
    cov_daily = returns.cov().values
    mu_ann = returns.mean().values * TRADING_DAYS
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w: np.ndarray) -> float:
        p_ret = float(w @ mu_ann)
        p_vol = float(np.sqrt(w.T @ cov_daily @ w) * np.sqrt(TRADING_DAYS))
        if p_vol <= 0 or np.isnan(p_vol):
            return 1e6
        return -((p_ret - rf_annual) / p_vol)

    result = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return None
    return result.x


@st.cache_data
def cached_gmv(returns):
    return optimize_gmv(returns)


@st.cache_data
def cached_tangency(returns, rf_annual):
    return optimize_tangency(returns, rf_annual)


def percentage_risk_contribution(weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    cov = cov_matrix.values
    sigma2 = float(weights.T @ cov @ weights)
    if sigma2 <= 0 or np.isnan(sigma2):
        return np.full(shape=weights.shape, fill_value=np.nan)
    # PRC_i = w_i * (Sigma w)_i / sigma_p^2
    sigma_w = cov @ weights
    return weights * sigma_w / sigma2


@st.cache_data(ttl=3600)
def constrained_efficient_frontier(returns: pd.DataFrame, num_points: int = 50) -> pd.DataFrame:
    """True frontier from constrained optimization over target returns."""
    n = returns.shape[1]
    mu_ann = returns.mean().values * TRADING_DAYS
    cov_daily = returns.cov().values

    min_target = float(mu_ann.min())
    max_target = float(mu_ann.max())
    targets = np.linspace(min_target, max_target, num_points)

    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    rows: List[Dict[str, float]] = []

    def var_obj(w: np.ndarray) -> float:
        return float(w.T @ cov_daily @ w)

    for t_ret in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=t_ret: float(w @ mu_ann) - target},
        )
        res = minimize(var_obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            vol = float(np.sqrt(res.x.T @ cov_daily @ res.x) * np.sqrt(TRADING_DAYS))
            rows.append({"Return": float(t_ret), "Volatility": vol})

    return pd.DataFrame(rows)


def format_weights(weights: np.ndarray, tickers: List[str]) -> str:
    pairs = [f"{t}:{w:.2%}" for t, w in zip(tickers, weights)]
    return " | ".join(pairs)


@st.cache_data(ttl=3600)
def sensitivity_analysis(
    returns: pd.DataFrame,
    rf_annual: float,
    windows: Tuple[Tuple[str, int], ...],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    weights_rows = []

    for label, lookback in windows:
        subset = returns.tail(lookback) if lookback > 0 else returns.copy()
        gmv = optimize_gmv(subset)
        tan = optimize_tangency(subset, rf_annual)

        if gmv is None or tan is None:
            rows.append(
                {
                    "Window": label,
                    "GMV Weights": "Optimization failed",
                    "GMV Return": np.nan,
                    "GMV Volatility": np.nan,
                    "Tangency Weights": "Optimization failed",
                    "Tangency Return": np.nan,
                    "Tangency Volatility": np.nan,
                    "Tangency Sharpe": np.nan,
                    "Status": "Optimization failed",
                }
            )
            continue

        gmv_metrics = portfolio_metrics(subset, gmv, rf_annual)
        tan_metrics = portfolio_metrics(subset, tan, rf_annual)
        rows.append(
            {
                "Window": label,
                "GMV Weights": format_weights(gmv, list(subset.columns)),
                "GMV Return": gmv_metrics["Annualized Return"],
                "GMV Volatility": gmv_metrics["Annualized Volatility"],
                "Tangency Weights": format_weights(tan, list(subset.columns)),
                "Tangency Return": tan_metrics["Annualized Return"],
                "Tangency Volatility": tan_metrics["Annualized Volatility"],
                "Tangency Sharpe": tan_metrics["Sharpe Ratio"],
                "Status": "OK",
            }
        )

        for ticker, w in zip(subset.columns, gmv):
            weights_rows.append(
                {
                    "Window": label,
                    "Portfolio": "GMV",
                    "Ticker": ticker,
                    "Weight": float(w),
                }
            )
        for ticker, w in zip(subset.columns, tan):
            weights_rows.append(
                {
                    "Window": label,
                    "Portfolio": "Tangency",
                    "Ticker": ticker,
                    "Weight": float(w),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(weights_rows)


# ==============================
# UI Input Section
# ==============================
st.title("Portfolio Analysis and Optimization")
st.caption("Adjusted close data, simple arithmetic returns, and constrained mean-variance optimization.")


with st.sidebar:
    st.header("Input Configuration")
    raw_tickers = st.text_input(
        "Stock tickers (3 to 10, comma-separated)",
        value="AAPL, MSFT, NVDA, AMZN",
    )

    today = date.today()
    default_start = date(today.year - 5, today.month, min(today.day, 28))
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=today)
    risk_free_pct = st.number_input(
        "Annualized risk-free rate (%)",
        min_value=-5.0,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )
    run_analysis = st.button("Fetch data and run analysis", type="primary")


def validate_user_inputs(tickers: List[str], start: date, end: date) -> Optional[str]:
    if len(tickers) < MIN_TICKERS:
        return "Please provide at least 3 ticker symbols."
    if len(tickers) > MAX_TICKERS:
        return "Please provide no more than 10 ticker symbols."
    if end <= start:
        return "End date must be after start date."
    if (end - start).days < 730:
        return "Please select a minimum date range of 2 years."
    return None


tabs = st.tabs(
    [
        "Data and Inputs",
        "Exploratory and Risk",
        "Correlation and Covariance",
        "Portfolio Optimization",
        "Sensitivity",
        "About and Methodology",
    ]
)


# ==============================
# Data Pipeline Trigger
# ==============================
if run_analysis:
    parsed_tickers = parse_tickers(raw_tickers)
    input_error = validate_user_inputs(parsed_tickers, start_date, end_date)
    if input_error:
        st.session_state["analysis_bundle"] = None
        st.error(input_error)
    else:
        with st.spinner("Downloading and validating adjusted close prices..."):
            dl = download_adjusted_close(tuple(parsed_tickers), start_date, end_date)

        if not dl.get("success", False):
            st.session_state["analysis_bundle"] = None
            st.error(str(dl.get("error", "Unknown data download error.")))
            diag = dl.get("diagnostics", {})
            bad = diag.get("invalid_or_missing", [])
            insuff = diag.get("insufficient_data", [])
            if bad:
                st.error("Invalid or unavailable tickers: " + ", ".join(bad))
            if insuff:
                st.error("Tickers with insufficient data coverage: " + ", ".join(insuff))
        else:
            with st.spinner("Computing return series and cached analytics inputs..."):
                r = compute_returns(dl["stock_prices"], dl["benchmark_prices"])

            st.session_state["analysis_bundle"] = {
                "tickers": list(dl["stock_prices"].columns),
                "stock_prices": dl["stock_prices"],
                "benchmark_prices": dl["benchmark_prices"],
                "stock_returns": r["stock_returns"],
                "benchmark_returns": r["benchmark_returns"],
                "combined_returns": r["combined_returns"],
                "diagnostics": dl["diagnostics"],
                "start_date": start_date,
                "end_date": end_date,
            }

            # Strictly surface partial-data issues as explicit user-facing errors/warnings.
            diag = dl["diagnostics"]
            if diag.get("invalid_or_missing"):
                st.error("Invalid or unavailable tickers: " + ", ".join(diag["invalid_or_missing"]))
            if diag.get("insufficient_data"):
                st.error("Tickers with insufficient data coverage: " + ", ".join(diag["insufficient_data"]))
            for w in diag.get("warnings", []):
                st.warning(w)
            for note in diag.get("notes", []):
                st.info(note)


bundle = st.session_state.get("analysis_bundle")
if bundle is None:
    with tabs[0]:
        st.info("Configure inputs in the sidebar and click 'Fetch data and run analysis'.")
        st.write("This app requires 3 to 10 valid tickers and at least 2 years of data.")
    with tabs[5]:
        st.subheader("About this app")
        st.write("The app performs exploratory analysis, risk analytics, and constrained portfolio optimization.")
    st.stop()


# Keep risk-free value dynamic without forcing re-download.
rf_annual = risk_free_pct / 100.0

tickers: List[str] = bundle["tickers"]
stock_prices: pd.DataFrame = bundle["stock_prices"]
benchmark_prices: pd.Series = bundle["benchmark_prices"]
stock_returns: pd.DataFrame = bundle["stock_returns"]
benchmark_returns: pd.Series = bundle["benchmark_returns"]
combined_returns: pd.DataFrame = bundle["combined_returns"]
diagnostics: Dict[str, List[str]] = bundle["diagnostics"]


# ==============================
# Tab 1: Data and Inputs
# ==============================
with tabs[0]:
    st.header("Data Configuration and Validation")
    st.write(
        f"Selected window: **{bundle['start_date']}** to **{bundle['end_date']}**. "
        f"Risk-free rate: **{rf_annual:.2%}** annualized."
    )
    st.write("Valid tickers used in the analysis:", ", ".join(tickers))

    if diagnostics.get("invalid_or_missing"):
        st.error("Invalid or unavailable tickers: " + ", ".join(diagnostics["invalid_or_missing"]))
    if diagnostics.get("insufficient_data"):
        st.error("Tickers with insufficient data coverage: " + ", ".join(diagnostics["insufficient_data"]))
    for w in diagnostics.get("warnings", []):
        st.warning(w)
    for n in diagnostics.get("notes", []):
        st.info(n)

    st.subheader("Summary Statistics")
    stats_df = summary_statistics(combined_returns)
    st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Cumulative Wealth Index (Initial $10,000)")
    selected_wealth_tickers = st.multiselect(
        "Select stocks for wealth chart",
        options=tickers,
        default=tickers,
    )

    wealth = pd.DataFrame(index=combined_returns.index)
    for t in selected_wealth_tickers:
        wealth[t] = (1 + combined_returns[t]).cumprod() * 10000
    wealth[BENCHMARK_TICKER] = (1 + combined_returns[BENCHMARK_TICKER]).cumprod() * 10000

    fig_wealth = go.Figure()
    for col in wealth.columns:
        fig_wealth.add_trace(go.Scatter(x=wealth.index, y=wealth[col], mode="lines", name=col))
    fig_wealth.update_layout(
        title="Growth of $10,000 in Selected Stocks and S&P 500",
        xaxis_title="Date",
        yaxis_title="Value ($)",
    )
    st.plotly_chart(fig_wealth, use_container_width=True)


# ==============================
# Tab 2: Exploratory and Risk
# ==============================
with tabs[1]:
    st.header("Exploratory Analysis and Risk")

    st.subheader("Distribution Plot")
    dist_ticker = st.selectbox("Select stock", options=tickers, key="dist_ticker")
    dist_view = st.radio(
        "Distribution view",
        options=["Histogram with fitted normal curve", "Q-Q plot against normal"],
        horizontal=True,
    )
    dist_series = stock_returns[dist_ticker].dropna()

    if dist_view == "Histogram with fitted normal curve":
        mu = float(dist_series.mean())
        sigma = float(dist_series.std())
        x_vals = np.linspace(float(dist_series.min()), float(dist_series.max()), 300)
        y_vals = stats.norm.pdf(x_vals, loc=mu, scale=sigma) if sigma > 0 else np.zeros_like(x_vals)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=dist_series,
                histnorm="probability density",
                nbinsx=50,
                name="Daily return histogram",
                opacity=0.75,
            )
        )
        fig_hist.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Fitted normal curve"))
        fig_hist.update_layout(
            title=f"Daily Return Distribution: {dist_ticker}",
            xaxis_title="Daily Return",
            yaxis_title="Density",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        q_theoretical, q_sample = stats.probplot(dist_series, dist="norm", fit=False)
        slope, intercept, _, _, _ = stats.linregress(q_theoretical, q_sample)
        ref_line = slope * np.array(q_theoretical) + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=q_theoretical, y=q_sample, mode="markers", name="Sample quantiles"))
        fig_qq.add_trace(go.Scatter(x=q_theoretical, y=ref_line, mode="lines", name="Reference line"))
        fig_qq.update_layout(
            title=f"Q-Q Plot: {dist_ticker} Daily Returns",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    st.subheader("Rolling Annualized Volatility")
    vol_window = st.select_slider("Rolling window (days)", options=[30, 60, 90, 120], value=60)
    rolling_vol = stock_returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)

    fig_roll_vol = go.Figure()
    for c in rolling_vol.columns:
        fig_roll_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[c], mode="lines", name=c))
    fig_roll_vol.update_layout(
        title=f"Rolling Annualized Volatility ({vol_window}-Day Window)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
    )
    st.plotly_chart(fig_roll_vol, use_container_width=True)

    st.subheader("Drawdown")
    drawdown_ticker = st.selectbox("Select stock for drawdown", options=tickers, key="drawdown_ticker")
    dd_df = drawdown_dataframe(stock_returns[drawdown_ticker].dropna())
    st.metric("Maximum Drawdown", f"{float(dd_df['drawdown'].min()):.2%}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_df.index, y=dd_df["drawdown"], mode="lines", name="Drawdown"))
    fig_dd.update_layout(
        title=f"Drawdown Chart: {drawdown_ticker}",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Risk-adjusted Metrics")
    risk_df = risk_adjusted_table(combined_returns, rf_annual)
    st.dataframe(risk_df.style.format("{:.4f}"), use_container_width=True)


# ==============================
# Tab 3: Correlation and Covariance
# ==============================
with tabs[2]:
    st.header("Correlation and Covariance")

    st.subheader("Correlation Heatmap (Daily Returns)")
    corr = stock_returns.corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text:.2f}",
            colorbar=dict(title="Correlation"),
        )
    )
    fig_corr.update_layout(
        title="Correlation Heatmap with Cell Annotations",
        xaxis_title="Assets",
        yaxis_title="Assets",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Rolling Correlation Between Two Stocks")
    col_a, col_b = st.columns(2)
    with col_a:
        stock_a = st.selectbox("Stock A", options=tickers, key="stock_a")
    with col_b:
        default_idx = 1 if len(tickers) > 1 else 0
        stock_b = st.selectbox("Stock B", options=tickers, index=default_idx, key="stock_b")

    corr_window = st.select_slider(
        "Rolling correlation window (days)", options=[30, 60, 90, 120], value=60, key="roll_corr_window"
    )

    if stock_a == stock_b:
        st.warning("Select two different tickers to compute rolling correlation.")
    else:
        rolling_corr = stock_returns[stock_a].rolling(corr_window).corr(stock_returns[stock_b])
        fig_roll_corr = go.Figure()
        fig_roll_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name="Rolling Corr"))
        fig_roll_corr.update_layout(
            title=f"Rolling Correlation: {stock_a} vs {stock_b} ({corr_window} Days)",
            xaxis_title="Date",
            yaxis_title="Correlation",
        )
        st.plotly_chart(fig_roll_corr, use_container_width=True)

    with st.expander("Covariance Matrix (Daily Returns)"):
        cov_matrix = stock_returns.cov()
        st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)


# ==============================
# Tab 4: Portfolio Optimization
# ==============================
with tabs[3]:
    st.header("Portfolio Construction and Optimization")

    n_assets = len(tickers)
    eq_weights = np.repeat(1 / n_assets, n_assets)
    eq_metrics = portfolio_metrics(stock_returns, eq_weights, rf_annual)
    eq_port_rets = portfolio_daily_returns(stock_returns, eq_weights)

    with st.spinner("Running GMV and tangency optimization with no-short constraints..."):
        gmv_weights = cached_gmv(stock_returns)
        tan_weights = cached_tangency(stock_returns, rf_annual)

    if gmv_weights is None:
        st.error("GMV optimization failed. Try changing tickers or date range.")
    if tan_weights is None:
        st.error("Tangency optimization failed. Try changing tickers or date range.")

    st.subheader("Custom Portfolio")
    st.write("Set raw slider values. The app normalizes them so normalized weights sum to 1.")
    slider_cols = st.columns(min(5, n_assets))
    ticker_signature = "_".join(tickers)
    raw_custom_values: Dict[str, int] = {}
    for i, t in enumerate(tickers):
        with slider_cols[i % len(slider_cols)]:
            raw_custom_values[t] = st.slider(
                f"{t}",
                0,
                100,
                int(round(100 / n_assets)),
                key=f"custom_{ticker_signature}_{t}",
            )

    raw_custom = np.array([raw_custom_values[t] for t in tickers], dtype=float)
    if raw_custom.sum() <= 0:
        st.warning("All custom sliders are zero. Falling back to equal weights.")
        custom_weights = eq_weights.copy()
    else:
        custom_weights = raw_custom / raw_custom.sum()

    custom_weight_df = pd.DataFrame({"Ticker": tickers, "Normalized Weight": custom_weights}).set_index("Ticker")
    st.dataframe(custom_weight_df.style.format("{:.2%}"), use_container_width=True)

    custom_metrics = portfolio_metrics(stock_returns, custom_weights, rf_annual)
    custom_port_rets = portfolio_daily_returns(stock_returns, custom_weights)

    metric_rows = [
        {"Portfolio": "Equal Weight", **eq_metrics},
        {"Portfolio": "Custom", **custom_metrics},
    ]

    if gmv_weights is not None:
        gmv_metrics = portfolio_metrics(stock_returns, gmv_weights, rf_annual)
        gmv_port_rets = portfolio_daily_returns(stock_returns, gmv_weights)
        metric_rows.append({"Portfolio": "GMV", **gmv_metrics})
    else:
        gmv_metrics = None
        gmv_port_rets = None

    if tan_weights is not None:
        tan_metrics = portfolio_metrics(stock_returns, tan_weights, rf_annual)
        tan_port_rets = portfolio_daily_returns(stock_returns, tan_weights)
        metric_rows.append({"Portfolio": "Tangency", **tan_metrics})
    else:
        tan_metrics = None
        tan_port_rets = None

    st.subheader("Portfolio Metrics Summary")
    metrics_df = pd.DataFrame(metric_rows).set_index("Portfolio")
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Optimal Weights")
    weight_table = pd.DataFrame({"Ticker": tickers, "Equal Weight": eq_weights, "Custom": custom_weights})
    if gmv_weights is not None:
        weight_table["GMV"] = gmv_weights
    if tan_weights is not None:
        weight_table["Tangency"] = tan_weights
    st.dataframe(weight_table.set_index("Ticker").style.format("{:.2%}"), use_container_width=True)

    st.subheader("Weights Chart")
    weights_long = weight_table.melt(id_vars="Ticker", var_name="Portfolio", value_name="Weight")
    fig_weights = px.bar(weights_long, x="Ticker", y="Weight", color="Portfolio", barmode="group")
    fig_weights.update_layout(title="Portfolio Weights by Strategy", xaxis_title="Ticker", yaxis_title="Weight")
    st.plotly_chart(fig_weights, use_container_width=True)

    if gmv_weights is not None or tan_weights is not None:
        st.subheader("Percentage Risk Contribution (PRC)")
        st.info(
            "Risk contribution can differ from weight. A stock can contribute more risk than its weight if covariance is high. For example, a stock with a 10% portfolio weight but 25% risk contribution is a disproportionate source of portfolio volatility."
        )
        cov_daily = stock_returns.cov()

        if gmv_weights is not None:
            gmv_prc = percentage_risk_contribution(gmv_weights, cov_daily)
            gmv_prc_df = pd.DataFrame({"Ticker": tickers, "Weight": gmv_weights, "PRC": gmv_prc})
            st.write("GMV PRC Table")
            st.dataframe(gmv_prc_df.style.format({"Weight": "{:.2%}", "PRC": "{:.2%}"}), use_container_width=True)
            st.write(f"PRC sum (GMV): {np.nansum(gmv_prc):.6f}")
            fig_gmv_prc = px.bar(gmv_prc_df, x="Ticker", y="PRC", title="GMV Percentage Risk Contribution")
            fig_gmv_prc.update_layout(xaxis_title="Ticker", yaxis_title="PRC")
            st.plotly_chart(fig_gmv_prc, use_container_width=True)

        if tan_weights is not None:
            tan_prc = percentage_risk_contribution(tan_weights, cov_daily)
            tan_prc_df = pd.DataFrame({"Ticker": tickers, "Weight": tan_weights, "PRC": tan_prc})
            st.write("Tangency PRC Table")
            st.dataframe(tan_prc_df.style.format({"Weight": "{:.2%}", "PRC": "{:.2%}"}), use_container_width=True)
            st.write(f"PRC sum (Tangency): {np.nansum(tan_prc):.6f}")
            fig_tan_prc = px.bar(tan_prc_df, x="Ticker", y="PRC", title="Tangency Percentage Risk Contribution")
            fig_tan_prc.update_layout(xaxis_title="Ticker", yaxis_title="PRC")
            st.plotly_chart(fig_tan_prc, use_container_width=True)

    st.subheader("Efficient Frontier and CAL")
    with st.spinner("Computing constrained efficient frontier..."):
        frontier = constrained_efficient_frontier(stock_returns, num_points=60)

    fig_frontier = go.Figure()
    if not frontier.empty:
        fig_frontier.add_trace(
            go.Scatter(
                x=frontier["Volatility"],
                y=frontier["Return"],
                mode="lines",
                name="Efficient Frontier",
            )
        )
    else:
        st.warning("Efficient frontier could not be computed for enough target returns.")

    # Individual stocks
    indiv = pd.DataFrame(
        {
            "Ticker": tickers,
            "Return": stock_returns.mean().values * TRADING_DAYS,
            "Volatility": stock_returns.std().values * np.sqrt(TRADING_DAYS),
        }
    )
    fig_frontier.add_trace(
        go.Scatter(
            x=indiv["Volatility"],
            y=indiv["Return"],
            mode="markers+text",
            text=indiv["Ticker"],
            textposition="top center",
            name="Individual Stocks",
            marker=dict(size=9),
        )
    )

    # Benchmark point for comparison only
    bench_ret = annualized_return(benchmark_returns)
    bench_vol = annualized_volatility(benchmark_returns)
    fig_frontier.add_trace(
        go.Scatter(
            x=[bench_vol],
            y=[bench_ret],
            mode="markers+text",
            text=[BENCHMARK_TICKER],
            textposition="bottom center",
            name="Benchmark",
            marker=dict(size=10, symbol="diamond"),
        )
    )

    def add_portfolio_point(name: str, m: Dict[str, float], symbol: str) -> None:
        fig_frontier.add_trace(
            go.Scatter(
                x=[m["Annualized Volatility"]],
                y=[m["Annualized Return"]],
                mode="markers+text",
                text=[name],
                textposition="top right",
                name=name,
                marker=dict(size=11, symbol=symbol),
            )
        )

    add_portfolio_point("Equal Weight", eq_metrics, "circle")
    add_portfolio_point("Custom", custom_metrics, "square")
    if gmv_metrics is not None:
        add_portfolio_point("GMV", gmv_metrics, "triangle-up")
    if tan_metrics is not None:
        add_portfolio_point("Tangency", tan_metrics, "star")

    # Capital Allocation Line: rf + Sharpe_tangency * sigma
    if tan_metrics is not None and np.isfinite(tan_metrics["Sharpe Ratio"]):
        x_max = max(
            [
                float(frontier["Volatility"].max()) if not frontier.empty else 0.0,
                float(indiv["Volatility"].max()),
                float(bench_vol),
                float(tan_metrics["Annualized Volatility"]),
            ]
        )
        x_line = np.linspace(0.0, x_max * 1.1, 200)
        y_line = rf_annual + tan_metrics["Sharpe Ratio"] * x_line
        fig_frontier.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Capital Allocation Line",
                line=dict(dash="dash"),
            )
        )

    fig_frontier.update_layout(
        title="Efficient Frontier in Mean-Volatility Space",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    st.write(
        "The efficient frontier is solved by minimizing variance at each target return under no-short constraints. "
        "The Capital Allocation Line starts at the risk-free rate and passes through the tangency portfolio."
    )

    st.subheader("Portfolio Comparison: Cumulative Wealth")
    comparison_wealth = pd.DataFrame(index=stock_returns.index)
    comparison_wealth["Equal Weight"] = (1 + eq_port_rets).cumprod() * 10000
    comparison_wealth["Custom"] = (1 + custom_port_rets).cumprod() * 10000
    comparison_wealth["Benchmark (^GSPC)"] = (1 + benchmark_returns.loc[stock_returns.index]).cumprod() * 10000
    if gmv_port_rets is not None:
        comparison_wealth["GMV"] = (1 + gmv_port_rets).cumprod() * 10000
    if tan_port_rets is not None:
        comparison_wealth["Tangency"] = (1 + tan_port_rets).cumprod() * 10000

    fig_compare = go.Figure()
    for col in comparison_wealth.columns:
        fig_compare.add_trace(go.Scatter(x=comparison_wealth.index, y=comparison_wealth[col], mode="lines", name=col))
    fig_compare.update_layout(
        title="Cumulative Wealth Comparison (Initial $10,000)",
        xaxis_title="Date",
        yaxis_title="Value ($)",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Portfolio Comparison Table")
    comparison_rows = [
        {"Portfolio": "Equal Weight", **eq_metrics},
        {"Portfolio": "Custom", **custom_metrics},
        {
            "Portfolio": "Benchmark (^GSPC)",
            "Annualized Return": annualized_return(benchmark_returns.loc[stock_returns.index]),
            "Annualized Volatility": annualized_volatility(benchmark_returns.loc[stock_returns.index]),
            "Sharpe Ratio": sharpe_ratio(benchmark_returns.loc[stock_returns.index], rf_annual),
            "Sortino Ratio": sortino_ratio(benchmark_returns.loc[stock_returns.index], rf_annual),
            "Maximum Drawdown": maximum_drawdown(benchmark_returns.loc[stock_returns.index]),
        },
    ]
    if gmv_metrics is not None:
        comparison_rows.append({"Portfolio": "GMV", **gmv_metrics})
    if tan_metrics is not None:
        comparison_rows.append({"Portfolio": "Tangency", **tan_metrics})

    comparison_table = pd.DataFrame(comparison_rows).set_index("Portfolio")
    st.dataframe(comparison_table.style.format("{:.4f}"), use_container_width=True)


# ==============================
# Tab 5: Sensitivity
# ==============================
with tabs[4]:
    st.header("Estimation Window Sensitivity")

    n_obs = len(stock_returns)

    # Ensure at least three feasible options whenever possible with 2y minimum input.
    candidates = [
        ("Trailing 1 Year", TRADING_DAYS),
        ("Trailing 2 Years", 2 * TRADING_DAYS),
        ("Trailing 3 Years", 3 * TRADING_DAYS),
        ("Trailing 5 Years", 5 * TRADING_DAYS),
    ]
    feasible = [(lbl, win) for lbl, win in candidates if n_obs >= win]
    feasible.append(("Full Sample", 0))

    st.write("Only lookback windows feasible for your selected date range are shown.")

    with st.spinner("Running optimization across lookback windows..."):
        sens_table, sens_weights_long = sensitivity_analysis(stock_returns, rf_annual, tuple(feasible))

    if sens_table.empty:
        st.warning("Sensitivity analysis could not be generated.")
    else:
        st.subheader("Sensitivity Comparison Table")
        st.dataframe(sens_table, use_container_width=True)

    if not sens_weights_long.empty:
        st.subheader("Weights Across Lookback Windows")
        portfolio_choice = st.selectbox("Portfolio type", ["GMV", "Tangency"])
        plot_df = sens_weights_long[sens_weights_long["Portfolio"] == portfolio_choice]
        fig_sens = px.bar(
            plot_df,
            x="Window",
            y="Weight",
            color="Ticker",
            barmode="group",
            title=f"{portfolio_choice} Weights Across Lookback Windows",
        )
        fig_sens.update_layout(xaxis_title="Lookback Window", yaxis_title="Weight")
        st.plotly_chart(fig_sens, use_container_width=True)

    st.info(
        "Optimization is sensitive to estimated means and covariances, so weights and performance "
        "can change across estimation windows."
    )


# ==============================
# Tab 6: Methodology
# ==============================
with tabs[5]:
    st.header("About and Methodology")
    st.markdown(
        """
### Data and Returns
- Data source: yfinance adjusted close prices.
- Assets analyzed: user-selected stocks plus benchmark ^GSPC for comparison.
- Portfolio optimization excludes the benchmark.
- Return convention: simple daily arithmetic returns from `pct_change()`.

### Annualization and Risk-Free Rate
- Annualized return = mean daily return * 252.
- Annualized volatility = std daily return * sqrt(252).
- User enters annualized risk-free rate.
- Daily risk-free rate = annualized risk-free rate / 252.

### Risk Metrics
- Sharpe ratio uses excess daily return relative to daily risk-free rate.
- Sortino ratio uses downside deviation relative to daily risk-free rate.
- Drawdown is computed from cumulative wealth `(1 + r).cumprod()`.

### Optimization Setup
- No short selling.
- Bounds on each weight: 0 to 1.
- Equality constraint: sum(weights) = 1.
- GMV portfolio minimizes variance `w^T Sigma w`.
- Tangency portfolio maximizes Sharpe via minimizing negative Sharpe.
- Efficient frontier is solved via constrained optimization at target returns.
- CAL is drawn from risk-free rate through the tangency portfolio.

### Risk Contribution
- Percentage risk contribution for asset i:
  - PRC_i = w_i * (Sigma w)_i / sigma_p^2
- PRCs should sum to 1 up to numerical tolerance.

### Robustness and Deployment
- Expensive computations use `@st.cache_data(ttl=3600)`.
- The app includes input validation and user-friendly error handling for common mistakes.
"""
    )
