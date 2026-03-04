"""
==============================================================================
  Moving Average Crossover — Backtesting Dashboard
  Quantitative Research Portfolio by Rhameyza Faiqo Susanto
==============================================================================
  A Streamlit-based interactive dashboard for backtesting a simple
  Moving Average Crossover strategy. Compares strategy performance
  against a passive Buy & Hold benchmark.
==============================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MA Crossover Backtest — Rhameyza Faiqo Susanto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Deep Blue & Black Elegant Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global overrides */
    .stApp {
        background: linear-gradient(145deg, #020817 0%, #0a1628 40%, #0d1f3c 100%);
        color: #c8d6e5;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020a18 0%, #081830 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #7dd3fc !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(8,24,48,0.85) 0%, rgba(15,35,65,0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(56,189,248,0.05);
    }

    div[data-testid="stMetric"] label {
        color: #7dd3fc !important;
        font-weight: 500;
        letter-spacing: 0.03em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0f2fe !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #e0f2fe !important;
        font-family: 'Inter', sans-serif;
    }

    /* Branding footer */
    .branding-bar {
        text-align: center;
        padding: 28px 0 12px 0;
        color: rgba(125, 211, 252, 0.45);
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 500;
        border-top: 1px solid rgba(56,189,248,0.08);
        margin-top: 40px;
    }

    /* Branding hero */
    .branding-hero {
        text-align: center;
        padding: 8px 0 4px 0;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: rgba(56,189,248,0.35);
        font-weight: 400;
    }

    /* Section dividers */
    .section-header {
        border-left: 3px solid #38bdf8;
        padding-left: 14px;
        margin-bottom: 8px;
    }

    /* Button style */
    .stButton > button {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        color: #e0f2fe;
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%);
        border-color: rgba(56,189,248,0.5);
        box-shadow: 0 0 20px rgba(56,189,248,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Deep-blue Plotly layout template
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG = "#020a18"
GRID_COLOR = "rgba(56,189,248,0.06)"
AXIS_COLOR = "rgba(200,214,229,0.35)"

def _base_layout(**overrides) -> dict:
    """Return a styled Plotly layout dict with deep-blue / black palette."""
    layout = dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color="#c8d6e5", size=13),
        title_font=dict(size=18, color="#e0f2fe"),
        legend=dict(
            bgcolor="rgba(8,24,48,0.75)",
            bordercolor="rgba(56,189,248,0.12)",
            borderwidth=1,
            font=dict(color="#c8d6e5", size=12),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR),
            title_font=dict(color="#7dd3fc"),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=AXIS_COLOR),
            title_font=dict(color="#7dd3fc"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0a1628",
            bordercolor="rgba(56,189,248,0.3)",
            font_color="#e0f2fe",
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    layout.update(overrides)
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — User Inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Backtest Parameters")
    st.markdown("---")

    ticker = st.text_input(
        "📌 Ticker Symbol",
        value="AAPL",
        help="Contoh: AAPL, NVDA, MSFT, BBCA.JK, TLKM.JK",
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "📅 Start Date",
            value=datetime.today() - timedelta(days=5 * 365),
        )
    with col_d2:
        end_date = st.date_input(
            "📅 End Date",
            value=datetime.today(),
        )

    st.markdown("---")
    st.markdown("### Moving Average Periods")

    fast_period = st.number_input(
        "⚡ Fast MA Period",
        min_value=2,
        max_value=500,
        value=50,
        step=1,
    )

    slow_period = st.number_input(
        "🐢 Slow MA Period",
        min_value=5,
        max_value=500,
        value=200,
        step=1,
    )

    st.markdown("---")
    run_backtest = st.button("🚀 Run Backtest", use_container_width=True)

    # Validation
    if fast_period >= slow_period:
        st.warning("⚠️ Fast MA period harus lebih kecil dari Slow MA period.")

    st.markdown("---")
    st.markdown(
        '<p class="branding-hero">Quantitative Research Portfolio<br>'
        "<b>Rhameyza Faiqo Susanto</b></p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Moving Average Crossover — Backtesting Dashboard")
st.markdown(
    '<p class="branding-hero">Quantitative Research Portfolio by '
    "<b>Rhameyza Faiqo Susanto</b></p>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core Backtesting Logic
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📡 Downloading market data …")
def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df

    # Flatten MultiIndex columns if necessary (yfinance >= 0.2.36 quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def run_strategy(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """
    Compute MA crossover signals, daily & cumulative returns.

    Signal logic
    ────────────
    • Position =  1  when Fast MA > Slow MA  (bullish)
    • Position = -1  when Fast MA < Slow MA  (bearish / short)
    • Strategy return = previous-day position × today's market return
    """
    data = df.copy()

    # ── Moving Averages ──────────────────────────────────────────────────
    data["Fast_MA"] = data["Close"].rolling(window=fast).mean()
    data["Slow_MA"] = data["Close"].rolling(window=slow).mean()

    # ── Signal: +1 (long) / -1 (short) ──────────────────────────────────
    data["Signal"] = 0
    data.loc[data["Fast_MA"] > data["Slow_MA"], "Signal"] = 1
    data.loc[data["Fast_MA"] <= data["Slow_MA"], "Signal"] = -1

    # ── Daily market return ──────────────────────────────────────────────
    data["Market_Return"] = data["Close"].pct_change()

    # ── Strategy return (signal is lagged by 1 day to avoid look-ahead) ─
    data["Strategy_Return"] = data["Signal"].shift(1) * data["Market_Return"]

    # ── Cumulative returns ───────────────────────────────────────────────
    data["Cumulative_Market"] = (1 + data["Market_Return"]).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

    # ── Drawdown series (strategy) ───────────────────────────────────────
    cum = data["Cumulative_Strategy"]
    running_max = cum.cummax()
    data["Drawdown"] = (cum - running_max) / running_max

    data.dropna(inplace=True)
    return data


def compute_metrics(data: pd.DataFrame) -> dict:
    """Compute key performance statistics for the strategy and the market."""
    total_days = len(data)
    trading_days_per_year = 252

    # ── Strategy ─────────────────────────────────────────────────────────
    strat_total = data["Cumulative_Strategy"].iloc[-1] / data["Cumulative_Strategy"].iloc[0] - 1
    years = total_days / trading_days_per_year
    strat_annual = (1 + strat_total) ** (1 / years) - 1 if years > 0 else 0
    strat_max_dd = data["Drawdown"].min()

    # Sharpe Ratio (annualised, risk-free ≈ 0 for simplicity)
    excess = data["Strategy_Return"]
    strat_sharpe = (excess.mean() / excess.std()) * np.sqrt(trading_days_per_year) if excess.std() != 0 else 0

    # Volatility (annualised)
    strat_vol = data["Strategy_Return"].std() * np.sqrt(trading_days_per_year)

    # ── Market (Buy & Hold) ──────────────────────────────────────────────
    mkt_total = data["Cumulative_Market"].iloc[-1] / data["Cumulative_Market"].iloc[0] - 1
    mkt_annual = (1 + mkt_total) ** (1 / years) - 1 if years > 0 else 0

    cum_mkt = data["Cumulative_Market"]
    mkt_running_max = cum_mkt.cummax()
    mkt_dd = ((cum_mkt - mkt_running_max) / mkt_running_max).min()

    return {
        "strat_total": strat_total,
        "strat_annual": strat_annual,
        "strat_max_dd": strat_max_dd,
        "strat_sharpe": strat_sharpe,
        "strat_vol": strat_vol,
        "mkt_total": mkt_total,
        "mkt_annual": mkt_annual,
        "mkt_max_dd": mkt_dd,
        "total_days": total_days,
        "years": years,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Execution on button click (or first load with default params)
# ─────────────────────────────────────────────────────────────────────────────
# Initialise session state so the dashboard is shown on first load
if "has_run" not in st.session_state:
    st.session_state.has_run = True
    run_backtest = True  # auto-run on first visit

if run_backtest and fast_period < slow_period:
    # ── Fetch data ───────────────────────────────────────────────────────
    raw = fetch_data(ticker.upper().strip(), str(start_date), str(end_date))

    if raw.empty:
        st.error(
            f"❌ Tidak ditemukan data untuk **{ticker}**. "
            "Periksa simbol ticker dan rentang tanggal."
        )
        st.stop()

    # ── Run strategy ─────────────────────────────────────────────────────
    result = run_strategy(raw, fast_period, slow_period)

    if result.empty:
        st.error(
            "⚠️ Data tidak cukup setelah menghitung Moving Average. "
            "Coba perlebar rentang tanggal atau gunakan periode MA yang lebih kecil."
        )
        st.stop()

    metrics = compute_metrics(result)

    # ─────────────────────────────────────────────────────────────────────
    # KPI Metric Cards
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>📈 Performance Metrics</h3></div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Strategy Total Return",
        f"{metrics['strat_total']:.2%}",
        delta=f"{metrics['strat_total'] - metrics['mkt_total']:+.2%} vs Market",
    )
    k2.metric(
        "Annualised Return",
        f"{metrics['strat_annual']:.2%}",
    )
    k3.metric(
        "Max Drawdown",
        f"{metrics['strat_max_dd']:.2%}",
    )
    k4.metric(
        "Sharpe Ratio",
        f"{metrics['strat_sharpe']:.2f}",
    )
    k5.metric(
        "Market Total Return",
        f"{metrics['mkt_total']:.2%}",
    )

    st.caption(
        f"📅 Period: **{result.index[0].strftime('%Y-%m-%d')}** → "
        f"**{result.index[-1].strftime('%Y-%m-%d')}** · "
        f"**{metrics['total_days']:,}** trading days · "
        f"**{metrics['years']:.1f}** years"
    )

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # Chart 1 — Price + Moving Averages + Crossover Signals
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>🕹️ Price & Moving Averages</h3></div>', unsafe_allow_html=True)

    fig_price = go.Figure()

    # Close price (subtle area fill)
    fig_price.add_trace(
        go.Scatter(
            x=result.index,
            y=result["Close"],
            name="Close Price",
            line=dict(color="rgba(148,163,184,0.55)", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.03)",
        )
    )

    # Fast MA
    fig_price.add_trace(
        go.Scatter(
            x=result.index,
            y=result["Fast_MA"],
            name=f"Fast MA ({fast_period})",
            line=dict(color="#22d3ee", width=2),  # cyan-400
        )
    )

    # Slow MA
    fig_price.add_trace(
        go.Scatter(
            x=result.index,
            y=result["Slow_MA"],
            name=f"Slow MA ({slow_period})",
            line=dict(color="#3b82f6", width=2),  # blue-500
        )
    )

    # Buy signals (crossover up)
    signal_diff = result["Signal"].diff()
    buy_signals = result[signal_diff == 2]
    sell_signals = result[signal_diff == -2]

    fig_price.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            mode="markers",
            name="Buy Signal",
            marker=dict(
                color="#4ade80",
                size=10,
                symbol="triangle-up",
                line=dict(color="#166534", width=1.5),
            ),
        )
    )

    fig_price.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Close"],
            mode="markers",
            name="Sell Signal",
            marker=dict(
                color="#f87171",
                size=10,
                symbol="triangle-down",
                line=dict(color="#991b1b", width=1.5),
            ),
        )
    )

    fig_price.update_layout(
        **_base_layout(
            title=dict(text=f"{ticker.upper()} — Price & MA Crossover"),
            yaxis_title="Price",
            xaxis_title="Date",
            height=520,
            xaxis_rangeslider_visible=False,
        )
    )

    st.plotly_chart(fig_price, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # Chart 2 — Equity Curve: Strategy vs Market
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>💹 Equity Curve — Strategy vs Buy & Hold</h3></div>', unsafe_allow_html=True)

    fig_equity = go.Figure()

    # Strategy curve
    fig_equity.add_trace(
        go.Scatter(
            x=result.index,
            y=result["Cumulative_Strategy"],
            name="MA Crossover Strategy",
            line=dict(color="#06b6d4", width=2.5),  # cyan-500
            fill="tozeroy",
            fillcolor="rgba(6,182,212,0.06)",
        )
    )

    # Market (Buy & Hold) curve
    fig_equity.add_trace(
        go.Scatter(
            x=result.index,
            y=result["Cumulative_Market"],
            name="Buy & Hold (Market)",
            line=dict(color="#6366f1", width=2, dash="dot"),  # indigo-500
        )
    )

    # Reference line at 1.0
    fig_equity.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(148,163,184,0.2)",
        annotation_text="Break Even",
        annotation_font_color="rgba(148,163,184,0.4)",
    )

    fig_equity.update_layout(
        **_base_layout(
            title=dict(text="Cumulative Returns Comparison"),
            yaxis_title="Growth of $1",
            xaxis_title="Date",
            height=480,
        )
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # Chart 3 — Drawdown
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>📉 Strategy Drawdown</h3></div>', unsafe_allow_html=True)

    fig_dd = go.Figure()

    fig_dd.add_trace(
        go.Scatter(
            x=result.index,
            y=result["Drawdown"],
            name="Drawdown",
            line=dict(color="#f43f5e", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(244,63,94,0.10)",
        )
    )

    fig_dd.update_layout(
        **_base_layout(
            title=dict(text="Strategy Underwater (Drawdown) Curve"),
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            xaxis_title="Date",
            height=350,
        )
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>📋 Summary Comparison</h3></div>', unsafe_allow_html=True)

    summary_data = {
        "Metric": [
            "Total Return",
            "Annualised Return",
            "Maximum Drawdown",
            "Annualised Volatility",
            "Sharpe Ratio",
        ],
        "MA Crossover Strategy": [
            f"{metrics['strat_total']:.2%}",
            f"{metrics['strat_annual']:.2%}",
            f"{metrics['strat_max_dd']:.2%}",
            f"{metrics['strat_vol']:.2%}",
            f"{metrics['strat_sharpe']:.2f}",
        ],
        "Buy & Hold (Market)": [
            f"{metrics['mkt_total']:.2%}",
            f"{metrics['mkt_annual']:.2%}",
            f"{metrics['mkt_max_dd']:.2%}",
            "—",
            "—",
        ],
    }

    st.dataframe(
        pd.DataFrame(summary_data).set_index("Metric"),
        use_container_width=True,
    )

elif not run_backtest:
    # Placeholder when no backtest has been run yet
    st.info(
        "👈 Atur parameter di **sidebar**, lalu klik **🚀 Run Backtest** untuk memulai."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branding Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="branding-bar">'
    "Quantitative Research Portfolio by <b>Rhameyza Faiqo Susanto</b>"
    "</div>",
    unsafe_allow_html=True,
)
