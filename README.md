# Moving Average Crossover — Backtesting Dashboard

An interactive Streamlit dashboard for backtesting a **Moving Average Crossover** trading strategy. Compare strategy performance against a passive Buy & Hold benchmark with professional-grade visualizations.

> **Quantitative Research Portfolio by Rhameyza Faiqo Susanto**

---

##  Features

- **Configurable Parameters** — Ticker symbol, date range, Fast/Slow MA periods
- **Real-Time Data** — Pulls historical prices from Yahoo Finance via `yfinance`
- **MA Crossover Strategy** — Long when Fast MA > Slow MA, Short when below
- **Performance Metrics** — Total Return, Annualised Return, Max Drawdown, Sharpe Ratio
- **Interactive Charts** — Built with Plotly (deep blue/black theme)
  - Price chart with MA overlays & Buy/Sell signal markers
  - Equity Curve — Strategy vs Buy & Hold
  - Drawdown curve

##  Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ma-crossover-backtest.git
cd ma-crossover-backtest

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

##  Requirements

- Python 3.9+
- streamlit
- yfinance
- plotly
- pandas
- numpy

## 📄 License

MIT
