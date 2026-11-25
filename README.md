# ChronoRisk

**Multi-Asset Financial Volatility Forecasting & Risk Management System**

ChronoRisk is an end-to-end time series analytics pipeline designed to forecast market volatility, estimate Value at Risk (VaR), and detect market regimes across equities and cryptocurrencies. Instead of attempting to predict noisy raw asset prices, the system focuses on risk quantification and decision support to help manage position sizing and drawdown exposure.

---

## Key Features

- **Multi-Source Data Ingestion:** Automated OHLCV data collection supporting Crypto (`ccxt`), US Equities (`yfinance`), and Vietnamese Equities (`vnstock`), stored idempotently in Parquet / PostgreSQL.
- **Statistical & Quantitative Preprocessing:**
  - Log-return transformation and Augmented Dickey-Fuller (ADF) stationarity verification.
  - Technical and volatility feature engineering (Rolling Volatility, ATR, RSI, MACD, Bollinger Bandwidth, and cyclical time embeddings).
- **Leakage-Free Validation Pipeline:** Implements **Purged Walk-Forward Validation** (`TimeSeriesSplit` with configurable gap intervals) and strict train-only scaler fitting to eliminate look-ahead bias and feature overlap.
- **Multi-Paradigm Modeling Benchmark:**
  - **Statistical Baseline:** Naive Forecast & ARIMA-GARCH for conditional volatility modeling.
  - **Gradient Boosted Trees:** LightGBM / XGBoost trained on engineered lag and microstructure features.
  - **Deep Learning:** PyTorch LSTM / GRU with custom directional penalty loss to mitigate the common "lag-1" prediction trap.
- **Risk Analytics & Backtesting:** Evaluates models beyond standard regression errors (RMSE/MAE) using financial risk metrics: **Value at Risk (VaR 95%/99%)**, **CVaR**, **Maximum Drawdown (MDD)**, **Sharpe Ratio**, and **Volatility-Adjusted Position Sizing**.

---

## Project Structure

```text
chronorisk-financial-timeseries/
├── data/                   # Raw and processed datasets (.parquet)
├── docs/                   # Architecture and quantitative decision records
├── notebooks/              # EDA, stationarity tests (ADF/ACF), and prototyping
├── src/
│   ├── ingestion/          # API scrapers (ccxt, vnstock, yfinance)
│   ├── features/           # Technical indicators, rolling stats, regime filters
│   ├── models/             # GARCH, LightGBM, and PyTorch LSTM implementations
│   ├── evaluation/         # Purged Walk-Forward split, VaR, and backtest engine
│   └── api/                # FastAPI endpoints serving risk metrics & forecasts
├── tests/                  # Unit tests for data leakage and pipeline integrity
├── docker-compose.yml      # Containerized setup (PostgreSQL + FastAPI)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Data & Quantitative Analysis:** Python, Pandas, NumPy, Statsmodels, `ta`
- **Machine Learning & Deep Learning:** Scikit-learn, LightGBM, PyTorch
- **Backend & Storage:** FastAPI, PostgreSQL, Parquet
- **Deployment & Tooling:** Docker, Git, Jupyter Notebook

---

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ZojithichIT/Chronorisk-Financial.git
   cd Chronorisk-Financial
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data ingestion & backtest pipeline:**
   ```bash
   python -m src.ingestion.fetch_data --symbol BTC/USDT --timeframe 1d
   python -m src.evaluation.run_walk_forward --model lightgbm
   ```

---

## Disclaimer

*This project is built strictly for educational, research, and quantitative portfolio demonstration purposes. Nothing in this repository constitutes financial or investment advice.*
