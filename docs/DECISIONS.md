# Architecture & Quantitative Decisions — ChronoRisk

1. **Prediction Target:** Forecast realized volatility and log-returns rather than raw closing prices to prevent lag-1 naive convergence.
2. **Validation Strategy:** Purged Walk-Forward Validation (`TimeSeriesSplit` with `gap >= forecast_horizon`) to eliminate overlapping rolling-window leakage.
3. **Storage Format:** Local development uses `.parquet` (UTC timestamps); production/API layer uses PostgreSQL with `(symbol, timestamp)` unique constraints.
4. **Evaluation Standard:** Models must beat a Naive Baseline on both statistical metrics and risk-adjusted financial metrics (VaR breach rate, Sharpe Ratio, Max Drawdown).
5. **Timeframe-Aware Annualization:** Volatility and Sharpe ratios scale dynamically by asset class and timeframe (`1d`, `4h`, `1h`, `15m`, `5m`, `1m`), accounting for 24/7 crypto markets (`365` days) vs. equity trading sessions (`252` days).
6. **Incremental Closed-Bar Ingestion:** Historical ingestion resumes incrementally from the latest stored `(symbol, timestamp)` and drops unclosed in-flight bars by default to prevent partial-bar volatility distortion during training/backtesting.
