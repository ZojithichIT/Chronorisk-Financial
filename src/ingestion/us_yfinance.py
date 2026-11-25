"""US equity OHLCV data ingestion via yfinance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from src.ingestion.base import (
    MarketDataFetcher,
    drop_incomplete_bars,
    empty_ohlcv_dataframe,
    normalize_ohlcv,
    validate_ohlcv,
    with_exponential_backoff,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _to_utc_timestamp(value: pd.Timestamp | str | None, default: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return default
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _default_yfinance_download(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    # yfinance end is exclusive; add 1 day so daily bar on `end` date is included.
    return ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=timeframe,
        auto_adjust=False,
    )


class YFinanceFetcher(MarketDataFetcher):
    """Fetch US equity OHLCV bars via ``yfinance`` with UTC normalization."""

    def __init__(
        self,
        download_fn: Callable[[str, str, pd.Timestamp, pd.Timestamp], pd.DataFrame] | None = None,
    ) -> None:
        """Initialize the Yahoo Finance fetcher.

        Args:
            download_fn: Optional custom history loader (injectable for offline tests).
        """
        self._download_fn = download_fn or _default_yfinance_download

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        drop_unclosed: bool = True,
    ) -> pd.DataFrame:
        """Fetch normalized and validated US equity OHLCV bars.

        Args:
            symbol: Equity or ETF ticker (e.g., ``"SPY"``, ``"QQQ"``).
            timeframe: Bar interval supported by Yahoo Finance (``"1d"``, ``"1h"``, ``"15m"``, ``"5m"``).
            start: Start timestamp (defaults to 4 years prior to ``end``).
            end: End timestamp (defaults to current UTC time).
            drop_unclosed: Whether to remove an unclosed intraday bar.

        Returns:
            Canonical OHLCV DataFrame.
        """
        now_utc = pd.Timestamp.now(tz="UTC")
        end_ts = _to_utc_timestamp(end, default=now_utc)
        start_ts = _to_utc_timestamp(start, default=end_ts - pd.Timedelta(days=365 * 4))

        raw_df = with_exponential_backoff(
            lambda: self._download_fn(symbol, timeframe, start_ts, end_ts)
        )
        if raw_df is None or raw_df.empty:
            logger.warning("No OHLCV bars returned by yfinance for %s (%s).", symbol, timeframe)
            return empty_ohlcv_dataframe()

        prepared = raw_df.copy()
        if isinstance(prepared.columns, pd.MultiIndex):
            prepared.columns = [col[0] for col in prepared.columns]

        if "timestamp" not in {str(c).strip().lower() for c in prepared.columns}:
            prepared = prepared.reset_index()
            first_col = prepared.columns[0]
            prepared = prepared.rename(columns={first_col: "timestamp"})

        prepared.columns = [str(col).strip().lower() for col in prepared.columns]
        ts_series = pd.to_datetime(prepared["timestamp"])
        if ts_series.dt.tz is None:
            ts_series = ts_series.dt.tz_localize("America/New_York")
        prepared["timestamp"] = ts_series.dt.tz_convert("UTC")

        normalized = normalize_ohlcv(prepared, symbol=symbol)
        if drop_unclosed and timeframe.strip().lower() != "1d":
            normalized = drop_incomplete_bars(normalized, timeframe=timeframe, now=now_utc)

        return validate_ohlcv(normalized, timeframe=timeframe)
