"""Vietnamese equity OHLCV data ingestion via vnstock."""

from __future__ import annotations

from collections.abc import Callable

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

_VNSTOCK_INTERVAL_MAP: dict[str, str] = {
    "1d": "1D",
    "1h": "1H",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}


def _to_utc_timestamp(value: pd.Timestamp | str | None, default: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return default
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _default_vnstock_history(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
) -> pd.DataFrame:
    from vnstock import Vnstock

    interval = _VNSTOCK_INTERVAL_MAP.get(timeframe.strip().lower(), timeframe)
    stock = Vnstock().stock(symbol=symbol, source=source)
    return stock.quote.history(
        start=start.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m-%d"),
        end=end.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m-%d"),
        interval=interval,
    )


class VNStockFetcher(MarketDataFetcher):
    """Fetch Vietnamese equity OHLCV bars via ``vnstock`` with Asia/Ho_Chi_Minh -> UTC conversion."""

    def __init__(
        self,
        source: str = "VCI",
        history_fn: Callable[[str, str, pd.Timestamp, pd.Timestamp, str], pd.DataFrame] | None = None,
    ) -> None:
        """Initialize the vnstock fetcher.

        Args:
            source: Upstream data provider supported by ``vnstock`` (default ``"VCI"``).
            history_fn: Optional custom history loader (injectable for offline tests).
        """
        self.source = source
        self._history_fn = history_fn or _default_vnstock_history

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        drop_unclosed: bool = True,
    ) -> pd.DataFrame:
        """Fetch normalized and validated Vietnamese equity OHLCV bars.

        Args:
            symbol: Ticker or index symbol (e.g., ``"FPT"``, ``"VNINDEX"``).
            timeframe: Bar interval (``"1d"``, ``"1h"``, ``"15m"``, ``"5m"``, ``"1m"``).
            start: Start timestamp (defaults to 4 years prior to ``end``).
            end: End timestamp (defaults to current UTC time).
            drop_unclosed: Whether to drop unclosed intraday bars.

        Returns:
            Canonical OHLCV DataFrame.
        """
        now_utc = pd.Timestamp.now(tz="UTC")
        end_ts = _to_utc_timestamp(end, default=now_utc)
        start_ts = _to_utc_timestamp(start, default=end_ts - pd.Timedelta(days=365 * 4))

        raw_df = with_exponential_backoff(
            lambda: self._history_fn(symbol, timeframe, start_ts, end_ts, self.source)
        )
        if raw_df is None or raw_df.empty:
            logger.warning("No OHLCV bars returned by vnstock for %s (%s).", symbol, timeframe)
            return empty_ohlcv_dataframe()

        prepared = raw_df.copy()
        prepared.columns = [str(col).strip().lower() for col in prepared.columns]
        if "timestamp" not in prepared.columns:
            if "time" in prepared.columns:
                prepared = prepared.rename(columns={"time": "timestamp"})
            elif "date" in prepared.columns:
                prepared = prepared.rename(columns={"date": "timestamp"})
            else:
                prepared = prepared.reset_index()
                first_col = str(prepared.columns[0]).strip().lower()
                prepared.columns = [str(col).strip().lower() for col in prepared.columns]
                prepared = prepared.rename(columns={first_col: "timestamp"})

        ts_series = pd.to_datetime(prepared["timestamp"])
        if ts_series.dt.tz is None:
            ts_series = ts_series.dt.tz_localize("Asia/Ho_Chi_Minh")
        prepared["timestamp"] = ts_series.dt.tz_convert("UTC")

        normalized = normalize_ohlcv(prepared, symbol=symbol)
        if drop_unclosed and timeframe.strip().lower() != "1d":
            normalized = drop_incomplete_bars(normalized, timeframe=timeframe, now=now_utc)

        return validate_ohlcv(normalized, timeframe=timeframe)
