"""Cryptocurrency OHLCV data ingestion via ccxt."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ingestion.base import (
    MarketDataFetcher,
    drop_incomplete_bars,
    empty_ohlcv_dataframe,
    normalize_ohlcv,
    timeframe_to_timedelta,
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


class CCXTFetcher(MarketDataFetcher):
    """Fetch paginated crypto OHLCV bars using ``ccxt`` with rate-limit awareness."""

    def __init__(
        self,
        exchange_id: str = "binance",
        exchange: Any | None = None,
        limit: int = 1000,
    ) -> None:
        """Initialize the CCXT fetcher.

        Args:
            exchange_id: Exchange name registered in ``ccxt`` (default ``"binance"``).
            exchange: Optional pre-configured ``ccxt`` exchange instance (useful for testing).
            limit: Maximum bars requested per API page.
        """
        self.exchange_id = exchange_id
        self._exchange = exchange
        self.limit = limit

    def _get_exchange(self) -> Any:
        if self._exchange is not None:
            return self._exchange

        import ccxt

        exchange_cls = getattr(ccxt, self.exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown ccxt exchange: {self.exchange_id!r}")
        self._exchange = exchange_cls({"enableRateLimit": True})
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        drop_unclosed: bool = True,
    ) -> pd.DataFrame:
        """Fetch normalized and validated OHLCV bars from a crypto exchange.

        Args:
            symbol: Trading pair symbol (e.g., ``"BTC/USDT"``).
            timeframe: Bar interval (e.g., ``"1d"``, ``"4h"``, ``"1h"``, ``"15m"``).
            start: Inclusive start timestamp (defaults to 4 years prior to ``end``).
            end: Inclusive end timestamp (defaults to current UTC time).
            drop_unclosed: Whether to remove the current in-flight unclosed bar.

        Returns:
            Canonical OHLCV DataFrame.
        """
        now_utc = pd.Timestamp.now(tz="UTC")
        end_ts = _to_utc_timestamp(end, default=now_utc)
        start_ts = _to_utc_timestamp(start, default=end_ts - pd.Timedelta(days=365 * 4))

        since_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        bar_ms = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)

        exchange = self._get_exchange()
        all_rows: list[list[float]] = []

        while since_ms <= end_ms:
            batch = with_exponential_backoff(
                lambda s=since_ms: exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=s,
                    limit=self.limit,
                )
            )
            if not batch:
                break

            filtered_batch = [row for row in batch if int(row[0]) <= end_ms]
            if not filtered_batch:
                break

            all_rows.extend(filtered_batch)
            last_ts_ms = int(filtered_batch[-1][0])

            if len(batch) < self.limit or last_ts_ms >= end_ms:
                break

            next_since_ms = last_ts_ms + bar_ms
            if next_since_ms <= since_ms:
                break
            since_ms = next_since_ms

        if not all_rows:
            logger.warning("No OHLCV bars returned by ccxt for %s (%s).", symbol, timeframe)
            return empty_ohlcv_dataframe()

        raw_df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], unit="ms", utc=True)

        normalized = normalize_ohlcv(raw_df, symbol=symbol)
        if drop_unclosed:
            normalized = drop_incomplete_bars(normalized, timeframe=timeframe, now=now_utc)
        return validate_ohlcv(normalized, timeframe=timeframe)
