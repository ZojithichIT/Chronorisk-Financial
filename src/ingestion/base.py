"""Base interfaces, normalization, validation, and retry helpers for OHLCV ingestion."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

OHLCV_COLUMNS: Final[list[str]] = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

_NUMERIC_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]

_TIMEFRAME_DELTAS: Final[dict[str, pd.Timedelta]] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "10m": pd.Timedelta(minutes=10),
    "15m": pd.Timedelta(minutes=15),
    "20m": pd.Timedelta(minutes=20),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
}


def empty_ohlcv_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame matching the canonical OHLCV schema."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "symbol": pd.Series(dtype="string"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert a ChronoRisk timeframe string into a ``pd.Timedelta``.

    Args:
        timeframe: Supported bar interval (e.g., ``"1d"``, ``"4h"``, ``"1h"``, ``"15m"``).

    Returns:
        Corresponding ``pd.Timedelta``.
    """
    tf = timeframe.strip().lower()
    if tf not in _TIMEFRAME_DELTAS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return _TIMEFRAME_DELTAS[tf]


def with_exponential_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Execute a callable with exponential backoff retries on transient failures.

    Args:
        func: Zero-argument callable to execute.
        max_retries: Maximum retry attempts after the initial call.
        base_delay: Initial wait duration in seconds.
        max_delay: Upper bound on sleep delay in seconds.
        retry_exceptions: Exception types that trigger a retry.
        sleep_fn: Sleep implementation (injectable for deterministic tests).

    Returns:
        Return value of ``func``.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_exceptions as exc:
            if attempt >= max_retries:
                logger.error("Operation failed after %d attempts: %s", attempt + 1, exc)
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.2fs.",
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            sleep_fn(delay)
    raise RuntimeError("Unreachable backoff loop exit")


def normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize raw OHLCV data to lowercase schema, UTC timestamps, and sorted unique rows.

    Args:
        df: Raw market data DataFrame containing ``timestamp`` and OHLCV columns.
        symbol: Canonical asset identifier (e.g., ``"BTC/USDT"``, ``"SPY"``, ``"FPT"``).

    Returns:
        DataFrame with columns ``['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']``,
        ``datetime64[ns, UTC]`` timestamps, ``float64`` prices/volumes, sorted ascending,
        and deduplicated on ``(symbol, timestamp)``.
    """
    if df.empty:
        return empty_ohlcv_dataframe()

    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]

    required = {"timestamp", *_NUMERIC_COLUMNS}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    normalized["symbol"] = pd.Series([symbol] * len(normalized), index=normalized.index, dtype="string")
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).dt.as_unit("ns")

    for col in _NUMERIC_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype("float64")

    normalized = normalized.loc[:, OHLCV_COLUMNS]
    normalized = normalized.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    normalized = normalized.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return normalized


def drop_incomplete_bars(
    df: pd.DataFrame,
    timeframe: str,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Remove bars whose closing time has not yet passed.

    Args:
        df: Normalized OHLCV DataFrame with UTC ``timestamp`` representing bar open time.
        timeframe: Bar interval string.
        now: Reference UTC timestamp (defaults to current UTC clock).

    Returns:
        DataFrame containing only fully closed bars.
    """
    if df.empty:
        return df.copy()

    if now is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    else:
        now_utc = pd.Timestamp(now)
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")

    bar_delta = timeframe_to_timedelta(timeframe)
    closed_mask = (df["timestamp"] + bar_delta) <= now_utc
    dropped = int((~closed_mask).sum())
    if dropped > 0:
        logger.info("Dropped %d incomplete %s bar(s) after %s.", dropped, timeframe, now_utc.isoformat())

    return df.loc[closed_mask].reset_index(drop=True)


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str | None = None,
    strict: bool = False,
    max_gap_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Validate OHLCV integrity and report unexpected timestamp gaps.

    Args:
        df: Normalized OHLCV DataFrame.
        timeframe: Optional bar interval for gap diagnostics.
        strict: If ``True``, raise ``ValueError`` on any invalid bar; otherwise filter invalid bars.
        max_gap_multiplier: Warn when consecutive bar gaps exceed ``timeframe * max_gap_multiplier``.

    Returns:
        Validated OHLCV DataFrame.
    """
    if df.empty:
        return df.copy()

    null_mask = df.loc[:, OHLCV_COLUMNS].isna().any(axis=1)
    positive_price_mask = (
        (df["open"] > 0.0)
        & (df["high"] > 0.0)
        & (df["low"] > 0.0)
        & (df["close"] > 0.0)
    )
    non_negative_vol_mask = df["volume"] >= 0.0
    high_bound_mask = df["high"] >= np.maximum(df["open"], df["close"])
    low_bound_mask = df["low"] <= np.minimum(df["open"], df["close"])
    range_mask = df["high"] >= df["low"]

    valid_mask = (
        (~null_mask)
        & positive_price_mask
        & non_negative_vol_mask
        & high_bound_mask
        & low_bound_mask
        & range_mask
    )

    invalid_count = int((~valid_mask).sum())
    if invalid_count > 0:
        message = f"Found {invalid_count} invalid OHLCV bar(s) violating price/volume bounds."
        if strict:
            raise ValueError(message)
        logger.warning("%s Filtering invalid bars.", message)

    cleaned = df.loc[valid_mask].reset_index(drop=True)

    if timeframe is not None and len(cleaned) > 1:
        expected_delta = timeframe_to_timedelta(timeframe)
        gap_threshold = expected_delta * max_gap_multiplier
        diffs = cleaned["timestamp"].diff().dropna()
        large_gaps = diffs[diffs > gap_threshold]
        if not large_gaps.empty:
            logger.warning(
                "Detected %d timestamp gap(s) exceeding %.1fx %s (max gap: %s).",
                len(large_gaps),
                max_gap_multiplier,
                timeframe,
                large_gaps.max(),
            )

    return cleaned


class MarketDataFetcher(ABC):
    """Abstract base class for market data ingestion providers."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        drop_unclosed: bool = True,
    ) -> pd.DataFrame:
        """Fetch, normalize, and validate OHLCV bars for a single symbol."""
