"""CLI entrypoint for multi-source incremental OHLCV data ingestion."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import get_settings
from src.ingestion.base import MarketDataFetcher
from src.ingestion.crypto_ccxt import CCXTFetcher
from src.ingestion.storage import (
    get_latest_parquet_timestamp,
    get_parquet_path,
    load_from_parquet,
    save_to_parquet,
    upsert_to_postgres,
)
from src.ingestion.us_yfinance import YFinanceFetcher
from src.ingestion.vn_vnstock import VNStockFetcher
from src.utils.logging import get_logger

logger = get_logger(__name__)

SourceName = Literal["auto", "ccxt", "yfinance", "vnstock"]

_VN_INDICES = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOM"}


def infer_source(symbol: str) -> Literal["ccxt", "yfinance", "vnstock"]:
    """Infer the most likely ingestion provider from a symbol string."""
    upper = symbol.strip().upper()
    if "/" in upper:
        return "ccxt"
    if upper in _VN_INDICES or upper.endswith(".VN"):
        return "vnstock"
    return "yfinance"


def build_fetcher(source: Literal["ccxt", "yfinance", "vnstock"], exchange: str) -> MarketDataFetcher:
    """Instantiate the concrete market data fetcher for a source name."""
    if source == "ccxt":
        return CCXTFetcher(exchange_id=exchange)
    if source == "yfinance":
        return YFinanceFetcher()
    if source == "vnstock":
        return VNStockFetcher()
    raise ValueError(f"Unsupported ingestion source: {source!r}")


def run_ingestion(
    symbol: str,
    timeframe: str = "1d",
    source: SourceName = "auto",
    start: str | None = None,
    end: str | None = None,
    exchange: str | None = None,
    full_refresh: bool = False,
    drop_unclosed: bool = True,
    to_postgres: bool = False,
    raw_dir: Path | None = None,
    fetcher: MarketDataFetcher | None = None,
) -> Path:
    """Fetch OHLCV bars (incrementally by default) and persist idempotently.

    Args:
        symbol: Asset ticker or pair (e.g., ``"BTC/USDT"``, ``"SPY"``, ``"FPT"``).
        timeframe: Bar interval (``"1d"``, ``"4h"``, ``"1h"``, etc.).
        source: Data provider (``"auto"``, ``"ccxt"``, ``"yfinance"``, ``"vnstock"``).
        start: Optional explicit start date/timestamp; overrides incremental resume.
        end: Optional end date/timestamp.
        exchange: Exchange ID when ``source == "ccxt"``.
        full_refresh: Ignore existing Parquet checkpoint and fetch default lookback window.
        drop_unclosed: Drop the latest bar if it has not closed yet.
        to_postgres: Also upsert fetched bars into configured PostgreSQL database.
        raw_dir: Optional override for raw Parquet storage directory.
        fetcher: Optional pre-built ``MarketDataFetcher`` (primarily for testing).

    Returns:
        Path to the saved Parquet file.
    """
    settings = get_settings()
    target_dir = raw_dir or settings.raw_data_dir
    resolved_exchange = exchange or settings.default_exchange
    resolved_source = infer_source(symbol) if source == "auto" else source

    effective_start: pd.Timestamp | str | None = start
    if effective_start is None and not full_refresh:
        latest_ts = get_latest_parquet_timestamp(symbol, timeframe, raw_dir=target_dir)
        if latest_ts is not None:
            effective_start = latest_ts
            logger.info(
                "Resuming incremental ingestion for %s (%s) from %s.",
                symbol,
                timeframe,
                latest_ts.isoformat(),
            )

    active_fetcher = fetcher or build_fetcher(resolved_source, exchange=resolved_exchange)
    fetched = active_fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start=effective_start,
        end=end,
        drop_unclosed=drop_unclosed,
    )

    output_path = get_parquet_path(symbol, timeframe, raw_dir=target_dir)
    if fetched.empty:
        logger.warning("No new bars fetched for %s (%s).", symbol, timeframe)
        if output_path.exists():
            return output_path
        return save_to_parquet(fetched, symbol=symbol, timeframe=timeframe, raw_dir=target_dir)

    output_path = save_to_parquet(fetched, symbol=symbol, timeframe=timeframe, raw_dir=target_dir)
    stored = load_from_parquet(symbol, timeframe, raw_dir=target_dir)

    if to_postgres:
        if not settings.database_url:
            raise ValueError("CHRONORISK_DATABASE_URL must be set when using --to-postgres.")
        upsert_to_postgres(fetched, database_url=settings.database_url)

    logger.info(
        "Ingestion complete for %s (%s): fetched=%d, stored_total=%d, range=[%s .. %s]",
        symbol,
        timeframe,
        len(fetched),
        len(stored),
        stored["timestamp"].min().isoformat(),
        stored["timestamp"].max().isoformat(),
    )
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for market data ingestion."""
    parser = argparse.ArgumentParser(description="ChronoRisk multi-source OHLCV ingestion CLI")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., BTC/USDT, SPY, FPT)")
    parser.add_argument("--timeframe", default="1d", help="Bar interval (1d, 4h, 1h, 15m, 5m)")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "ccxt", "yfinance", "vnstock"],
        help="Upstream provider (default: auto-detect)",
    )
    parser.add_argument("--start", default=None, help="Start date/timestamp (ISO-8601)")
    parser.add_argument("--end", default=None, help="End date/timestamp (ISO-8601)")
    parser.add_argument("--exchange", default=None, help="CCXT exchange ID (default: binance)")
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore incremental checkpoint and fetch full lookback window",
    )
    parser.add_argument(
        "--include-unclosed",
        action="store_true",
        help="Keep the current unclosed bar (useful for intraday monitoring)",
    )
    parser.add_argument(
        "--to-postgres",
        action="store_true",
        help="Also upsert bars to PostgreSQL configured via CHRONORISK_DATABASE_URL",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run_ingestion(
        symbol=args.symbol,
        timeframe=args.timeframe,
        source=args.source,
        start=args.start,
        end=args.end,
        exchange=args.exchange,
        full_refresh=args.full_refresh,
        drop_unclosed=not args.include_unclosed,
        to_postgres=args.to_postgres,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
