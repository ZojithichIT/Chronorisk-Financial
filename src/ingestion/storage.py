"""Idempotent Parquet and PostgreSQL persistence for normalized OHLCV data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import get_settings
from src.ingestion.base import OHLCV_COLUMNS, empty_ohlcv_dataframe, normalize_ohlcv
from src.utils.logging import get_logger

logger = get_logger(__name__)

_VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def symbol_to_filename(symbol: str, timeframe: str) -> str:
    """Convert a market symbol and timeframe into a filesystem-safe Parquet filename."""
    safe_symbol = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol.strip())
    safe_timeframe = re.sub(r"[^A-Za-z0-9._-]+", "_", timeframe.strip().lower())
    return f"{safe_symbol}_{safe_timeframe}.parquet"


def get_parquet_path(
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
) -> Path:
    """Resolve the Parquet storage path for a given symbol and timeframe."""
    base_dir = raw_dir or get_settings().raw_data_dir
    return base_dir / symbol_to_filename(symbol, timeframe)


def load_from_parquet(
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load normalized OHLCV data from Parquet if the file exists.

    Args:
        symbol: Asset identifier.
        timeframe: Bar interval string.
        raw_dir: Optional directory override.

    Returns:
        Canonical OHLCV DataFrame (or empty canonical DataFrame if missing).
    """
    path = get_parquet_path(symbol, timeframe, raw_dir=raw_dir)
    if not path.exists():
        return empty_ohlcv_dataframe()

    df = pd.read_parquet(path)
    return normalize_ohlcv(df, symbol=symbol)


def get_latest_parquet_timestamp(
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
) -> pd.Timestamp | None:
    """Return the most recent UTC timestamp stored in Parquet for incremental updates."""
    existing = load_from_parquet(symbol, timeframe, raw_dir=raw_dir)
    if existing.empty:
        return None
    return pd.Timestamp(existing["timestamp"].max())


def save_to_parquet(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
) -> Path:
    """Merge and deduplicate OHLCV bars on ``(symbol, timestamp)`` and write atomically to Parquet.

    Args:
        df: Normalized OHLCV DataFrame.
        symbol: Asset identifier.
        timeframe: Bar interval string.
        raw_dir: Optional directory override.

    Returns:
        Destination Parquet ``Path``.
    """
    target_path = get_parquet_path(symbol, timeframe, raw_dir=raw_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    incoming = normalize_ohlcv(df, symbol=symbol)
    if target_path.exists():
        existing = load_from_parquet(symbol, timeframe, raw_dir=raw_dir)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    deduped = normalize_ohlcv(combined, symbol=symbol)
    tmp_path = target_path.with_suffix(".parquet.tmp")
    deduped.to_parquet(tmp_path, index=False)
    tmp_path.replace(target_path)

    logger.info("Saved %d total bar(s) to %s.", len(deduped), target_path)
    return target_path


def upsert_to_postgres(
    df: pd.DataFrame,
    database_url: str,
    table_name: str = "ohlcv_bars",
) -> int:
    """Idempotently upsert OHLCV bars into PostgreSQL (or SQLite for tests).

    Args:
        df: Canonical OHLCV DataFrame.
        database_url: SQLAlchemy connection URL.
        table_name: Target table name containing ``PRIMARY KEY (symbol, timestamp)``.

    Returns:
        Number of deduplicated rows upserted.
    """
    if not _VALID_TABLE_NAME.fullmatch(table_name):
        raise ValueError(f"Invalid SQL table name: {table_name!r}")
    if df.empty:
        return 0

    clean_df = df.loc[:, OHLCV_COLUMNS].copy()
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], utc=True)
    clean_df = clean_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    clean_df = clean_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    create_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        )
        """
    )

    upsert_sql = text(
        f"""
        INSERT INTO {table_name} (symbol, timestamp, open, high, low, close, volume)
        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume
        """
    )

    engine = create_engine(database_url, future=True)
    is_sqlite = engine.dialect.name == "sqlite"

    records = [
        {
            "symbol": str(row.symbol),
            "timestamp": (
                row.timestamp.isoformat()
                if is_sqlite
                else row.timestamp.to_pydatetime()
            ),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in clean_df.itertuples(index=False)
    ]

    with engine.begin() as conn:
        conn.execute(create_sql)
        conn.execute(upsert_sql, records)

    logger.info("Upserted %d OHLCV row(s) into table %s.", len(records), table_name)
    return len(records)
