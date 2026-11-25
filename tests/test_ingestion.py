"""Unit tests for Phase 0 configuration and Phase 1 OHLCV ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.config import get_annualization_factor
from src.ingestion.base import (
    OHLCV_COLUMNS,
    drop_incomplete_bars,
    normalize_ohlcv,
    validate_ohlcv,
)
from src.ingestion.crypto_ccxt import CCXTFetcher
from src.ingestion.fetch_data import infer_source, run_ingestion
from src.ingestion.storage import (
    get_latest_parquet_timestamp,
    load_from_parquet,
    save_to_parquet,
    upsert_to_postgres,
)
from src.ingestion.us_yfinance import YFinanceFetcher
from src.ingestion.vn_vnstock import VNStockFetcher


def test_annualization_factors() -> None:
    assert get_annualization_factor("crypto", "1d") == pytest.approx(365.0)
    assert get_annualization_factor("crypto", "4h") == pytest.approx(365.0 * 6.0)
    assert get_annualization_factor("crypto", "1h") == pytest.approx(365.0 * 24.0)
    assert get_annualization_factor("us_equity", "1d") == pytest.approx(252.0)
    assert get_annualization_factor("us_equity", "1h") == pytest.approx(252.0 * 6.5)
    assert get_annualization_factor("vn_equity", "1d") == pytest.approx(252.0)


def test_normalize_ohlcv_schema_utc_and_dedup() -> None:
    raw = pd.DataFrame(
        {
            "Timestamp": ["2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "Open": [101.0, 100.0, 100.5],
            "High": [105.0, 102.0, 103.0],
            "Low": [99.0, 98.0, 99.0],
            "Close": [104.0, 101.0, 102.0],
            "Volume": [500.0, 400.0, 450.0],
        }
    )

    norm = normalize_ohlcv(raw, symbol="BTC/USDT")

    assert list(norm.columns) == OHLCV_COLUMNS
    assert len(norm) == 2
    assert str(norm["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert pd.api.types.is_string_dtype(norm["symbol"])
    assert norm["timestamp"].is_monotonic_increasing
    assert norm.loc[0, "close"] == pytest.approx(102.0)


def test_validate_ohlcv_filters_and_strict_mode() -> None:
    df = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                ],
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [105.0, 95.0, 105.0, 105.0],  # row 1: high < open/close
                "low": [98.0, 98.0, 102.0, 98.0],    # row 2: low > open
                "close": [102.0, 101.0, 103.0, -1.0], # row 3: negative close
                "volume": [10.0, 10.0, 10.0, 10.0],
            }
        ),
        symbol="SPY",
    )

    cleaned = validate_ohlcv(df, timeframe="1d", strict=False)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "close"] == pytest.approx(102.0)

    with pytest.raises(ValueError):
        validate_ohlcv(df, timeframe="1d", strict=True)


def test_drop_incomplete_bars() -> None:
    df = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
                "open": [100.0, 102.0],
                "high": [103.0, 104.0],
                "low": [99.0, 101.0],
                "close": [102.0, 103.0],
                "volume": [10.0, 5.0],
            }
        ),
        symbol="BTC/USDT",
    )
    now = pd.Timestamp("2024-01-01T01:30:00Z")
    closed = drop_incomplete_bars(df, timeframe="1h", now=now)

    assert len(closed) == 1
    assert closed.loc[0, "timestamp"] == pd.Timestamp("2024-01-01T00:00:00Z")


def test_ccxt_fetcher_pagination_and_retry() -> None:
    t0 = int(pd.Timestamp("2024-01-01T00:00:00Z").timestamp() * 1000)
    t1 = int(pd.Timestamp("2024-01-02T00:00:00Z").timestamp() * 1000)
    t2 = int(pd.Timestamp("2024-01-03T00:00:00Z").timestamp() * 1000)

    class MockExchange:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient timeout")
            if since <= t0:
                return [
                    [t0, 42000.0, 43000.0, 41500.0, 42500.0, 1000.0],
                    [t1, 42500.0, 43500.0, 42000.0, 43200.0, 1200.0],
                ]
            return [[t2, 43200.0, 44000.0, 43000.0, 43800.0, 900.0]]

    fetcher = CCXTFetcher(exchange=MockExchange(), limit=2)
    df = fetcher.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1d",
        start="2024-01-01T00:00:00Z",
        end="2024-01-03T00:00:00Z",
        drop_unclosed=False,
    )

    assert len(df) == 3
    assert list(df.columns) == OHLCV_COLUMNS
    assert df["close"].tolist() == [42500.0, 43200.0, 43800.0]


def test_yfinance_and_vnstock_timezone_normalization() -> None:
    def mock_yf(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.DatetimeIndex(["2024-01-02 09:30:00", "2024-01-03 09:30:00"], name="Datetime")
        return pd.DataFrame(
            {
                "Open": [470.0, 471.0],
                "High": [473.0, 474.0],
                "Low": [469.0, 470.0],
                "Close": [472.0, 473.0],
                "Volume": [100000.0, 110000.0],
            },
            index=idx,
        )

    yf_df = YFinanceFetcher(download_fn=mock_yf).fetch_ohlcv("SPY", timeframe="1d")
    assert yf_df.loc[0, "timestamp"] == pd.Timestamp("2024-01-02 14:30:00+00:00")

    def mock_vn(
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        source: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": ["2024-01-02 09:00:00", "2024-01-03 09:00:00"],
                "open": [95000.0, 96000.0],
                "high": [97000.0, 98000.0],
                "low": [94500.0, 95500.0],
                "close": [96500.0, 97500.0],
                "volume": [500000.0, 600000.0],
            }
        )

    vn_df = VNStockFetcher(history_fn=mock_vn).fetch_ohlcv("FPT", timeframe="1d")
    assert vn_df.loc[0, "timestamp"] == pd.Timestamp("2024-01-02 02:00:00+00:00")


def test_parquet_idempotent_storage_and_incremental_resume(tmp_path: Path) -> None:
    batch_1 = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                "open": [100.0, 102.0],
                "high": [103.0, 105.0],
                "low": [99.0, 101.0],
                "close": [102.0, 104.0],
                "volume": [10.0, 20.0],
            }
        ),
        symbol="BTC/USDT",
    )
    save_to_parquet(batch_1, symbol="BTC/USDT", timeframe="1d", raw_dir=tmp_path)

    # Second run overlaps 2024-01-02 with updated close and adds 2024-01-03
    batch_2 = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
                "open": [102.0, 104.5],
                "high": [106.0, 107.0],
                "low": [101.0, 103.0],
                "close": [105.0, 106.5],
                "volume": [25.0, 30.0],
            }
        ),
        symbol="BTC/USDT",
    )
    save_to_parquet(batch_2, symbol="BTC/USDT", timeframe="1d", raw_dir=tmp_path)

    loaded = load_from_parquet("BTC/USDT", "1d", raw_dir=tmp_path)
    assert len(loaded) == 3
    assert loaded.loc[1, "close"] == pytest.approx(105.0)
    assert get_latest_parquet_timestamp("BTC/USDT", "1d", raw_dir=tmp_path) == pd.Timestamp(
        "2024-01-03T00:00:00Z"
    )


def test_sql_upsert_idempotency(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    df_initial = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                "open": [100.0, 102.0],
                "high": [103.0, 105.0],
                "low": [99.0, 101.0],
                "close": [102.0, 104.0],
                "volume": [10.0, 20.0],
            }
        ),
        symbol="BTC/USDT",
    )
    upsert_to_postgres(df_initial, database_url=db_url)

    df_update = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ["2024-01-02T00:00:00Z"],
                "open": [102.0],
                "high": [108.0],
                "low": [101.0],
                "close": [107.0],
                "volume": [35.0],
            }
        ),
        symbol="BTC/USDT",
    )
    upsert_to_postgres(df_update, database_url=db_url)

    engine = create_engine(db_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT timestamp, close FROM ohlcv_bars ORDER BY timestamp ASC")
        ).fetchall()

    assert len(rows) == 2
    assert rows[1][1] == pytest.approx(107.0)


def test_cli_incremental_workflow_and_source_inference(tmp_path: Path) -> None:
    assert infer_source("BTC/USDT") == "ccxt"
    assert infer_source("VNINDEX") == "vnstock"
    assert infer_source("SPY") == "yfinance"

    captured_starts: list[object] = []

    class FakeFetcher:
        def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "1d",
            start: pd.Timestamp | str | None = None,
            end: pd.Timestamp | str | None = None,
            drop_unclosed: bool = True,
        ) -> pd.DataFrame:
            captured_starts.append(start)
            if start is None:
                return normalize_ohlcv(
                    pd.DataFrame(
                        {
                            "timestamp": ["2024-01-01T00:00:00Z"],
                            "open": [100.0],
                            "high": [102.0],
                            "low": [99.0],
                            "close": [101.0],
                            "volume": [10.0],
                        }
                    ),
                    symbol=symbol,
                )
            return normalize_ohlcv(
                pd.DataFrame(
                    {
                        "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                        "open": [100.0, 101.0],
                        "high": [102.0, 104.0],
                        "low": [99.0, 100.0],
                        "close": [101.0, 103.0],
                        "volume": [10.0, 15.0],
                    }
                ),
                symbol=symbol,
            )

    fake = FakeFetcher()
    run_ingestion(symbol="BTC/USDT", timeframe="1d", raw_dir=tmp_path, fetcher=fake)
    run_ingestion(symbol="BTC/USDT", timeframe="1d", raw_dir=tmp_path, fetcher=fake)

    assert captured_starts[0] is None
    assert captured_starts[1] == pd.Timestamp("2024-01-01T00:00:00Z")
    stored = load_from_parquet("BTC/USDT", "1d", raw_dir=tmp_path)
    assert len(stored) == 2
