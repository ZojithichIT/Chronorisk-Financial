"""Central configuration and annualization helpers for ChronoRisk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

AssetClass = Literal["crypto", "us_equity", "vn_equity"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TRADING_DAYS_PER_YEAR: dict[AssetClass, int] = {
    "crypto": 365,
    "us_equity": 252,
    "vn_equity": 252,
}

_TRADING_HOURS_PER_DAY: dict[AssetClass, float] = {
    "crypto": 24.0,
    "us_equity": 6.5,
    "vn_equity": 4.5,
}

_TIMEFRAME_HOURS: dict[str, float] = {
    "1d": 24.0,
    "4h": 4.0,
    "1h": 1.0,
    "30m": 0.5,
    "20m": 1.0 / 3.0,
    "15m": 0.25,
    "10m": 1.0 / 6.0,
    "5m": 5.0 / 60.0,
    "1m": 1.0 / 60.0,
}


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="CHRONORISK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default=_PROJECT_ROOT)
    database_url: str | None = Field(default=None)
    default_exchange: str = Field(default="binance")
    log_level: str = Field(default="INFO")
    seed: int = Field(default=42)
    trading_fees: dict[str, float] = Field(
        default_factory=lambda: {
            "crypto": 0.001,
            "vn_equity": 0.0015,
            "us_equity": 0.0005,
        }
    )

    @property
    def data_dir(self) -> Path:
        """Return the base data directory."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Return the directory for raw normalized OHLCV Parquet files."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Return the directory for engineered feature datasets."""
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        """Return the directory for saved trained model artifacts."""
        return self.project_root / "artifacts"


def get_annualization_factor(
    asset_class: AssetClass,
    timeframe: str = "1d",
) -> float:
    """Compute annualization bars-per-year factor for a given asset class and timeframe.

    Args:
        asset_class: Market category (``"crypto"``, ``"us_equity"``, or ``"vn_equity"``).
        timeframe: Bar interval string (``"1d"``, ``"4h"``, ``"1h"``, ``"30m"``, ``"20m"``,
            ``"15m"``, ``"10m"``, ``"5m"``, or ``"1m"``).

    Returns:
        Number of bars per year used to scale per-bar volatility and Sharpe ratio.
    """
    if asset_class not in _TRADING_DAYS_PER_YEAR:
        raise ValueError(f"Unsupported asset_class: {asset_class!r}")
    tf = timeframe.strip().lower()
    if tf not in _TIMEFRAME_HOURS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    trading_days = float(_TRADING_DAYS_PER_YEAR[asset_class])
    if tf == "1d":
        return trading_days

    bars_per_day = _TRADING_HOURS_PER_DAY[asset_class] / _TIMEFRAME_HOURS[tf]
    return trading_days * bars_per_day


def get_settings() -> Settings:
    """Return a fresh ``Settings`` instance."""
    return Settings()
