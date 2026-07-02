"""Data loading, validation, and quality reporting."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "screening_hitrate.csv"

EXPECTED_SHAPE = (480, 15)

INACTIVE_SERIES = ["ListB_field5"]
PROBLEMATIC_SERIES = ["ListB_field6"]

KNOWN_HOLIDAYS = {
    pd.Timestamp("2023-12-25", tz="UTC"),
    pd.Timestamp("2024-01-01", tz="UTC"),
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the screening hit rate CSV.

    Returns DataFrame with UTC DatetimeIndex plus hour_of_day and day_of_week columns.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected {EXPECTED_SHAPE}, got {df.shape}")

    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["is_holiday"] = df.index.normalize().isin(KNOWN_HOLIDAYS)

    assert df["day_of_week"].max() <= 4, "Weekend rows found"

    return df


def get_series_cols(df: pd.DataFrame) -> list[str]:
    """Return the 15 hit rate series columns only."""
    meta = {"hour_of_day", "day_of_week", "is_holiday"}
    return [c for c in df.columns if c not in meta]


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series quality summary."""
    cols = get_series_cols(df)
    data = df[cols]
    return pd.DataFrame(
        {
            "n_missing": data.isnull().sum(),
            "pct_missing": (data.isnull().sum() / len(data) * 100).round(1),
            "n_zero": (data == 0).sum(),
            "pct_zero": ((data == 0).sum() / len(data) * 100).round(1),
            "n_nonzero": (data > 0).sum(),
            "max_value": data.max(),
            "median_nonzero": data[data > 0].median(),
        }
    )
