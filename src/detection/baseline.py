"""Hourly baseline fitting: robust median and MAD per series × hour-of-day."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.load import INACTIVE_SERIES

MAD_CONSISTENCY_FACTOR = 1.4826
MAD_FLOOR = 1e-6
MIN_OBS_FOR_BASELINE = 3


def _mad(values: np.ndarray) -> float:
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def fit_baseline(df: pd.DataFrame, series_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Fit per-series, per-hour-of-day baseline statistics.

    Returns dict:
        'median'  : DataFrame (24 × n_series)
        'mad'     : DataFrame (24 × n_series), scaled by 1.4826
        'count'   : DataFrame (24 × n_series)
        'low_conf': DataFrame (24 × n_series), bool — True when count < MIN_OBS
    """
    active_cols = [c for c in series_cols if c not in INACTIVE_SERIES]

    medians, mads, counts = {}, {}, {}

    for col in active_cols:
        col_median, col_mad, col_count = {}, {}, {}

        for hour in range(24):
            mask = df["hour_of_day"] == hour
            vals = df.loc[mask, col].dropna().values

            col_count[hour] = len(vals)

            if len(vals) == 0:
                col_median[hour] = np.nan
                col_mad[hour] = np.nan
            else:
                col_median[hour] = float(np.median(vals))
                col_mad[hour] = max(_mad(vals) * MAD_CONSISTENCY_FACTOR, MAD_FLOOR)

        medians[col] = col_median
        mads[col] = col_mad
        counts[col] = col_count

    median_df = pd.DataFrame(medians).rename_axis("hour_of_day")
    mad_df = pd.DataFrame(mads).rename_axis("hour_of_day")
    count_df = pd.DataFrame(counts).rename_axis("hour_of_day")

    return {
        "median": median_df,
        "mad": mad_df,
        "count": count_df,
        "low_conf": count_df < MIN_OBS_FOR_BASELINE,
    }


def save_baseline(baseline: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline["median"].to_csv(out_dir / "baseline_median.csv")
    baseline["mad"].to_csv(out_dir / "baseline_mad.csv")
    baseline["count"].to_csv(out_dir / "baseline_count.csv")
