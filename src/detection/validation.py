"""Synthetic drop-injection validation.

No ground truth exists for this dataset, so the only quantitative way to prove
the detector works is to inject *known* drops into clean hours and measure how
many are recovered. This converts "the flags look sensible" into a recovery
curve: detection rate vs drop magnitude.

Method (analytical injection from baseline)
-------------------------------------------
For every active (series, hour-of-day) cell — one where the hourly median is
meaningfully above zero and the baseline has sufficient observations — we
simulate what the z-score would be if the hit rate dropped by a fraction f of
the expected baseline:

    obs_injected  = median × (1 - f)
    z             = (obs_injected - median) / mad
                  = -f × median / mad

"Detected" if z < DROP_THRESHOLD (−2.5).

Injecting from the median rather than from a real observed value is deliberate:
it measures the detector's ability to catch a drop from the *expected* baseline,
unclouded by whether the day happened to be above or below average. It is the
exact analogue of asking "what drop size would an analyst have to engineer to
bypass this system?"

The minimum detectable drop fraction per cell is therefore:

    min_f = |DROP_THRESHOLD| × mad / median

When mad/median is small (high signal-to-noise), even a small drop is caught.
When mad/median is large (noisy hour), a large drop is needed to clear the
threshold.

False-positive baseline
-----------------------
The observed FP rate on clean (non-holiday, non-injected) data is reported from
the existing anomalies.csv. It is the operational baseline the detector already
commits to.
"""

import numpy as np
import pandas as pd

from src.processing.load import INACTIVE_SERIES, PROBLEMATIC_SERIES
from src.detection.anomaly import (
    DROP_THRESHOLD,
    DROP_THRESHOLD_WIDE,
    CONTEXTUAL_ZERO_MIN_MEDIAN,
)

# Drop fractions to test: 10% → 90% reduction from the expected baseline
DROP_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Minimum meaningful median for a cell to participate in the injection test.
# Cells below this threshold have no real baseline to drop from.
MIN_ACTIVE_MEDIAN = CONTEXTUAL_ZERO_MIN_MEDIAN  # reuse 0.005


def run_injection_test(
    baseline: dict[str, pd.DataFrame],
    series_cols: list[str],
) -> pd.DataFrame:
    """Analytical drop-injection test across all (series, hour-of-day) cells.

    For each active cell, computes the minimum detectable drop fraction and
    whether each test magnitude would be detected.

    Returns a long DataFrame with one row per (series, hour_of_day).
    Columns:
        series, hour_of_day, median, mad, snr,
        min_detectable_drop,
        detected_at_<N>pct  (bool, one per DROP_FRACTION),
        active, low_conf, excluded, threshold_used
    """
    med_df = baseline["median"]   # shape (24, n_series), MAD already ×1.4826
    mad_df = baseline["mad"]
    low_conf_df = baseline["low_conf"]

    records = []
    for series in series_cols:
        excluded = series in INACTIVE_SERIES
        is_wide = series in PROBLEMATIC_SERIES
        threshold = DROP_THRESHOLD_WIDE if is_wide else DROP_THRESHOLD

        for hour in range(24):
            m = float(med_df.loc[hour, series]) if series in med_df.columns else np.nan
            mad = float(mad_df.loc[hour, series]) if series in mad_df.columns else np.nan
            lc = bool(low_conf_df.loc[hour, series]) if series in low_conf_df.columns else True

            if np.isnan(m) or np.isnan(mad):
                continue

            active = (m > MIN_ACTIVE_MEDIAN) and (not excluded)
            snr = (m / mad) if mad > 0 else np.nan

            # minimum fraction that trips z < threshold (negative threshold)
            # z = -f*m/mad < threshold  =>  f > |threshold|*mad/m
            if active and mad > 0:
                min_det = float(min(abs(threshold) * mad / m, 1.0))
            else:
                min_det = np.nan

            row: dict = {
                "series": series,
                "hour_of_day": hour,
                "median": round(m, 6),
                "mad": round(mad, 6),
                "snr": round(float(snr), 3) if not np.isnan(snr) else np.nan,
                "min_detectable_drop": round(min_det, 4) if not np.isnan(min_det) else np.nan,
                "active": active,
                "low_conf": lc,
                "excluded": excluded,
                "threshold_used": threshold,
            }

            for f in DROP_FRACTIONS:
                col_name = f"detected_at_{int(f * 100)}pct"
                if active and mad > 0:
                    z = -f * m / mad
                    row[col_name] = bool(z < threshold)
                else:
                    row[col_name] = False

            records.append(row)

    return pd.DataFrame(records)


def recovery_rates(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recovery rates across all active cells, per series and magnitude.

    Returns DataFrame indexed by series with one column per DROP_FRACTION (as %).
    Also includes 'n_active_hours' (how many cells contributed per series).
    """
    active = results[results["active"] & ~results["excluded"]]

    rows = []
    for series, grp in active.groupby("series"):
        row: dict = {"series": series, "n_active_hours": len(grp)}
        for f in DROP_FRACTIONS:
            col_name = f"detected_at_{int(f * 100)}pct"
            row[f"{int(f * 100)}%"] = round(grp[col_name].mean(), 4) if col_name in grp else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("series")


def overall_recovery(results: pd.DataFrame) -> pd.Series:
    """Mean detection rate across ALL active (series, hour) cells per magnitude."""
    active = results[results["active"] & ~results["excluded"]]
    rates = {}
    for f in DROP_FRACTIONS:
        col_name = f"detected_at_{int(f * 100)}pct"
        rates[f"{int(f * 100)}%"] = active[col_name].mean() if col_name in active else np.nan
    return pd.Series(rates, name="overall_recovery")


def coverage_summary(results: pd.DataFrame) -> dict:
    """Key statistics for reporting: coverage, blind spots, median minimum drop."""
    active = results[results["active"] & ~results["excluded"]]
    n_cells = len(active)

    det_vals = active["min_detectable_drop"].dropna()

    stats = {
        "n_active_cells": n_cells,
        "n_total_cells": len(results[~results["excluded"]]),
        "pct_cells_active": round(n_cells / max(len(results[~results["excluded"]]), 1), 3),
        "median_min_detectable_drop": round(det_vals.median(), 3) if len(det_vals) else np.nan,
        "pct_cells_detectable_at_50pct": round((det_vals <= 0.50).mean(), 3) if len(det_vals) else np.nan,
        "pct_cells_detectable_at_25pct": round((det_vals <= 0.25).mean(), 3) if len(det_vals) else np.nan,
    }

    for f in DROP_FRACTIONS:
        col = f"detected_at_{int(f * 100)}pct"
        n_det = int(active[col].sum()) if col in active else 0
        stats[f"recovery_at_{int(f * 100)}pct"] = round(n_det / n_cells, 3) if n_cells else np.nan

    return stats


# Thresholds to compare when defending the -2.5 operating point.
THRESHOLD_GRID = [-2.0, -2.5, -3.0]


def threshold_tradeoff(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    n_hours: int = 480,
    magnitudes: tuple[float, ...] = (0.50, 0.90),
) -> pd.DataFrame:
    """Quantify the false-alarm vs sensitivity trade-off across candidate thresholds.

    For each threshold in THRESHOLD_GRID, measures both sides of the knee-point
    decision:

        false-alarm side : non-holiday z-score drop flags on the real data
                           (the base flag rate on presumed-clean hours)
        sensitivity side : analytical recovery of injected drops of each
                           magnitude, across all active (series, hour) cells

    Recovery is closed-form: a drop of fraction f in a cell trips z < t exactly
    when snr > |t| / f, so the recovery rate is the share of active cells whose
    signal-to-noise clears that bar. No simulation or randomness.

    Returns a DataFrame indexed by threshold with columns:
        n_fp_flags, fp_rate, recovery_at_<N>  (one per magnitude, as a fraction)
    """
    from src.detection.anomaly import score_anomalies

    cells = run_injection_test(baseline, series_cols)
    active = cells[cells["active"] & ~cells["excluded"]]
    snr = active["snr"].dropna().to_numpy()

    n_series = len([c for c in series_cols if c not in INACTIVE_SERIES])
    total_pairs = n_series * n_hours

    rows = []
    for t in THRESHOLD_GRID:
        a = score_anomalies(df, series_cols, baseline, drop_threshold=t)
        n_fp = int(((a["flag_type"] == "drop_zscore") & (~a["is_holiday"])).sum())
        row: dict = {
            "threshold": t,
            "n_fp_flags": n_fp,
            "fp_rate": round(n_fp / total_pairs, 5),
        }
        for f in magnitudes:
            # detected in a cell iff snr * f > |t|  (i.e. z = -f*snr < t)
            row[f"recovery_at_{int(f * 100)}"] = (
                round(float((snr * f > abs(t)).mean()), 4) if len(snr) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("threshold")


def false_positive_rate(
    anomalies: pd.DataFrame, n_series: int, n_hours: int = 480
) -> tuple[float, int, int]:
    """Observed FP rate: non-holiday z-score drops / total scored (series, hour) pairs.

    Returns (rate, n_fp_flags, n_scored_pairs).
    """
    non_holiday_flags = anomalies[
        (anomalies["flag_type"] == "drop_zscore") & (~anomalies["is_holiday"])
    ]
    total_pairs = n_series * n_hours
    n_fp = len(non_holiday_flags)
    return round(n_fp / total_pairs, 5), n_fp, total_pairs
