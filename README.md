# Screening Hit Rate Anomaly Detection

Detects drop anomalies in sanctions screening hit rate data. A drop anomaly is an hour where the hit rate for a given list/field falls well below what is normal for that hour of day. The point is early warning: if a sanctions list is truncated or corrupted, its hit rate falls silently and transactions that should be flagged can pass through unscreened. Drops matter more than spikes, which are noisy but safe.

## Data

-   480 hourly rows, 20 weekdays, 2023-12-18 to 2024-01-12 (UTC). Weekends excluded.
-   15 series = 5 sanctions lists x 3 message fields each.
-   Hit rate = total hits / total screening requests. It can exceed 1, because one request can match several list entries.
-   A zero means no requests arrived that hour (not an anomaly). A NaN means data collection failed (a separate operational issue).

The raw CSV lives at `data/screening_hitrate.csv`.

## Approach

The method is a robust per-series, per-hour-of-day baseline. For each series and each hour of the day we take the 20 weekday values, compute the median and the median absolute deviation (MAD), and score every observation:

```
z = (value - hourly_median) / (hourly_MAD * 1.4826)
```

An hour is flagged as a drop when `z < -2.5`. Median and MAD are used instead of mean and standard deviation because the hit rate distributions are skewed and zero-heavy, so robust statistics are not dragged around by outliers or holiday spikes. The baseline is hour-specific because 03:00 and 11:00 behave very differently, and a single global mean would produce constant false alarms overnight.

A second check (PCA) confirms the 15 series are effectively independent, which is why each series is modelled on its own rather than jointly. An Isolation Forest with SHAP reason codes is included as a second, independent view on the same data.

Special cases:

-   `ListB_field5` is inactive (always zero) and is excluded.
-   `ListB_field6` has high missingness and is treated with a wider threshold.
-   Public holidays (Dec 25, Jan 1) are tagged; zeros on those days are treated as a volume effect rather than a list failure.

## Setup

Requires Python 3.10+. Using Poetry:

```
poetry install
```

Or with pip:

```
pip install -e .
```

## Running

```
poetry run python -m pipelines.processing    # data checks + EDA plots
poetry run python -m pipelines.detection      # baseline, scoring, plots, results
poetry run python -m pipelines.validation     # synthetic drop-injection test
```

(Drop `poetry run` if you installed with pip and activated the environment.)

## Outputs

Each pipeline writes into its own folder under `outputs/`, split into `figures/`
(plots) and `tables/` (CSV results):

```
outputs/
  processing/   figures/  EDA charts (time series, diurnal profiles, data quality)
                tables/   data_quality_report.csv
  detection/    figures/  per-series anomaly charts, PCA, isolation forest, dashboard
                tables/   anomalies.csv, events.csv, baseline_*.csv, ...
  validation/   figures/  recovery curve + coverage heatmap
                tables/   injection and recovery results
```

Key result tables: `detection/tables/anomalies.csv` (every flagged hour with its
z-score and flag type) and `detection/tables/events.csv` (flags grouped into
operational events, so a sustained multi-hour drop is one event, not many alerts).

## Project structure

```
src/processing/   data loading, validation, EDA plots
src/detection/    baseline, anomaly scoring, PCA check, isolation forest, plots
pipelines/        entry points (processing, detection, validation)
data/             raw CSV (not committed)
outputs/          results, organised per pipeline (see above)
```

## Limitations

-   Only the hit rate ratio is available, not the underlying request volume. Without request counts we cannot fully separate a genuine drop from low-volume noise, which is why overnight hours are noisier. Logging request counts would be the single most useful addition.
-   There are no ground-truth labels, so the detector is validated by injecting known drops and measuring how many are recovered, not by precision/recall.
-   Only 20 observations per hour per series, so baselines carry some uncertainty at sparse hours.