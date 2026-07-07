"""Validation pipeline: synthetic drop-injection test.

Answers the question that cannot be answered any other way when there is no
ground truth: *does the detector actually catch drops?*

The approach is analytical injection from the baseline:

    obs_injected = hourly_median × (1 - drop_fraction)

We then score obs_injected with the *same* z-score formula used in production
and check whether it would have been flagged. Aggregating over all active
(series, hour-of-day) cells gives a recovery curve — detection rate as a
function of drop magnitude — without any randomness or synthetic time series.

Results are written to outputs/validation/ and figure 13 is saved.

Run: python -m pipelines.validation
"""

from src.processing.load import load_data, get_series_cols
from src.detection.baseline import fit_baseline
from src.detection.anomaly import score_anomalies
from src.detection.validation import (
    run_injection_test,
    recovery_rates,
    overall_recovery,
    coverage_summary,
    false_positive_rate,
    DROP_FRACTIONS,
)
from src.detection.visualise import plot_validation_recovery
from src.processing.eda import tables_dir


def main() -> None:
    print("Loading data and fitting baseline...")
    df = load_data()
    cols = get_series_cols(df)
    baseline = fit_baseline(df, cols)

    print("\nRunning anomaly detector (for FP-rate baseline)...")
    anomalies = score_anomalies(df, cols, baseline)
    n_series = len([c for c in cols if c not in ["ListB_field5"]])
    fp, n_fp, n_scored = false_positive_rate(anomalies, n_series=n_series, n_hours=len(df))
    print(f"  Observed FP rate on clean hours: {fp:.5f} ({fp*100:.3f}%)")
    print(f"  Non-holiday critical flags: {n_fp} / {n_scored} scored series-hours")

    print("\nRunning synthetic drop-injection test...")
    results = run_injection_test(baseline, cols)

    active_cells = results[results["active"] & ~results["excluded"]]
    print(f"  Active (series, hour) cells tested: {len(active_cells)}")

    rates = recovery_rates(results)
    overall = overall_recovery(results)
    stats = coverage_summary(results)
    stats["n_fp_flags"] = n_fp
    stats["n_scored_pairs"] = n_scored

    # Print recovery table
    print("\nRecovery rates by drop magnitude (across all active cells):\n")
    print(f"  {'Magnitude':>12}  {'Detection rate':>14}")
    print(f"  {'-'*12}  {'-'*14}")
    for f in DROP_FRACTIONS:
        key = f"{int(f * 100)}%"
        rate = stats.get(f"recovery_at_{int(f * 100)}pct", float("nan"))
        bar = "#" * int(rate * 30)
        print(f"  {key:>12}  {rate:>13.1%}  {bar}")

    print(f"\n  Median minimum detectable drop: {stats['median_min_detectable_drop']:.1%}")
    print(f"  Cells detectable at 50% drop : {stats['pct_cells_detectable_at_50pct']:.1%}")
    print(f"  Cells detectable at 25% drop : {stats['pct_cells_detectable_at_25pct']:.1%}")

    # Per-series breakdown
    print("\nPer-series recovery at 50% drop magnitude:")
    print(f"  {'Series':>22}  {'Active hrs':>10}  {'50% recovery':>12}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*12}")
    for series in rates.index:
        r50 = rates.loc[series, "50%"]
        n_hrs = int(rates.loc[series, "n_active_hours"])
        print(f"  {series:>22}  {n_hrs:>10}  {r50:>12.1%}")

    # Save outputs
    out_dir = tables_dir("validation")

    results.to_csv(out_dir / "injection_test_cells.csv", index=False)
    rates.to_csv(out_dir / "recovery_by_series.csv")
    overall.to_csv(out_dir / "overall_recovery.csv")

    import pandas as pd
    pd.DataFrame([stats]).to_csv(out_dir / "coverage_summary.csv", index=False)

    print("\nGenerating validation figure (13_validation.png)...")
    plot_validation_recovery(results, rates, overall, fp, stats)

    print(f"\nDone. Outputs in {out_dir}/")
    print("  injection_test_cells.csv  — per-(series, hour) detection results")
    print("  recovery_by_series.csv    — recovery rates per series")
    print("  overall_recovery.csv      — mean recovery per magnitude")
    print("  coverage_summary.csv      — headline stats")
    print("  outputs/validation/figures/13_validation.png  — recovery curve + coverage heatmap")


if __name__ == "__main__":
    main()
