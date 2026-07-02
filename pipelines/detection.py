"""Detection pipeline: baseline fitting, anomaly scoring, visualisations.

Run: python main_detection.py
Depends on: main_processing.py having been run first (data must be loadable).
"""

from pathlib import Path

from src.processing.load import load_data, get_series_cols
from src.detection.baseline import fit_baseline, save_baseline
from src.detection.anomaly import score_anomalies, anomaly_summary, threshold_sensitivity
from src.detection.visualise import plot_series_with_anomalies, plot_anomaly_overview, plot_threshold_sensitivity

OUTPUTS = Path("outputs")


def main() -> None:
    print("Loading data...")
    df = load_data()
    cols = get_series_cols(df)

    print("\nFitting hourly baseline (median + MAD per series x hour)...")
    baseline = fit_baseline(df, cols)
    save_baseline(baseline, OUTPUTS / "baseline")
    n_low = baseline["low_conf"].sum().sum()
    print(f"  Low-confidence cells (n_obs < 3): {n_low}")

    print("\nThreshold sensitivity analysis...")
    sens = threshold_sensitivity(df, cols, baseline)
    print(sens.to_string())
    plot_threshold_sensitivity(sens)

    print("\nScoring anomalies (threshold = -2.5)...")
    anomalies = score_anomalies(df, cols, baseline)
    anomalies.to_csv(OUTPUTS / "anomalies.csv", index=False)

    n_critical = len(anomalies[anomalies["flag_type"].isin(["drop_zscore", "contextual_zero"])])
    n_holiday  = len(anomalies[anomalies["flag_type"] == "contextual_zero_holiday"])
    print(f"  Critical flags : {n_critical}")
    print(f"    drop_zscore (non-holiday) : {(anomalies['flag_type'].eq('drop_zscore') & ~anomalies['is_holiday']).sum()}")
    print(f"    drop_zscore (holiday)     : {(anomalies['flag_type'].eq('drop_zscore') &  anomalies['is_holiday']).sum()}")
    print(f"    contextual_zero           : {anomalies['flag_type'].eq('contextual_zero').sum()}")
    print(f"  Holiday zeros  : {n_holiday}  (volume effect, not list failure)")

    print("\nAnomaly summary:")
    print(anomaly_summary(anomalies).to_string())

    print("\nGenerating anomaly plots...")
    plot_anomaly_overview(df, cols, baseline, anomalies)
    for s in anomalies["series"].unique():
        plot_series_with_anomalies(df, s, baseline, anomalies)

    print(f"\nDone. {len(anomalies['series'].unique())} series flagged. All outputs in outputs/")


if __name__ == "__main__":
    main()
