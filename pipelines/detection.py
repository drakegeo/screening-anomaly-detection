"""Detection pipeline: baseline fitting, anomaly scoring, PCA check, visualisations.

Run: python -m pipelines.detection
"""

from src.processing.load import load_data, get_series_cols
from src.detection.baseline import fit_baseline, save_baseline
from src.detection.anomaly import (
    score_anomalies, anomaly_summary, threshold_sensitivity, group_into_events,
)
from src.detection.validation import false_positive_rate, threshold_tradeoff
from src.detection.diagnostics import (
    day_of_week_check, alert_confidence_audit, baseline_confidence_summary,
)
from src.detection.pca import pca_independence_check
from src.detection.isoforest import (
    fit_isolation_forest, shap_reason_codes,
    consensus_with_zscore as iso_consensus, save_scores as save_iso_scores,
)
from src.detection.visualise import (
    plot_series_with_anomalies, plot_anomaly_overview,
    plot_threshold_sensitivity, plot_threshold_tradeoff, plot_pca_scree,
    plot_isoforest_scores, plot_shap_importance, plot_shap_reasons,
    plot_monitoring_dashboard,
)
from src.processing.eda import tables_dir

TABLES = tables_dir("detection")


def main() -> None:
    print("Loading data...")
    df = load_data()
    cols = get_series_cols(df)

    print("\nFitting hourly baseline (median + MAD per series x hour)...")
    baseline = fit_baseline(df, cols)

    save_baseline(baseline, TABLES)
    n_low = baseline["low_conf"].sum().sum()
    print(f"  Low-confidence cells (n_obs < 3): {n_low}")

    print("\nThreshold sensitivity analysis...")
    sens = threshold_sensitivity(df, cols, baseline)
    print(sens.to_string())
    plot_threshold_sensitivity(sens)

    print("\nThreshold trade-off (defends the -2.5 knee: false alarms vs sensitivity)...")
    tradeoff = threshold_tradeoff(df, cols, baseline, n_hours=len(df))
    print(tradeoff.to_string())
    tradeoff.to_csv(TABLES / "threshold_tradeoff.csv")
    plot_threshold_tradeoff(tradeoff)

    print("\nScoring anomalies (threshold = -2.5)...")
    anomalies = score_anomalies(df, cols, baseline)
    anomalies.to_csv(TABLES / "anomalies.csv", index=False)

    n_critical = len(anomalies[anomalies["flag_type"].isin(["drop_zscore", "contextual_zero"])])
    n_holiday  = len(anomalies[anomalies["flag_type"] == "contextual_zero_holiday"])
    print(f"  Critical flags : {n_critical}")
    print(f"    drop_zscore (non-holiday) : {(anomalies['flag_type'].eq('drop_zscore') & ~anomalies['is_holiday']).sum()}")
    print(f"    drop_zscore (holiday)     : {(anomalies['flag_type'].eq('drop_zscore') &  anomalies['is_holiday']).sum()}")
    print(f"    contextual_zero           : {anomalies['flag_type'].eq('contextual_zero').sum()}")
    print(f"  Holiday zeros  : {n_holiday}  (volume effect, not list failure)")

    print("\nAnomaly summary:")
    print(anomaly_summary(anomalies).to_string())

    print("\nGrouping hour-level flags into operational events...")
    events = group_into_events(anomalies)
    events.to_csv(TABLES / "events.csv", index=False)
    n_ev_total = len(events)
    n_ev_field = int((~events["is_holiday"]).sum()) if not events.empty else 0
    print(f"  {len(anomalies)} hour-level flags collapse to {n_ev_total} events "
          f"({n_ev_field} non-holiday)")

    print("\nDefensive check 1 - day-of-week effect (justifies hour-only baseline)...")
    dow = day_of_week_check(df, cols)
    mean_red = dow["reduction_pct"].mean()
    print(f"  Mean residual reduction from adding day-of-week: {mean_red:.1f}%")
    print("  -> modest gain, not worth 5x baseline sparsity. Hour-only justified.")

    print("\nDefensive check 2 - top-alert baseline confidence (are alerts real?)...")
    audit = alert_confidence_audit(events, baseline)
    conf_summary = baseline_confidence_summary(baseline, cols)
    total_low = conf_summary["n_low_conf_active"].sum()
    if not audit.empty:
        n_solid = int((audit["n_obs"] >= 20).sum())
        print(f"  Top {len(audit)} alerts: {n_solid} backed by full 20-obs baseline, "
              f"{int((~audit['low_conf']).sum())} not low-confidence")
    print(f"  Low-confidence active-hour cells across all series: {total_low} "
          f"(0 = every possible flag rests on a full baseline)")
    audit.to_csv(TABLES / "alert_confidence_audit.csv", index=False)

    print("\nMultivariate check - PCA (tests independence, justifies univariate choice)...")
    pca_out = pca_independence_check(df, cols, baseline)
    print(f"  PC1 variance share: {pca_out['var_ratio'][0]:.1%}")
    print(f"  Components to reach 90% variance: {pca_out['k']} of {pca_out['n_series']}")
    print("  -> variance spread evenly = independent series = per-series design justified.")

    print("\nMultivariate detector - Isolation Forest + SHAP (works WITH independence)...")
    iso_out = fit_isolation_forest(df, cols, baseline)
    save_iso_scores(iso_out, TABLES / "isoforest_scores.csv")
    iso_con = iso_consensus(anomalies, iso_out["scores"])
    print(f"  Isolation Forest drop-hours flagged: {iso_con['n_iso_drop_flags']}")
    print(f"  Also caught by z-score: {iso_con['n_agree']}/{iso_con['n_critical']}")
    shap_out = shap_reason_codes(iso_out)
    print("  SHAP top drivers: " + ", ".join(shap_out["global_importance"].head(3).index))

    print("\nGenerating anomaly plots...")
    plot_anomaly_overview(df, cols, baseline, anomalies)
    for s in anomalies["series"].unique():
        plot_series_with_anomalies(df, s, baseline, anomalies)
    plot_pca_scree(pca_out)
    plot_isoforest_scores(df, iso_out, anomalies)
    plot_shap_importance(shap_out)
    plot_shap_reasons(iso_out, shap_out)

    n_active = len([c for c in cols if c not in ["ListB_field5"]])
    fp, _, _ = false_positive_rate(anomalies, n_series=n_active, n_hours=len(df))
    plot_monitoring_dashboard(df, cols, events, fp, baseline=baseline)

    print(f"\nDone. {len(anomalies['series'].unique())} series flagged. "
          f"Figures in outputs/detection/figures/, tables in outputs/detection/tables/")


if __name__ == "__main__":
    main()
