"""Data processing pipeline: load, validate, EDA plots, quality report.

Run: python main_processing.py
"""

from src.processing.load import load_data, get_series_cols, data_quality_report
from src.processing.eda import (
    tables_dir,
    plot_time_series_overview,
    plot_diurnal_heatmap,
    plot_data_quality_map,
    plot_hourly_profiles,
    plot_distributions,
    plot_skew_mean_vs_median,
    plot_correlation,
)


def main() -> None:
    print("Loading data...")
    df = load_data()
    cols = get_series_cols(df)
    print(f"  {df.shape[0]} rows x {len(cols)} series | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Holidays: {sorted(df[df['is_holiday']].index.normalize().unique())}")

    print("\nData quality report:")
    report = data_quality_report(df)
    print(report.to_string())
    report.to_csv(tables_dir("processing") / "data_quality_report.csv")

    print("\nGenerating EDA plots...")
    plot_time_series_overview(df, cols)
    plot_diurnal_heatmap(df, cols)
    plot_data_quality_map(df, cols)
    plot_hourly_profiles(df, cols)
    plot_distributions(df, cols)
    plot_skew_mean_vs_median(df, cols)
    plot_correlation(df, cols)

    print("\nDone. EDA outputs saved to outputs/processing/")


if __name__ == "__main__":
    main()
