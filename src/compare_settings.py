import os
import pandas as pd
import numpy as np


def main():
    base_path = os.path.join("outputs", "features")

    baseline_csv = os.path.join(base_path, "features_resampled_1mm.csv")
    resampled_csv = os.path.join(base_path, "features_resampled_bin50.csv")

    df_base = pd.read_csv(baseline_csv)
    df_res = pd.read_csv(resampled_csv)

    # Merge on feature name
    df = df_base.merge(
        df_res,
        on="feature",
        suffixes=("_baseline", "_resampled")
    )

    # Force numeric conversion, non-numeric -> NaN
    df["value_baseline"] = pd.to_numeric(df["value_baseline"], errors="coerce")
    df["value_resampled"] = pd.to_numeric(df["value_resampled"], errors="coerce")

    # Drop non-numeric features
    df_numeric = df.dropna(subset=["value_baseline", "value_resampled"])

    # Percentage change
    df_numeric = df_numeric.copy()
    df_numeric.loc[:, "percent_change"] = (
        (df_numeric["value_resampled"] - df_numeric["value_baseline"]).abs()
        / (df_numeric["value_baseline"].abs() + 1e-8)
    ) * 100

    df_sorted = df_numeric.sort_values(
        "percent_change", ascending=False
    )

    output_path = os.path.join(
        "outputs", "comparisons", "binwidth_comparison.csv"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sorted.to_csv(output_path, index=False)

    print(f"Comparison saved to {output_path}")
    print("\nTop 10 most affected numeric features:")
    print(df_sorted.head(10)[["feature", "percent_change"]])


if __name__ == "__main__":
    main()
