import os
import pandas as pd


def get_feature_group(feature_name: str) -> str:
    """
    Extract feature group from PyRadiomics feature name.
    Example:
    original_glcm_Contrast -> glcm
    original_firstorder_Mean -> firstorder
    """
    parts = feature_name.split("_")
    if len(parts) < 2:
        return "unknown"
    return parts[1]


def main():
    comparison_csv = os.path.join(
        "outputs", "comparisons", "binwidth_comparison.csv"
    )

    df = pd.read_csv(comparison_csv)

    # Assign feature group
    df["group"] = df["feature"].apply(get_feature_group)

    # Aggregate statistics per group
    summary = (
        df.groupby("group")["percent_change"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    output_path = os.path.join(
        "outputs", "comparisons", "binwidth_feature_group_summary.csv"
    )
    summary.to_csv(output_path, index=False)

    print("Feature group stability summary:\n")
    print(summary)


if __name__ == "__main__":
    main()
