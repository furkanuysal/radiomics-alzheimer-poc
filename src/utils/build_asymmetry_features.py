import pandas as pd
import os

LEFT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_left.csv"
RIGHT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"
OUTPUT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_asymmetry.csv"

def build_asymmetry():
    print("Loading left & right hippocampus feature sets...")

    left_df = pd.read_csv(LEFT_PATH)
    right_df = pd.read_csv(RIGHT_PATH)

    print(f"Left count : {len(left_df)}")
    print(f"Right count: {len(right_df)}")

    merged = left_df.merge(
        right_df,
        on="id",
        suffixes=("_left", "_right"),
        how="inner"
    )

    print(f"After intersection: {len(merged)} subjects")

    feature_cols = [c.replace("_left", "") for c in merged.columns if c.endswith("_left") and c not in ["cdr_left"]]

    asym_data = {}

    for base in feature_cols:
        asym_data[base] = (merged[f"{base}_left"] - merged[f"{base}_right"]).abs()

    asym_df = pd.DataFrame(asym_data)
    asym_df.insert(0, "cdr", merged["cdr_left"].values)
    asym_df.insert(0, "id", merged["id"].values)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    asym_df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 40)
    print("Asymmetry feature matrix created (clean version).")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Shape: {asym_df.shape}")

if __name__ == "__main__":
    build_asymmetry()
