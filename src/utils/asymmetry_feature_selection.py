import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def prepare_asymmetry_features(
    csv_path: str,
    k: int = 50,
    drop_mmse: bool = True,
    verbose: bool = True
):
    """
    Loads asymmetry feature CSV, creates binary labels, scales features,
    applies SelectKBest feature selection, and returns processed matrices.

    Parameters:
        csv_path (str): Path to asymmetry feature CSV.
        k (int): Number of top features to select.
        drop_mmse (bool): Whether to drop MMSE column if exists.
        verbose (bool): Print debug info.

    Returns:
        X_selected (np.ndarray): Selected feature matrix.
        y (np.ndarray): Binary labels.
        selected_feature_names (list): Names of selected features.
        full_feature_names (list): All original feature names (after drop).
    """

    if verbose:
        print("Loading asymmetry feature matrix...")

    df = pd.read_csv(csv_path)

    if verbose:
        print("Original CDR distribution:")
        print(df["cdr"].value_counts())

    # Safety filter
    df = df[df["cdr"].isin([0, 0.5, 1, 2])]

    # Binary label: 0 = CN, 1 = AD
    df["label"] = df["cdr"].apply(lambda x: 0 if x == 0 else 1)

    if verbose:
        print("\nBinary label distribution:")
        print(df["label"].value_counts())

    drop_cols = ["id", "cdr", "label"]
    if drop_mmse and "mmse" in df.columns:
        drop_cols.append("mmse")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["label"].values

    if verbose:
        print(f"\nFeature matrix shape: {X.shape}")

    full_feature_names = X.columns.tolist()

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Selection
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    selected_mask = selector.get_support()
    selected_feature_names = list(np.array(full_feature_names)[selected_mask])

    if verbose:
        print(f"Selected feature shape: {X_selected.shape}")
        print("\nTop selected features:")
        for i, f in enumerate(selected_feature_names[:15], 1):
            print(f"{i}. {f}")

        shape_count = sum(1 for f in selected_feature_names if "shape" in f.lower())
        if shape_count > 0:
            print(f"\n✨ {shape_count} shape-based features selected (VERY GOOD sign).")

    return X_selected, y, selected_feature_names, full_feature_names
