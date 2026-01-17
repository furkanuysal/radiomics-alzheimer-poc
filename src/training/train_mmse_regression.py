import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# CONFIGURATION
DATA_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"

N_SPLITS = 5
RANDOM_STATE = 42
N_ESTIMATORS = 300

def main():
    print("=== MRI → MMSE REGRESSION (RIGHT HIPPOCAMPUS) ===")

    df = pd.read_csv(DATA_PATH)

    # --- Safety filters ---
    df = df.dropna(subset=["mmse"])
    df = df[df["mmse"].between(15, 30)]

    print(f"Total subjects: {len(df)}")
    print("MMSE distribution:")
    print(df["mmse"].describe())

    X = df.drop(columns=["id", "cdr", "mmse"])
    y = df["mmse"].values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    maes = []
    corrs = []

    print("\n--- Cross-Validation ---")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        corr, _ = spearmanr(y_test, y_pred)

        maes.append(mae)
        corrs.append(corr)

        print(f"\nFold {fold}")
        print(f"MAE        : {mae:.2f} MMSE points")
        print(f"Spearman r : {corr:.3f}")

    print("\n=== MEAN PERFORMANCE ===")
    print(f"Mean MAE        : {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"Mean Spearman r : {np.mean(corrs):.3f}")

    final_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X, y)

if __name__ == "__main__":
    main()
