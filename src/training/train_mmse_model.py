import os
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

FEATURE_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"
MODEL_PATH = "outputs/models/hippo_mmse_regression.pkl"

def main():
    print("Loading feature matrix...")
    df = pd.read_csv(FEATURE_PATH)

    print("Total samples:", len(df))

    df = df.dropna(subset=["mmse"])
    print("After dropping NaN MMSE:", len(df))

    X = df.drop(columns=["id", "cdr", "mmse"])
    y = df["mmse"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k=100)),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.7))
    ])

    print("\n--- Cross-Validation ---")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    maes = []
    cors = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        corr, _ = spearmanr(y_test, preds)

        maes.append(mae)
        cors.append(corr)

        print(f"Fold {fold} | MAE={mae:.2f} | Spearman={corr:.3f}")

    print("\n=== MEAN PERFORMANCE ===")
    print("Mean MAE:", round(sum(maes)/len(maes), 2))
    print("Mean Spearman:", round(sum(cors)/len(cors), 3))

    print("\nTraining final model on all data...")
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": list(X.columns)
    }, MODEL_PATH)

    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
