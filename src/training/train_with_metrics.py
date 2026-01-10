import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    recall_score
)

INPUT_PATH = "outputs/features/feature_matrix_hippo.csv"

def run_cross_validation_with_metrics():
    print("=== ROC + Sensitivity + Specificity Analysis ===")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)

    # Cleaning
    df = df.drop(columns=[c for c in df.columns if "diagnostics" in c], errors="ignore")
    df = df.dropna(subset=["cdr"])

    y = (df["cdr"] > 0).astype(int)  # 1 = Alzheimer, 0 = Healthy
    X = df.drop(columns=[c for c in ["id", "cdr", "mmse"] if c in df.columns])

    print(f"Total Patients: {len(X)}")
    print(f"Number of Features: {X.shape[1]}")
    print("Class Distribution:")
    print(y.value_counts().rename({0: "Healthy", 1: "Alzheimer"}))
    print("-" * 50)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("var", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=15)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs = []
    rocs = []
    sensitivities = []
    specificities = []

    fold = 1

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        accs.append(acc)
        rocs.append(roc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        print(f"Fold {fold}")
        print(f"  Accuracy    : {acc:.3f}")
        print(f"  ROC-AUC     : {roc:.3f}")
        print(f"  Sensitivity : {sensitivity:.3f}")
        print(f"  Specificity : {specificity:.3f}")
        print(f"  Confusion Matrix:\n{cm}")
        print("-" * 30)

        fold += 1

    print("\n=== AVERAGE RESULTS ===")
    print(f"Mean Accuracy    : {np.mean(accs)*100:.2f}%")
    print(f"Mean ROC-AUC     : {np.mean(rocs):.3f}")
    print(f"Mean Sensitivity : {np.mean(sensitivities):.3f}")
    print(f"Mean Specificity : {np.mean(specificities):.3f}")
    print(f"Std Accuracy     : ±{np.std(accs)*100:.2f}%")
    print(f"Std ROC-AUC      : ±{np.std(rocs):.3f}")

if __name__ == "__main__":
    run_cross_validation_with_metrics()
