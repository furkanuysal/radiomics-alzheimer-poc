import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from utils.asymmetry_feature_selection import prepare_asymmetry_features


DATA_PATH = "outputs/features/hippocampus/feature_matrix_hippo_asymmetry.csv"


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm
    }


def main():
    print("\n=== ASYMMETRY-BASED ALZHEIMER CLASSIFICATION ===")

    # --- 1. Feature Preparation ---
    X, y, selected_features, _ = prepare_asymmetry_features(
        csv_path=DATA_PATH,
        k=50,
        verbose=True
    )

    print("\n--- Model Training (5-Fold Stratified CV) ---")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    all_importances = np.zeros(len(selected_features))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_results.append(metrics)

        all_importances += model.feature_importances_

        print(f"Accuracy    : {metrics['accuracy']:.3f}")
        print(f"ROC-AUC     : {metrics['roc_auc']:.3f}")
        print(f"Sensitivity : {metrics['sensitivity']:.3f}")
        print(f"Specificity : {metrics['specificity']:.3f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

    # --- 2. Aggregate Results ---
    print("\n=== MEAN PERFORMANCE (5-Fold CV) ===")

    mean_acc = np.mean([m["accuracy"] for m in fold_results])
    mean_auc = np.mean([m["roc_auc"] for m in fold_results])
    mean_sens = np.mean([m["sensitivity"] for m in fold_results])
    mean_spec = np.mean([m["specificity"] for m in fold_results])

    std_acc = np.std([m["accuracy"] for m in fold_results])
    std_auc = np.std([m["roc_auc"] for m in fold_results])

    print(f"Mean Accuracy    : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Mean ROC-AUC     : {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Mean Sensitivity : {mean_sens:.3f}")
    print(f"Mean Specificity : {mean_spec:.3f}")

    # --- 3. Feature Importance ---
    print("\n=== TOP 20 MOST IMPORTANT ASYMMETRY FEATURES ===")

    mean_importances = all_importances / 5
    importance_df = pd.DataFrame({
        "feature": selected_features,
        "importance": mean_importances
    }).sort_values(by="importance", ascending=False)

    for i, row in importance_df.head(20).iterrows():
        print(f"{row['feature']:<60} {row['importance']:.5f}")

    shape_count = sum(1 for f in importance_df.head(20)["feature"] if "shape" in f.lower())
    if shape_count > 0:
        print(f"\n✨ {shape_count} SHAPE-based features in top 20 (this is extremely meaningful biologically).")

    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
