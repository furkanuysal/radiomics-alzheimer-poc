import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"
MODEL_SAVE_PATH = "outputs/models/hippo_right_texture_model.pkl"

def main():
    print("=== FINAL DIAGNOSTIC MODEL TRAINING (RIGHT HIPPOCAMPUS) ===")

    df = pd.read_csv(DATA_PATH)

    # Binary label: 0 = CN, 1 = AD
    df = df[df["cdr"].isin([0, 0.5, 1, 2])]
    df["label"] = df["cdr"].apply(lambda x: 0 if x == 0 else 1)

    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values

    # Select only texture + firstorder features
    texture_cols = [c for c in X.columns if "firstorder" in c or "gl" in c]
    X = X[texture_cols]

    print("Total samples:", len(df))
    print("Selected texture features:", len(texture_cols))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(score_func=f_classif, k=50)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_test_sel = selector.transform(X_test_scaled)

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train_sel, y_train)

        y_pred = model.predict(X_test_sel)
        y_proba = model.predict_proba(X_test_sel)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"Fold {fold} | Acc={acc:.3f} | AUC={auc:.3f}")

        acc_scores.append(acc)
        auc_scores.append(auc)

    print("\n=== MEAN PERFORMANCE ===")
    print("Accuracy:", np.mean(acc_scores))
    print("AUC:", np.mean(auc_scores))

    # Train final model on full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=50)
    X_sel = selector.fit_transform(X_scaled, y)

    final_model = GradientBoostingClassifier(random_state=42)
    final_model.fit(X_sel, y)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    joblib.dump({
        "scaler": scaler,
        "selector": selector,
        "model": final_model,
        "features": texture_cols
    }, MODEL_SAVE_PATH)

    print("\nModel saved to:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
