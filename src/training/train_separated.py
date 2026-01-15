import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score


# =========================
# CONFIG
# =========================

LEFT_PATH  = "outputs/features/hippocampus/feature_matrix_hippo_left.csv"
RIGHT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"

N_SPLITS = 5
K_BEST = 50
RANDOM_STATE = 42


# =========================
# UTILITIES
# =========================

def load_dataset(path: str):
    print(f"\nLoading: {path}")
    df = pd.read_csv(path)

    if "id" not in df.columns or "cdr" not in df.columns:
        raise ValueError("CSV must contain 'id' and 'cdr' columns.")

    # Binary label: 0 = CN, 1 = AD/MCI
    df = df[df["cdr"].isin([0.0, 0.5, 1.0, 2.0])]
    df["label"] = (df["cdr"] != 0.0).astype(int)

    print("CDR distribution:")
    print(df["cdr"].value_counts())
    print("Binary label distribution:")
    print(df["label"].value_counts())

    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values

    print(f"Feature matrix shape: {X.shape}")

    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    return acc, auc, sens, spec, cm


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=K_BEST)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])


def run_cv_experiment(name: str, X: pd.DataFrame, y: np.ndarray):
    print("\n" + "=" * 60)
    print(f"=== {name.upper()} HIPPOCAMPUS MODEL ===")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline()

    accs, aucs, senss, specs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n--- Fold {fold} ---")

        X_train = X.iloc[train_idx].values
        y_train = y[train_idx]
        X_test  = X.iloc[test_idx].values
        y_test  = y[test_idx]

        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc, auc, sens, spec, cm = compute_metrics(y_test, y_pred, y_prob)

        print(f"Accuracy    : {acc:.3f}")
        print(f"ROC-AUC     : {auc:.3f}")
        print(f"Sensitivity : {sens:.3f}")
        print(f"Specificity : {spec:.3f}")
        print("Confusion Matrix:")
        print(cm)

        accs.append(acc)
        aucs.append(auc)
        senss.append(sens)
        specs.append(spec)

    print("\n--- MEAN PERFORMANCE ---")
    print(f"Mean Accuracy    : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"Mean ROC-AUC     : {np.mean(aucs):.3f}")
    print(f"Mean Sensitivity : {np.mean(senss):.3f}")
    print(f"Mean Specificity : {np.mean(specs):.3f}")

    return {
        "acc": np.mean(accs),
        "auc": np.mean(aucs),
        "sens": np.mean(senss),
        "spec": np.mean(specs)
    }


# =========================
# MAIN
# =========================

def main():
    print("=== SEPARATED HIPPOCAMPUS TRAINING (LEFT vs RIGHT) ===")

    X_left, y_left = load_dataset(LEFT_PATH)
    X_right, y_right = load_dataset(RIGHT_PATH)

    print("\n\n########## RUNNING LEFT HIPPOCAMPUS ##########")
    left_metrics = run_cv_experiment("Left", X_left, y_left)

    print("\n\n########## RUNNING RIGHT HIPPOCAMPUS ##########")
    right_metrics = run_cv_experiment("Right", X_right, y_right)

    print("\n\n" + "#" * 60)
    print("=== FINAL COMPARISON ===")
    print("#" * 60)

    print(f"LEFT  -> Acc: {left_metrics['acc']*100:.2f}% | "
          f"AUC: {left_metrics['auc']:.3f} | "
          f"Sens: {left_metrics['sens']:.3f} | "
          f"Spec: {left_metrics['spec']:.3f}")

    print(f"RIGHT -> Acc: {right_metrics['acc']*100:.2f}% | "
          f"AUC: {right_metrics['auc']:.3f} | "
          f"Sens: {right_metrics['sens']:.3f} | "
          f"Spec: {right_metrics['spec']:.3f}")

    print("\nInterpretation hint:")
    print("- Higher AUC   -> more discriminative side")
    print("- Higher Sens  -> better at catching patients")
    print("- Higher Spec  -> better at rejecting healthy controls")


if __name__ == "__main__":
    main()
