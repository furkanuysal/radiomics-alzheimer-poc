import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

# ---------------- PATHS ----------------
LEFT_PATH  = "outputs/features/hippocampus/feature_matrix_hippo_left.csv"
RIGHT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"
ASYM_PATH  = "outputs/features/hippocampus/feature_matrix_hippo_asymmetry.csv"
FULL_PATH = "outputs/features/hippocampus/feature_matrix_hippo.csv"

ALLOWED_CDR = {0.0, 0.5, 1.0}
K_BEST = 50
N_SPLITS = 5
RANDOM_STATE = 42


def load_basic(path):
    df = pd.read_csv(path)
    df = df[df["cdr"].isin(ALLOWED_CDR)].copy()
    df["label"] = (df["cdr"] != 0.0).astype(int)
    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values
    return X, y


def load_fusion():
    left = pd.read_csv(LEFT_PATH)
    right = pd.read_csv(RIGHT_PATH)
    asym = pd.read_csv(ASYM_PATH)

    for df in [left, right, asym]:
        df = df[df["cdr"].isin(ALLOWED_CDR)]

    # prefix
    left = left.rename(columns=lambda c: f"L_{c}" if c not in ["id", "cdr"] else c)
    right = right.rename(columns=lambda c: f"R_{c}" if c not in ["id", "cdr"] else c)
    asym = asym.rename(columns=lambda c: f"A_{c}" if c not in ["id", "cdr"] else c)

    df = left.merge(right, on=["id", "cdr"]).merge(asym, on=["id", "cdr"])
    df["label"] = (df["cdr"] != 0.0).astype(int)

    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values
    return X, y


def run_model(name, X, y):
    print(f"\n===== {name} =====")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=K_BEST)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, aucs = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        accs.append(acc)
        aucs.append(auc)

        print(f"Fold {fold}: Acc={acc:.3f} | AUC={auc:.3f}")

    print(f">>> MEAN {name}: Acc={np.mean(accs)*100:.2f}% | AUC={np.mean(aucs):.3f}")

    return np.mean(accs), np.mean(aucs)


def main():
    results = []

    X_L, y_L = load_basic(LEFT_PATH)
    acc, auc = run_model("LEFT HIPPOCAMPUS", X_L, y_L)
    results.append(("LEFT", acc, auc))

    X_R, y_R = load_basic(RIGHT_PATH)
    acc, auc = run_model("RIGHT HIPPOCAMPUS", X_R, y_R)
    results.append(("RIGHT", acc, auc))

    X_A, y_A = load_basic(ASYM_PATH)
    acc, auc = run_model("ASYMMETRY", X_A, y_A)
    results.append(("ASYMMETRY", acc, auc))

    X_Fu, y_Fu = load_basic(FULL_PATH)
    acc, auc = run_model("FULL HIPPOCAMPUS", X_Fu, y_Fu)
    results.append(("FULL", acc, auc))

    X_F, y_F = load_fusion()
    acc, auc = run_model("FUSION (L+R+A)", X_F, y_F)
    results.append(("FUSION", acc, auc))

    print("\n================ FINAL COMPARISON ================")
    print("Model        | Accuracy | AUC")
    print("---------------------------------")
    for name, acc, auc in results:
        print(f"{name:<12} | {acc*100:6.2f}% | {auc:.3f}")


if __name__ == "__main__":
    main()
