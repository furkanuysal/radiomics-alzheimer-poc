import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

RIGHT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"

N_SPLITS = 5
K_BEST = 30
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(RIGHT_PATH)

    print("Original CDR distribution:")
    print(df["cdr"].value_counts())

    # Binary label
    df = df[df["cdr"].isin([0.0, 0.5, 1.0])]
    df["label"] = (df["cdr"] != 0.0).astype(int)

    print("\nBinary label distribution:")
    print(df["label"].value_counts())

    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values

    print("\nFull feature matrix shape:", X.shape)
    return X, y


def split_feature_groups(X: pd.DataFrame):
    shape_cols = [c for c in X.columns if c.startswith("original_shape")]
    texture_cols = [
        c for c in X.columns
        if (
            c.startswith("original_gl") or
            c.startswith("wavelet-") or
            c.startswith("original_firstorder")
        )
    ]

    print(f"\nDetected {len(shape_cols)} shape features")
    print(f"Detected {len(texture_cols)} texture+firstorder features")

    X_shape = X[shape_cols]
    X_texture = X[texture_cols]
    X_full = X.copy()

    return X_shape, X_texture, X_full


def run_cv(name, X, y):
    print("\n" + "=" * 60)
    print(f"=== {name} MODEL ===")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, aucs, senss, specs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=min(K_BEST, X.shape[1]))),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ))
        ])

        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0

        print(f"\n--- Fold {fold} ---")
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


def main():
    print("=== SHAPE vs TEXTURE ABLATION STUDY (RIGHT HIPPOCAMPUS) ===")

    X, y = load_data()
    X_shape, X_texture, X_full = split_feature_groups(X)

    results = {}

    results["SHAPE"] = run_cv("ONLY SHAPE", X_shape, y)
    results["TEXTURE"] = run_cv("ONLY TEXTURE + FIRSTORDER", X_texture, y)
    results["FULL"] = run_cv("SHAPE + TEXTURE (FULL)", X_full, y)

    print("\n" + "#" * 60)
    print("=== FINAL COMPARISON (Ablation Results) ===")
    print("#" * 60)
    print("Model     | Accuracy | AUC   | Sens  | Spec")
    print("---------------------------------------------")
    for k, v in results.items():
        print(f"{k:<9} | {v['acc']*100:6.2f}% | {v['auc']:.3f} | {v['sens']:.3f} | {v['spec']:.3f}")

    print("\nInterpretation Guide:")
    print("- If SHAPE > TEXTURE  → atrophy dominates (biologically strong signal)")
    print("- If TEXTURE > SHAPE  → microstructural changes dominate")
    print("- If FULL best        → combined effect (expected in mid-stage AD)")


if __name__ == "__main__":
    main()
