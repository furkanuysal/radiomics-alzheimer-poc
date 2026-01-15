import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, roc_curve, f1_score
)

# --- Models ---
USE_XGBOOST_IF_AVAILABLE = True

def build_model(random_state=42):
    if USE_XGBOOST_IF_AVAILABLE:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1
            )
        except Exception:
            pass

    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=400,
        random_state=random_state
    )

# --- Paths ---
LEFT_PATH  = "outputs/features/hippocampus/feature_matrix_hippo_left.csv"
RIGHT_PATH = "outputs/features/hippocampus/feature_matrix_hippo_right.csv"
ASYM_PATH  = "outputs/features/hippocampus/feature_matrix_hippo_asymmetry.csv"

REPORT_DIR = "outputs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Params ---
ALLOWED_CDR = {0.0, 0.5, 1.0}
K_BEST = 50
N_SPLITS = 5
RANDOM_STATE = 42

# ---------------- Utilities ----------------

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns or "cdr" not in df.columns:
        raise ValueError("CSV must contain 'id' and 'cdr' columns.")
    df = df.copy()
    df = df.sort_values("id")
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df

def _prefix_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = {"id", "cdr"}
    rename_map = {c: f"{prefix}{c}" for c in df.columns if c not in keep}
    return df.rename(columns=rename_map)

def load_and_prepare():
    print("=== HIPPOCAMPUS FUSION + THRESHOLD OPT + BOOSTING ===")
    print("Loading CSVs...")

    left  = _sanitize_df(pd.read_csv(LEFT_PATH))
    right = _sanitize_df(pd.read_csv(RIGHT_PATH))
    asym  = _sanitize_df(pd.read_csv(ASYM_PATH))

    left  = _prefix_features(left,  "L_")
    right = _prefix_features(right, "R_")
    asym  = _prefix_features(asym,  "A_")

    df = left.merge(right, on=["id", "cdr"], how="inner")
    df = df.merge(asym, on=["id", "cdr"], how="inner")

    df = df[df["cdr"].isin(ALLOWED_CDR)].copy()
    df["label"] = (df["cdr"] != 0.0).astype(int)

    print("\nCDR distribution:")
    print(df["cdr"].value_counts().sort_index())
    print("\nBinary label distribution:")
    print(df["label"].value_counts())

    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values

    print(f"\nFusion feature matrix shape: {X.shape}  (rows, features)")
    return df, X, y

def choose_threshold_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])

def inner_threshold_search(pipeline, X_train, y_train):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(y_train), dtype=float)

    for tr_idx, va_idx in inner_cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        pipeline.fit(X_tr, y_tr)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            p = pipeline.predict_proba(X_va)[:, 1]
        else:
            scores = pipeline.decision_function(X_va)
            p = 1 / (1 + np.exp(-scores))

        oof_prob[va_idx] = p

    return choose_threshold_youden(y_train, oof_prob)

def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return acc, auc, sens, spec, cm

# ---------------- MAIN ----------------

def main():
    df, X, y = load_and_prepare()

    model = build_model(RANDOM_STATE)

    pipe = Pipeline([
        ("variance", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=K_BEST)),
        ("model", model),
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, aucs, senss, specs, thrs = [], [], [], [], []
    fold_reports = []

    print("\n--- 5-Fold CV (Fusion + Threshold Optimization) ---")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n--- Fold {fold} ---")

        X_train = X.iloc[train_idx].values
        y_train = y[train_idx]
        X_test  = X.iloc[test_idx].values
        y_test  = y[test_idx]

        thr = inner_threshold_search(pipe, X_train, y_train)
        thrs.append(thr)

        pipe.fit(X_train, y_train)

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))

        acc, auc, sens, spec, cm = compute_metrics(y_test, y_prob, thr)

        print(f"Threshold   : {thr:.3f}")
        print(f"Accuracy    : {acc:.3f}")
        print(f"ROC-AUC     : {auc:.3f}")
        print(f"Sensitivity : {sens:.3f}")
        print(f"Specificity : {spec:.3f}")
        print("Confusion Matrix:")
        print(cm)

        accs.append(acc); aucs.append(auc); senss.append(sens); specs.append(spec)

        fold_reports.append(
            f"Fold {fold} | Acc={acc:.3f} AUC={auc:.3f} Sens={sens:.3f} Spec={spec:.3f} Thr={thr:.3f}\n{cm}\n"
        )

    mean_report = f"""
=== MEAN PERFORMANCE ===
Mean Threshold  : {np.mean(thrs):.3f} ± {np.std(thrs):.3f}
Mean Accuracy   : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%
Mean ROC-AUC    : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}
Mean Sensitivity: {np.mean(senss):.3f}
Mean Specificity: {np.mean(specs):.3f}
"""

    print(mean_report)

    # ---- Interpretability ----
    pipe.fit(X.values, y)
    selector = pipe.named_steps["select"]
    mask = selector.get_support()
    selected_feature_names = X.columns[mask].tolist()

    top_features_text = "\nTop selected features:\n" + "\n".join(
        [f"{i+1}. {name}" for i, name in enumerate(selected_feature_names[:20])]
    )

    print(top_features_text)

    # ---- Save report ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"fusion_run_{ts}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== HIPPOCAMPUS FUSION + THRESHOLD + BOOSTING REPORT ===\n\n")
        f.write(f"Samples: {len(X)}\nFeatures (raw): {X.shape[1]}\nK_BEST: {K_BEST}\n\n")
        for r in fold_reports:
            f.write(r + "\n")
        f.write(mean_report + "\n")
        f.write(top_features_text + "\n")

    print(f"\n📄 Report saved to: {report_path}")

if __name__ == "__main__":
    main()
