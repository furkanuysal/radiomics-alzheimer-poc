import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, roc_curve, f1_score
)

# --- Models (If XGBoost available, use it, otherwise sklearn boosting) ---
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

    # Fallback (sklearn)
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

# --- Label rule ---
ALLOWED_CDR = {0.0, 0.5, 1.0}
K_BEST = 50
N_SPLITS = 5
RANDOM_STATE = 42

# ---------------- Utilities ----------------

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: ensure id/cdr exist, drop duplicates by id."""
    if "id" not in df.columns or "cdr" not in df.columns:
        raise ValueError("CSV must contain 'id' and 'cdr' columns.")
    df = df.copy()
    df = df.sort_values("id")
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df

def _prefix_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix feature columns except id/cdr."""
    df = df.copy()
    keep = {"id", "cdr"}
    rename_map = {c: f"{prefix}{c}" for c in df.columns if c not in keep}
    return df.rename(columns=rename_map)

def load_and_prepare():
    print("=== HIPPOCAMPUS FUSION + THRESHOLD OPT + BOOSTING ===")
    print("Loading CSVs...")

    left  = _sanitize_df(pd.read_csv(LEFT_PATH))
    right = _sanitize_df(pd.read_csv(RIGHT_PATH))
    asym  = _sanitize_df(pd.read_csv(ASYM_PATH))

    # Prefix feature columns to avoid name collisions
    left  = _prefix_features(left,  "L_")
    right = _prefix_features(right, "R_")
    asym  = _prefix_features(asym,  "A_")

    # Merge on id
    df = left.merge(right, on=["id", "cdr"], how="inner")
    df = df.merge(asym, on=["id", "cdr"], how="inner")

    # Filter by CDR subset
    df = df[df["cdr"].isin(ALLOWED_CDR)].copy()

    # Binary label
    df["label"] = (df["cdr"] != 0.0).astype(int)

    print("\nCDR distribution:")
    print(df["cdr"].value_counts().sort_index())
    print("\nBinary label distribution:")
    print(df["label"].value_counts())

    # Build X/y
    X = df.drop(columns=["id", "cdr", "label"])
    y = df["label"].values

    print(f"\nFusion feature matrix shape: {X.shape}  (rows, features)")
    return df, X, y

def choose_threshold_youden(y_true, y_prob):
    """Threshold that maximizes Youden's J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx])

def choose_threshold_f1(y_true, y_prob):
    """Threshold that maximizes F1."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f = 0.5, -1
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t
    return float(best_t)

def inner_threshold_search(pipeline, X_train, y_train, method="youden"):
    """
    Leak-free threshold selection:
    - Train set üzerinde 3-fold CV ile out-of-fold prob üret
    - Bu prob üzerinden threshold seç
    """
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    oof_prob = np.zeros(len(y_train), dtype=float)

    for tr_idx, va_idx in inner_cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        pipeline.fit(X_tr, y_tr)

        # proba
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            p = pipeline.predict_proba(X_va)[:, 1]
        else:
            # decision_function fallback
            scores = pipeline.decision_function(X_va)
            p = 1 / (1 + np.exp(-scores))
        oof_prob[va_idx] = p

    if method == "f1":
        return choose_threshold_f1(y_train, oof_prob)
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

def main():
    df, X, y = load_and_prepare()

    # Pipeline: scaling -> select -> model
    model = build_model(RANDOM_STATE)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=K_BEST)),
        ("model", model),
    ])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs, aucs, senss, specs, thrs = [], [], [], [], []

    print("\n--- 5-Fold CV (Fusion + Threshold Optimization) ---")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n--- Fold {fold} ---")

        X_train = X.iloc[train_idx].values
        y_train = y[train_idx]
        X_test  = X.iloc[test_idx].values
        y_test  = y[test_idx]

        # Threshold selection on train (leak-free)
        thr = inner_threshold_search(pipe, X_train, y_train, method="youden")
        thrs.append(thr)

        # Fit on full train
        pipe.fit(X_train, y_train)

        # Predict prob on test
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))

        acc, auc, sens, spec, cm = compute_metrics(y_test, y_prob, thr)

        print(f"Threshold   : {thr:.3f}  (selected on train)")
        print(f"Accuracy    : {acc:.3f}")
        print(f"ROC-AUC     : {auc:.3f}")
        print(f"Sensitivity : {sens:.3f}")
        print(f"Specificity : {spec:.3f}")
        print("Confusion Matrix:")
        print(cm)

        accs.append(acc); aucs.append(auc); senss.append(sens); specs.append(spec)

    print("\n=== MEAN PERFORMANCE ===")
    print(f"Mean Threshold  : {np.mean(thrs):.3f} ± {np.std(thrs):.3f}")
    print(f"Mean Accuracy   : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"Mean ROC-AUC    : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"Mean Sensitivity: {np.mean(senss):.3f}")
    print(f"Mean Specificity: {np.mean(specs):.3f}")

    # ---- Interpretability: selected features + importance ----
    # Fit once on full data to inspect selected features
    pipe.fit(X.values, y)

    selector = pipe.named_steps["select"]
    mask = selector.get_support()
    selected_feature_names = X.columns[mask].tolist()

    print("\nTop selected features (fusion):")
    for i, name in enumerate(selected_feature_names[:20], start=1):
        print(f"{i}. {name}")

    # Feature importances if available
    model_step = pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        imps = model_step.feature_importances_
        imp_series = pd.Series(imps, index=selected_feature_names).sort_values(ascending=False)
        print("\n=== TOP 20 IMPORTANT FEATURES (ON FULL FIT) ===")
        print(imp_series.head(20))
    else:
        print("\nModel does not expose feature_importances_. (This is OK.)")

if __name__ == "__main__":
    main()
