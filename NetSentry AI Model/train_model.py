"""
NetSentry – train_model.py
Trains an XGBoost binary classifier on cleaned_training_data.csv.
Usage: python train_model.py
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_CSV   = "cleaned_training_data.csv"
MODEL_FILE  = "netsentry_model.pkl"
LABEL_COL   = "Label"
TEST_SIZE   = 0.20
SEED        = 42

XGB_PARAMS = {
    "n_estimators":      400,
    "max_depth":         7,
    "learning_rate":     0.1,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "logloss",
    "random_state":      SEED,
    "tree_method":       "hist",
    "n_jobs":            -1,
}
# ────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading {path} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"      {df.shape}")
    return df


def prepare(df: pd.DataFrame):
    print("[2/5] Preparing features …")
    col = LABEL_COL if LABEL_COL in df.columns else \
          next((c for c in df.columns if "label" in c.lower()), None)
    if col is None:
        raise ValueError("No Label column found.")

    # Binary: BENIGN → 0, everything else → 1
    y = (df[col].str.strip().str.upper() != "BENIGN").astype(int)
    X = df.drop(columns=[col])

    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"      Dropping non-numeric: {non_num}")
        X.drop(columns=non_num, inplace=True)

    # Class imbalance weight
    n_benign    = int((y == 0).sum())
    n_malicious = int((y == 1).sum())
    scale_weight = n_benign / n_malicious if n_malicious else 1.0

    print(f"      Features: {X.shape[1]}  |  Benign: {n_benign:,}  |  Malicious: {n_malicious:,}")
    print(f"      scale_pos_weight = {scale_weight:.2f}")
    return X, y, scale_weight, list(X.columns)


def train(X_train, y_train, scale_weight: float) -> XGBClassifier:
    print("[3/5] Training …")
    params = {**XGB_PARAMS, "scale_pos_weight": scale_weight}

    # Try GPU; fall back to CPU silently
    try:
        import xgboost as xgb
        probe = xgb.DMatrix(np.zeros((4, 2)), label=np.zeros(4))
        xgb.train({"device": "cuda", "objective": "binary:logistic"},
                  probe, num_boost_round=1)
        params["device"] = "cuda"
        print("      GPU detected ✅")
    except Exception:
        params["device"] = "cpu"
        print("      CPU mode")

    model = XGBClassifier(**params)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    return model


def evaluate(model, X_test, y_test):
    print("[4/5] Evaluating …")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  F1       : {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))


def save_model(model, feature_names: list):
    print(f"[5/5] Saving → {MODEL_FILE}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    size_mb = os.path.getsize(MODEL_FILE) / 1_048_576
    print(f"      Saved ({size_mb:.1f} MB)")


def main():
    df = load_data(INPUT_CSV)
    X, y, scale_weight, feat_names = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    model = train(X_train, y_train, scale_weight)
    evaluate(model, X_test, y_test)
    save_model(model, feat_names)
    print("✅ Done.")


if __name__ == "__main__":
    main()
