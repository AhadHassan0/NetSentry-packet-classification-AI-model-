"""
NetSentry – preprocess_data.py
Loads ALL CICIDS2017 CSVs in the current folder, cleans them,
and saves a single training-ready CSV.
Usage: python preprocess_data.py
"""

import glob, sys
import numpy as np
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_CSV = "cleaned_training_data.csv"
MAX_ROWS   = 500_000

IDENTITY_COLS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp",
    " Flow ID", " Source IP", " Source Port",
    " Destination IP", " Destination Port", " Timestamp",
]
# ────────────────────────────────────────────────────────────────────────────


def load_csv() -> pd.DataFrame:
    files = glob.glob("*.csv")
    files = [f for f in files if f != OUTPUT_CSV]

    if not files:
        print("ERROR: No CSV files found in the current directory.")
        sys.exit(1)

    print(f"[1/4] Found {len(files)} CSV file(s):")
    for f in files:
        print(f"      * {f}")

    chunks = []
    for f in files:
        print(f"      Loading {f} ...")
        for chunk in pd.read_csv(f, chunksize=100_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"      Total rows: {len(df):,}  |  Columns: {len(df.columns)}")
    return df


def find_label_col(df: pd.DataFrame) -> str:
    if "Label" in df.columns:
        return "Label"
    candidates = [c for c in df.columns if "label" in c.lower()]
    if not candidates:
        raise ValueError(f"No Label column found. Columns: {list(df.columns)}")
    print(f"      Using '{candidates[0]}' as label column.")
    return candidates[0]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/4] Cleaning ...")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"      Dropped {before - len(df):,} NaN/Inf rows.")

    # Only drop identity cols — never drop Label here
    label_col = find_label_col(df)
    drops = [c for c in IDENTITY_COLS if c in df.columns and c != label_col]
    df.drop(columns=drops, inplace=True)
    print(f"      Removed identity columns: {drops}")
    return df


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    print(f"[3/4] Stratified sample -> {n:,} rows ...")
    label_col = find_label_col(df)
    print(f"      Label column: '{label_col}'")
    print(f"      Classes before sampling:\n{df[label_col].value_counts().to_string()}")

    n = min(n, len(df))
    frac = n / len(df)

    # Sample each class proportionally
    sampled_parts = []
    for label, group in df.groupby(label_col):
        sampled_parts.append(group.sample(frac=frac, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True).head(n)

    print(f"      Result: {len(sampled):,} rows")
    print(f"      Classes after sampling:\n{sampled[label_col].value_counts().to_string()}")
    return sampled


def main():
    df = load_csv()
    df = clean(df)
    df = stratified_sample(df, MAX_ROWS)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[4/4] Saved -> {OUTPUT_CSV}  (shape {df.shape})")
    print("Done.")


if __name__ == "__main__":
    main()