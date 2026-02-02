# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 14:45:04 2026

@author: meir5
"""

# src/data_io.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str) -> pd.DataFrame:
    # Load dataset from CSV
    df = pd.read_csv(csv_path)
    return df


def basic_report(df: pd.DataFrame, target_col: str = "phishing") -> dict:
    # Compute minimal dataset report
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    report = {}
    report["shape"] = df.shape
    report["n_rows"] = int(df.shape[0])
    report["n_cols"] = int(df.shape[1])
    report["n_features"] = int(df.shape[1] - 1)

    y_counts = df[target_col].value_counts(dropna=False)
    y_ratio = df[target_col].value_counts(normalize=True, dropna=False)

    report["target_counts"] = y_counts.to_dict()
    report["target_ratio"] = y_ratio.round(6).to_dict()

    report["n_duplicates"] = int(df.duplicated().sum())
    report["n_missing_total"] = int(df.isna().sum().sum())

    return report


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Drop exact duplicate rows
    before = len(df)
    df2 = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df2)
    return df2, int(removed)


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    target_col: str = "phishing",
    random_state: int = 42
):
    """
    Stratified split into train/val/test using ratio-based logic:
    1) Combine val_ratio + test_ratio -> split train vs holdout
    2) Split holdout into val/test according to their relative proportions

    Example:
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        => train=70%, val=15%, test=15%
    """

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Step 1: Train vs Holdout (Val+Test)
    holdout_ratio = val_ratio + test_ratio

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Split Holdout into Val/Test according to relative ratios
    # Example: val=0.15, test=0.15 => val_share = 0.5
    val_share = val_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold,
        test_size=1 - val_share,   # because test_size refers to the *second* split
        random_state=random_state,
        stratify=y_hold
    )

    return X_train, X_val, X_test, y_train, y_val, y_test