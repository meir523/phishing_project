# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 10:56:25 2026

@author: meir5
"""

# src/preprocess.py

"""
preprocess.py - Train-fitted preprocessing for phishing dataset.

Responsibilities:
1) Drop constant features (fit on train only).
2) Drop perfectly identical duplicate columns (fit on train only).
3) Encode -1 sentinel in params-related features:
   - Add has_params indicator
   - Replace -1 -> 0 in params features

Notes:
- No scaling here (done later inside model pipelines if needed).
- All decisions are fitted on train and applied to both train/test consistently.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Utility functions
# =========================

def find_constant_columns(X: pd.DataFrame) -> List[str]:
    """Return columns that are constant (same value for all rows)."""
    const_cols = []
    for c in X.columns:
        # dropna=False so a column of all NaN would be treated as constant as well (not expected here)
        if X[c].nunique(dropna=False) <= 1:
            const_cols.append(c)
    return const_cols


def find_identical_duplicate_columns(X: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Return columns that are exact duplicates of other columns.
    Uses hashing to find candidates, then verifies equality.
    Output:
      - dup_cols_to_drop: list of columns to drop (keep the first occurrence)
      - dup_pairs_df: DataFrame with pairs (kept, dropped)
    """
    # Hash each column to reduce comparisons
    hash_buckets: Dict[int, List[str]] = {}
    for c in X.columns:
        h = int(pd.util.hash_pandas_object(X[c], index=False).sum())
        hash_buckets.setdefault(h, []).append(c)

    dup_cols_to_drop = []
    pairs = []

    for _, cols in hash_buckets.items():
        if len(cols) < 2:
            continue
        keep = cols[0]
        for other in cols[1:]:
            # Hash collisions are possible; confirm equality
            if X[keep].equals(X[other]):
                dup_cols_to_drop.append(other)
                pairs.append({"kept": keep, "dropped": other})

    dup_pairs_df = pd.DataFrame(pairs)
    return sorted(set(dup_cols_to_drop)), dup_pairs_df


def default_params_columns(columns: List[str]) -> List[str]:
    """
    Params group columns based on the dataset description.
    Covers:
      - all *_params columns
      - plus known params-related fields without the suffix
    """
    extra = {"qty_params", "params_length", "tld_present_params"}
    params_cols = [c for c in columns if c.endswith("_params") or c in extra]
    return sorted(set(params_cols))


def add_has_params_indicator(df: pd.DataFrame, params_cols: List[str], new_col: str = "has_params") -> pd.Series:
    """
    Build a robust indicator: has_params = 1 if NOT all params columns are -1 for that row.
    If params_cols is empty, returns a column of zeros.
    """
    if len(params_cols) == 0:
        return pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index, name=new_col)

    # If params are absent, these features are typically all -1 (based on EDA).
    all_minus1 = df[params_cols].eq(-1).all(axis=1)
    has_params = (~all_minus1).astype(np.int8)
    has_params.name = new_col
    return has_params


def replace_minus1_with_zero(df: pd.DataFrame, cols: List[str]) -> None:
    """In-place replacement: -1 -> 0 for the specified columns."""
    if len(cols) == 0:
        return
    df[cols] = df[cols].replace(-1, 0)
    
def _minus1_ratio(s: pd.Series) -> float:
    """Return fraction of rows equal to -1."""
    return float((s == -1).mean())


def _is_group_all_params(cols: List[str]) -> bool:
    extras = {"qty_params", "params_length", "tld_present_params"}
    return all(c.endswith("_params") or c in extras for c in cols)


def _is_group_all_directory(cols: List[str]) -> bool:
    return all(c.endswith("_directory") or c == "directory_length" for c in cols)


def _is_group_all_file(cols: List[str]) -> bool:
    return all(c.endswith("_file") or c == "file_length" for c in cols)


def find_minus1_mask_groups(
    X: pd.DataFrame,
    *,
    min_ratio: float = 0.01
) -> Dict[str, List[str]]:
    """
    Automatically group columns by identical (-1) masks.
    Returns mapping: indicator_name -> list of columns to encode.
    """
    cols = []
    for c in X.columns:
        if (X[c] == -1).any() and _minus1_ratio(X[c]) >= min_ratio:
            cols.append(c)

    if len(cols) == 0:
        return {}

    # Bucket by hash of the -1 mask, then verify equality to avoid collisions
    buckets: Dict[int, List[str]] = {}
    for c in cols:
        m = X[c].eq(-1)
        h = int(pd.util.hash_pandas_object(m, index=False).sum())
        buckets.setdefault(h, []).append(c)

    groups: List[List[str]] = []
    for _, bucket_cols in buckets.items():
        subgroups: List[List[str]] = []
        for c in bucket_cols:
            placed = False
            m_c = X[c].eq(-1)
            for g in subgroups:
                if X[g[0]].eq(-1).equals(m_c):
                    g.append(c)
                    placed = True
                    break
            if not placed:
                subgroups.append([c])
        groups.extend(subgroups)

    # Name groups with simple heuristics
    mapping: Dict[str, List[str]] = {}
    used_names = set()

    def _unique(name: str) -> str:
        if name not in used_names:
            used_names.add(name)
            return name
        k = 2
        while f"{name}_{k}" in used_names:
            k += 1
        used_names.add(f"{name}_{k}")
        return f"{name}_{k}"
    
    def _is_group_dir_or_file_mix(cols: List[str]) -> bool:
        allowed = {"directory_length", "file_length"}
        return all(c.endswith("_directory") or c.endswith("_file") or c in allowed for c in cols)


    ext_k = 1
    for cols_g in groups:
        cols_g_sorted = sorted(cols_g)

        if _is_group_all_params(cols_g_sorted):
            name = _unique("has_params")
        elif _is_group_all_directory(cols_g_sorted):
            name = _unique("has_directory")
        elif _is_group_all_file(cols_g_sorted):
            name = _unique("has_file")
        elif _is_group_dir_or_file_mix(cols_g_sorted):
            # In this dataset, directory+file often share the same -1 mask
            name = _unique("has_path")
        elif len(cols_g_sorted) == 1:
            # Single-feature availability indicator
            name = _unique(f"has_{cols_g_sorted[0]}")
        else:
            name = _unique(f"has_external_{ext_k}")
            ext_k += 1
        mapping[name] = cols_g_sorted

    return mapping



# =========================
# Preprocessor
# =========================

@dataclass
class PreprocessReport:
    constant_cols: List[str]
    duplicate_cols: List[str]
    duplicate_pairs: pd.DataFrame
    params_cols_encoded: List[str]
    added_indicator_cols: List[str]


class Preprocessor:
    """
    Fit on train, then transform train/test consistently.

    minus1_mode:
      - "params": encode only params-related columns (safe default).
      - "auto"  : automatically group columns by identical (-1) masks and encode each group.
      - "none"  : do not touch -1 at all.
    """

    def __init__(
        self,
        *,
        minus1_mode: str = "params",  # "params" | "auto" | "none"
        minus1_min_ratio: float = 0.01,
        params_cols: Optional[List[str]] = None,
    ):
        self.minus1_mode = minus1_mode
        self.minus1_min_ratio = minus1_min_ratio
        self.params_cols_user = params_cols

        # fitted attributes
        self.constant_cols_: List[str] = []
        self.duplicate_cols_: List[str] = []
        self.duplicate_pairs_: pd.DataFrame = pd.DataFrame()

        # NEW: mapping indicator_name -> list of columns to encode (-1 -> 0)
        self.minus1_groups_: Dict[str, List[str]] = {}
        self.added_indicator_cols_: List[str] = []

        self._is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "Preprocessor":
        """
        Fit preprocessing decisions on training data only.
        """
        X = X_train.copy()

        # 1) Constant columns
        self.constant_cols_ = find_constant_columns(X)

        # Remove constants before duplicate search (cleaner and avoids overlap confusion)
        X_nc = X.drop(columns=self.constant_cols_, errors="ignore")

        # 2) Identical duplicate columns (exact equality)
        self.duplicate_cols_, self.duplicate_pairs_ = find_identical_duplicate_columns(X_nc)

        # Apply the planned drops to decide -1 groups on the same feature space
        drop_cols = sorted(set(self.constant_cols_) | set(self.duplicate_cols_))
        X_work = X.drop(columns=drop_cols, errors="ignore")

        # 3) Decide which -1 groups to encode
        if self.minus1_mode == "none":
            self.minus1_groups_ = {}

        elif self.minus1_mode == "params":
            # Safe, domain-driven choice
            cols_params = self.params_cols_user
            if cols_params is None:
                cols_params = default_params_columns(list(X_train.columns))

            cols_params = [c for c in cols_params if c in X_work.columns]
            self.minus1_groups_ = {"has_params": cols_params} if len(cols_params) > 0 else {}

        elif self.minus1_mode == "auto":
            # Data-driven: group by identical -1 masks, name groups with heuristics
            self.minus1_groups_ = find_minus1_mask_groups(X_work, min_ratio=self.minus1_min_ratio)

        else:
            raise ValueError("minus1_mode must be one of: 'params', 'auto', 'none'")

        self.added_indicator_cols_ = list(self.minus1_groups_.keys())

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessReport]:
        """
        Transform a dataset using fitted preprocessing decisions.
        Returns:
          - X_out (transformed DataFrame)
          - report (what was done)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")

        df = X.copy()

        # Drop constants + duplicates (safe even if overlap exists)
        drop_cols = sorted(set(self.constant_cols_) | set(self.duplicate_cols_))
        df = df.drop(columns=drop_cols, errors="ignore")

        # Encode -1 groups: add indicator + replace -1 -> 0
        for ind_name, cols in self.minus1_groups_.items():
            cols_present = [c for c in cols if c in df.columns]
            if len(cols_present) == 0:
                continue

            # Indicator is computed BEFORE replacement, based on the original -1 pattern
            all_minus1 = df[cols_present].eq(-1).all(axis=1)
            df[ind_name] = (~all_minus1).astype(np.int8)

            # Replace -1 -> 0 in those columns
            replace_minus1_with_zero(df, cols_present)

        report = PreprocessReport(
            constant_cols=self.constant_cols_,
            duplicate_cols=self.duplicate_cols_,
            duplicate_pairs=self.duplicate_pairs_,
            params_cols_encoded=sorted(set([c for cols in self.minus1_groups_.values() for c in cols])),
            added_indicator_cols=self.added_indicator_cols_,
        )

        return df, report

    def fit_transform(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, PreprocessReport]:
        """Convenience method: fit on train, then transform train."""
        self.fit(X_train, y_train)
        return self.transform(X_train)

    def summary(self) -> dict:
        """Small fitted summary for logging / notebook."""
        if not self._is_fitted:
            return {"status": "not_fitted"}
        return {
            "constant_cols": len(self.constant_cols_),
            "duplicate_cols": len(self.duplicate_cols_),
            "minus1_mode": self.minus1_mode,
            "minus1_groups": {k: len(v) for k, v in self.minus1_groups_.items()},
            "added_indicator_cols": self.added_indicator_cols_,
        }
