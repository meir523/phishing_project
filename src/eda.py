# src/eda.py

"""
eda.py - Expanded EDA utilities for the Phishing Detection project.

Design goals:
- Train-only EDA (avoid leakage).
- Minimal, readable outputs (no large redundant tables).
- Action-oriented diagnostics: constants, near-constants, identical columns, outliers, strong predictors.
"""

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score


# ============================================================
# Public API
# ============================================================

def run_expanded_eda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    corr_method: str = "spearman",
    corr_min_abs_for_heatmap: float = 0.90,
    dist_plot_n_features: int = 4,
    boxplot_n_features: int = 2,
    suspicious_auc_threshold: float = 0.98,
    outlier_ratio_threshold: float = 0.10,
    outlier_z_thresh: float = 3.5,
    near_constant_threshold: float = 0.99,
    minus1_ratio_threshold: float = 0.01,
    strong_corr_threshold: float = 0.70,
    max_list_features: int = 15,
    identical_examples_k: int = 8,
):
    """
    Runs an expanded EDA report on the training set only.
    Prints key findings, produces a few plots, and returns small tables as a dict.
    """
    tables = {}

    _print_header("Expanded EDA")

    # -----------------------------
    # [1] Label balance
    # -----------------------------
    lb = _label_balance(y_train)
    tables["label_balance"] = lb
    _print_section("[1] Label balance (train)")
    print(lb)

    # -----------------------------
    # [2] Feature types & binary detection
    # -----------------------------
    dtype_counts, binary_cols = _feature_type_summary(X_train)
    tables["feature_types"] = pd.DataFrame([{
        "n_features": int(X_train.shape[1]),
        "n_binary": int(len(binary_cols)),
        "dtype_counts": str(dtype_counts),
    }])

    _print_section("[2] Feature types summary")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Dtype counts: {dtype_counts}")
    print(f"Binary features: {len(binary_cols)}")

    # -----------------------------
    # [3] Missing / infinite values
    # -----------------------------
    miss_inf = _missing_and_infinite_report(X_train)
    tables["missing_infinite"] = pd.DataFrame([miss_inf])
    _print_section("[3] Missing & infinite values")
    print(miss_inf)

    # -----------------------------
    # [4] Sentinel value check (-1 only)
    # -----------------------------
    minus1_flagged = _sentinel_minus1_report(X_train, ratio_threshold=minus1_ratio_threshold)
    tables["minus1_flagged"] = minus1_flagged

    _print_section(f"[4] Sentinel -1 check (ratio >= {minus1_ratio_threshold})")
    if len(minus1_flagged) == 0:
        print("No features with frequent -1 values were found.")
    else:
        print(minus1_flagged.head(max_list_features))

    # -----------------------------
    # [5] Constant + near-constant features (dominant value analysis)
    # -----------------------------
    const_df, near_df = _constant_and_near_constant_features(X_train, near_threshold=near_constant_threshold)
    tables["constant_features"] = const_df
    tables["near_constant_features"] = near_df

    _print_section("[5] Constant / near-constant features")
    print(f"Constant features (top_value_ratio == 1.0): {len(const_df)}")
    if len(const_df) > 0:
        print(const_df.head(max_list_features))

    print(f"\nNear-constant features (top_value_ratio >= {near_constant_threshold} and < 1.0): {len(near_df)}")
    if len(near_df) > 0:
        print(near_df.head(max_list_features))

    # For downstream computations (corr / heatmap), remove constant features to avoid warnings/noise
    X_nonconst = X_train.drop(columns=const_df["feature"].tolist(), errors="ignore")

    # -----------------------------
    # [6] Strong predictors: correlation with label (top + threshold)
    # -----------------------------
    corr_top = _corr_with_label(X_nonconst, y_train, method=corr_method, top_n=20)
    tables["corr_with_label_top20"] = corr_top

    strong_corr = corr_top[corr_top["abs_corr"] >= strong_corr_threshold].copy()
    tables["strong_corr_top20_filtered"] = strong_corr

    _print_section(f"[6] Top correlations with label ({corr_method}, top 20)")
    print(corr_top)

    # -----------------------------
    # [7] Univariate AUC (only for leakage screening, do not over-emphasize)
    # -----------------------------
    df_auc = _univariate_auc_per_feature(X_nonconst, y_train)
    suspicious = df_auc[df_auc["auc_abs"] >= suspicious_auc_threshold].copy()
    tables["suspicious_features_by_auc"] = suspicious

    _print_section(f"[7] Suspicious single-feature AUC (auc_abs >= {suspicious_auc_threshold})")
    if len(suspicious) == 0:
        print("None found.")
    else:
        print(suspicious.head(max_list_features))

    # -----------------------------
    # [8] Outlier ratios (robust MAD-z) for non-binary features
    # -----------------------------
    out_df = _outlier_ratios_table(X_nonconst, binary_cols=binary_cols, z_thresh=outlier_z_thresh)
    out_flag = out_df[out_df["outlier_ratio"] >= outlier_ratio_threshold].copy()
    tables["outlier_ratio_flagged"] = out_flag

    _print_section(f"[8] Features with high outlier ratio (>= {outlier_ratio_threshold})")
    if len(out_flag) == 0:
        print("None found.")
    else:
        print(out_flag.head(max_list_features))

    # -----------------------------
    # [9] Identical columns detection (perfect duplicates)
    # -----------------------------
    dup_pairs = _find_identical_columns(X_train)
    tables["identical_columns_pairs"] = dup_pairs

    _print_section("[9] Identical columns (perfect duplicates)")
    print(f"Identical duplicate pairs found: {len(dup_pairs)}")
    if len(dup_pairs) > 0:
        print(dup_pairs.head(identical_examples_k))

    # -----------------------------
    # [10] Feature groups summary (actionable): how many strong label-correlated features per group
    # -----------------------------
    groups = get_feature_groups()
    group_summary = _group_strong_corr_summary(
        groups=groups,
        X_full=X_train,
        X_for_corr=X_nonconst,
        y=y_train,
        method=corr_method,
        strong_corr_threshold=strong_corr_threshold,
        top_k_names=3
    )

    tables["group_summary"] = group_summary

    _print_section(f"[10] Group summary: strong |corr| >= {strong_corr_threshold}")
    print(group_summary)

    # ============================================================
    # Plots (small & focused)
    # ============================================================

    # Choose features for distributions: strongest abs corr with label (non-binary, non-constant)
    dist_features = _select_distribution_features(
        X=X_nonconst,
        y=y_train,
        binary_cols=binary_cols,
        method=corr_method,
        n=dist_plot_n_features
    )

    _print_section("[11] Distribution plots by label (selected by strongest abs corr)")
    if len(dist_features) == 0:
        print("Skipped (no suitable non-binary features found).")
    else:
        _plot_distributions_by_label(X_nonconst, y_train, dist_features, bins=50, max_cols=2)

    # Boxplots for outlier-heavy features
    _print_section("[12] Boxplots by label (outlier-heavy features)")
    box_candidates = out_df["feature"].head(boxplot_n_features).tolist() if len(out_df) > 0 else []
    if len(box_candidates) == 0:
        print("Skipped (no non-binary features available).")
    else:
        _plot_boxplots_by_label(X_nonconst, y_train, box_candidates, max_cols=2)

    # Heatmap: strongly correlated subset of features (no pairs table)
    _print_section("[13] Correlation heatmap (filtered subset)")
    heatmap_features = _select_heatmap_features(
        X=X_nonconst,
        method=corr_method,
        min_abs_corr=corr_min_abs_for_heatmap,
        max_features=30
    )
    if len(heatmap_features) < 4:
        print("Skipped (not enough strongly-correlated features).")
    else:
        _plot_corr_heatmap(X_nonconst, heatmap_features, method=corr_method)

    return tables


# ============================================================
# Helpers: printing
# ============================================================

def _print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _print_section(title: str):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# Helpers: basic stats
# ============================================================

def _label_balance(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False)
    ratios = y.value_counts(normalize=True, dropna=False)
    return pd.DataFrame({"count": counts, "ratio": ratios})


def _feature_type_summary(X: pd.DataFrame):
    dtype_counts = X.dtypes.value_counts().to_dict()

    binary_cols = []
    for c in X.columns:
        vals = X[c].dropna().unique()
        if len(vals) <= 2 and set(vals).issubset({0, 1}):
            binary_cols.append(c)

    return dtype_counts, set(binary_cols)


def _missing_and_infinite_report(X: pd.DataFrame) -> dict:
    na_total = int(X.isna().sum().sum())
    arr = X.to_numpy(dtype=float, copy=False)
    inf_total = int(np.isinf(arr).sum())
    return {"missing_total": na_total, "infinite_total": inf_total}


# ============================================================
# Helpers: sentinel / dominant value analysis
# ============================================================

def _sentinel_minus1_report(X: pd.DataFrame, ratio_threshold: float = 0.01) -> pd.DataFrame:
    n = len(X)
    rows = []
    for c in X.columns:
        cnt = int((X[c] == -1).sum())
        ratio = float(cnt / n)
        if ratio >= ratio_threshold:
            rows.append({"feature": c, "count_eq_-1": cnt, "ratio_eq_-1": ratio})
    return pd.DataFrame(rows).sort_values("ratio_eq_-1", ascending=False).reset_index(drop=True)


def _constant_and_near_constant_features(X: pd.DataFrame, near_threshold: float = 0.99):
    const_rows = []
    near_rows = []
    for c in X.columns:
        vc = X[c].value_counts(normalize=True, dropna=False)
        top_ratio = float(vc.iloc[0]) if len(vc) > 0 else 0.0

        if top_ratio == 1.0:
            const_rows.append({"feature": c, "top_value_ratio": top_ratio})
        elif top_ratio >= near_threshold:
            near_rows.append({"feature": c, "top_value_ratio": top_ratio})

    const_df = pd.DataFrame(const_rows).sort_values("top_value_ratio", ascending=False).reset_index(drop=True)
    near_df = pd.DataFrame(near_rows).sort_values("top_value_ratio", ascending=False).reset_index(drop=True)
    return const_df, near_df


# ============================================================
# Helpers: label relationship (corr, AUC)
# ============================================================

def _corr_with_label(X: pd.DataFrame, y: pd.Series, method="spearman", top_n: int = 20) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        try:
            r = X[c].corr(y, method=method)
        except Exception:
            r = np.nan
        rows.append({"feature": c, "corr": r, "abs_corr": np.nan if pd.isna(r) else abs(r)})

    df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).head(top_n).reset_index(drop=True)
    return df


def _univariate_auc_per_feature(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        x = X[c].values

        if np.nanmin(x) == np.nanmax(x):
            rows.append({"feature": c, "auc": np.nan, "auc_abs": np.nan})
            continue

        try:
            auc = roc_auc_score(y, x)
            auc_abs = max(auc, 1.0 - auc)
            rows.append({"feature": c, "auc": float(auc), "auc_abs": float(auc_abs)})
        except Exception:
            rows.append({"feature": c, "auc": np.nan, "auc_abs": np.nan})

    df_auc = pd.DataFrame(rows).sort_values("auc_abs", ascending=False).reset_index(drop=True)
    return df_auc


def _select_distribution_features(X, y, binary_cols, method="spearman", n=4):
    # Select non-binary features with the strongest abs correlation to label
    rows = []
    for c in X.columns:
        if c in binary_cols:
            continue
        try:
            r = X[c].corr(y, method=method)
        except Exception:
            r = np.nan
        if pd.isna(r):
            continue
        rows.append((c, abs(r)))

    rows.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in rows[:n]]


# ============================================================
# Helpers: outliers
# ============================================================

def _robust_outlier_ratio_mad(x, z_thresh=3.5) -> float:
    # Robust outlier ratio using MAD-based z-score
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return 0.0
    z = 0.6745 * (x - med) / mad
    return float(np.mean(np.abs(z) > z_thresh))


def _outlier_ratios_table(X: pd.DataFrame, binary_cols: set, z_thresh: float = 3.5) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        if c in binary_cols:
            continue
        ratio = _robust_outlier_ratio_mad(X[c].values, z_thresh=z_thresh)
        rows.append({"feature": c, "outlier_ratio": ratio})
    return pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False).reset_index(drop=True)


# ============================================================
# Helpers: identical columns (perfect duplicates)
# ============================================================

def _find_identical_columns(X: pd.DataFrame) -> pd.DataFrame:
    # Use hashing to find candidates, then confirm equality (fast enough for ~111 columns)
    hashes = {}
    for c in X.columns:
        h = pd.util.hash_pandas_object(X[c], index=False).sum()
        hashes.setdefault(h, []).append(c)

    pairs = []
    for _, cols in hashes.items():
        if len(cols) < 2:
            continue
        base = cols[0]
        for other in cols[1:]:
            if X[base].equals(X[other]):
                pairs.append({"feature_1": base, "feature_2": other})
            else:
                # Hash collision: ignore unless equal
                pass

    return pd.DataFrame(pairs)


# ============================================================
# Helpers: feature groups summary
# ============================================================

def get_feature_groups() -> dict:
    """
    Returns the exact feature grouping as defined in the paper (Tables 1-6).
    Total features: 111 (excluding target).
    """
    groups = {
        # Table 1: URL properties (20 features)
        "url": [
            "qty_dot_url", "qty_hyphen_url", "qty_underline_url", "qty_slash_url",
            "qty_questionmark_url", "qty_equal_url", "qty_at_url", "qty_and_url",
            "qty_exclamation_url", "qty_space_url", "qty_tilde_url", "qty_comma_url",
            "qty_plus_url", "qty_asterisk_url", "qty_hashtag_url", "qty_dollar_url",
            "qty_percent_url", "qty_tld_url", "length_url", "email_in_url"
        ],
        
        # Table 2: Domain properties (21 features)
        "domain": [
            "qty_dot_domain", "qty_hyphen_domain", "qty_underline_domain", "qty_slash_domain",
            "qty_questionmark_domain", "qty_equal_domain", "qty_at_domain", "qty_and_domain",
            "qty_exclamation_domain", "qty_space_domain", "qty_tilde_domain", "qty_comma_domain",
            "qty_plus_domain", "qty_asterisk_domain", "qty_hashtag_domain", "qty_dollar_domain",
            "qty_percent_domain", "qty_vowels_domain", "domain_length", "domain_in_ip",
            "server_client_domain"
        ],
        
        # Table 3: Directory properties (18 features)
        "directory": [
            "qty_dot_directory", "qty_hyphen_directory", "qty_underline_directory", "qty_slash_directory",
            "qty_questionmark_directory", "qty_equal_directory", "qty_at_directory", "qty_and_directory",
            "qty_exclamation_directory", "qty_space_directory", "qty_tilde_directory", "qty_comma_directory",
            "qty_plus_directory", "qty_asterisk_directory", "qty_hashtag_directory", "qty_dollar_directory",
            "qty_percent_directory", "directory_length"
        ],
        
        # Table 4: File properties (18 features)
        "file": [
            "qty_dot_file", "qty_hyphen_file", "qty_underline_file", "qty_slash_file",
            "qty_questionmark_file", "qty_equal_file", "qty_at_file", "qty_and_file",
            "qty_exclamation_file", "qty_space_file", "qty_tilde_file", "qty_comma_file",
            "qty_plus_file", "qty_asterisk_file", "qty_hashtag_file", "qty_dollar_file",
            "qty_percent_file", "file_length"
        ],
        
        # Table 5: Parameters properties (20 features)
        "params": [
            "qty_dot_params", "qty_hyphen_params", "qty_underline_params", "qty_slash_params",
            "qty_questionmark_params", "qty_equal_params", "qty_at_params", "qty_and_params",
            "qty_exclamation_params", "qty_space_params", "qty_tilde_params", "qty_comma_params",
            "qty_plus_params", "qty_asterisk_params", "qty_hashtag_params", "qty_dollar_params",
            "qty_percent_params", "params_length", "tld_present_params", "qty_params"
        ],

        # Table 6: External services (14 features)
        # Note: These are the ones that confuse rule-based logic
        "external": [
            "time_response", "domain_spf", "asn_ip", "time_domain_activation",
            "time_domain_expiration", "qty_ip_resolved", "qty_nameservers", "qty_mx_servers",
            "ttl_hostname", "tls_ssl_certificate", "qty_redirects", "url_google_index",
            "domain_google_index", "url_shortened"
        ]
    }
    
    return groups

def get_feature_list(
    X,
    *,
    include_groups=None,
    exclude_groups=None
):
    """
    Returns a list of feature names based on the paper-defined groups.

    Args:
        X (pd.DataFrame): Feature dataframe (columns are features).
        include_groups (list|None): If provided, keep only these groups (e.g. ["url","domain"]).
        exclude_groups (list|None): If provided, drop these groups (e.g. ["external"]).

    Notes:
        - Safely intersects with X.columns (ignores missing).
        - If both include_groups and exclude_groups are None -> returns all features in X.
    """
    groups = get_feature_groups()

    all_group_names = set(groups.keys())

    if include_groups is not None:
        include_set = set(include_groups)
        unknown = include_set - all_group_names
        if len(unknown) > 0:
            raise ValueError(f"Unknown include_groups: {sorted(list(unknown))}")
        chosen_groups = include_set
    else:
        chosen_groups = set(groups.keys())

    if exclude_groups is not None:
        exclude_set = set(exclude_groups)
        unknown = exclude_set - all_group_names
        if len(unknown) > 0:
            raise ValueError(f"Unknown exclude_groups: {sorted(list(unknown))}")
        chosen_groups = chosen_groups - exclude_set

    feats = []
    for g in chosen_groups:
        feats.extend(groups[g])

    # Keep only columns that exist in X
    cols = [c for c in feats if c in X.columns]
    return cols


def select_features(X, feature_list):
    """
    Returns X restricted to the given feature list (safe intersection).
    """
    cols = [c for c in feature_list if c in X.columns]
    return X[cols].copy()


def _group_strong_corr_summary(
    groups: dict,
    X_full: pd.DataFrame,
    X_for_corr: pd.DataFrame,
    y: pd.Series,
    *,
    method: str = "spearman",
    strong_corr_threshold: float = 0.70,
    top_k_names: int = 3
) -> pd.DataFrame:
    """
    Group summary that reports TOTAL group sizes (from X_full),
    while computing correlations only on X_for_corr (e.g., non-constant features).

    Returns columns:
    - n_features_total: total features defined in the group
    - n_used_for_corr: features available in X_for_corr (non-constant subset)
    - n_constant_removed: total - used_for_corr
    - n_strong_corr: count of features with |corr| >= threshold among used_for_corr
    - top_features_by_abs_corr: top-k feature names by |corr| (among used_for_corr)
    """
    rows = []

    full_cols_set = set(X_full.columns)
    corr_cols_set = set(X_for_corr.columns)

    for g, cols in groups.items():
        # Keep only features that truly exist in the dataset
        cols_full = [c for c in cols if c in full_cols_set]
        if len(cols_full) == 0:
            continue

        # Only these can be used for correlation (non-constant)
        cols_corr = [c for c in cols_full if c in corr_cols_set]

        corr_list = []
        for c in cols_corr:
            try:
                r = X_for_corr[c].corr(y, method=method)
            except Exception:
                r = np.nan
            if pd.isna(r):
                continue
            corr_list.append((c, r, abs(r)))

        corr_list.sort(key=lambda t: t[2], reverse=True)
        strong = [t for t in corr_list if t[2] >= strong_corr_threshold]
        top_names = [t[0] for t in corr_list[:top_k_names]]

        rows.append({
            "group": g,
            "n_features_total": len(cols_full),
            #"n_used_for_corr": len(cols_corr),  # Optional
            #"n_constant_removed": len(cols_full) - len(cols_corr), # Optional
            "n_strong_corr": len(strong),
            "top_features_by_abs_corr": ", ".join(top_names)
        })

    return pd.DataFrame(rows).sort_values("n_features_total", ascending=False).reset_index(drop=True)


# ============================================================
# Helpers: heatmap feature selection
# ============================================================

def _select_heatmap_features(X, method="spearman", min_abs_corr=0.90, max_features=30):
    corr = X.corr(method=method)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_upper = corr.where(mask)

    pairs = corr_upper.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()

    pairs = pairs[pairs["abs_corr"] >= min_abs_corr]
    if len(pairs) == 0:
        return []

    feats = sorted(set(pairs["feature_1"]).union(set(pairs["feature_2"])))
    return feats[:max_features]


# ============================================================
# Helpers: plots
# ============================================================

def _plot_distributions_by_label(X, y, features, bins=50, max_cols=2):
    n = len(features)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, f in enumerate(features):
        ax = axes[i]
        for label in sorted(y.unique()):
            sns.histplot(
                X.loc[y == label, f],
                bins=bins,
                stat="density",
                element="step",
                fill=False,
                ax=ax,
                label=f"class={label}"
            )
        ax.set_title(f"Distribution by label: {f}")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def _plot_boxplots_by_label(X, y, features, max_cols=2):
    n = len(features)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    df_plot = X[features].copy()
    df_plot["target"] = y.values

    for i, f in enumerate(features):
        ax = axes[i]
        sns.boxplot(data=df_plot, x="target", y=f, ax=ax)
        ax.set_title(f"Boxplot by label: {f}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def _plot_corr_heatmap(X, features, method="spearman"):
    corr = X[features].corr(method=method)
    plt.figure(figsize=(0.6 * len(features) + 6, 0.6 * len(features) + 4))
    sns.heatmap(corr, annot=False, square=True)
    plt.title(f"{method.title()} correlation heatmap (filtered subset)")
    plt.tight_layout()
    plt.show()
