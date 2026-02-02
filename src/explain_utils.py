"""
explain_utils.py

Utilities for Model Explainability (XAI) using XGBoost built-ins and SHAP.
Includes automated explanations for plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from typing import List, Optional, Tuple, Any
# Local module for group definitions
import eda 

def _print_header(title, what, how, example=None):
    """Internal helper to print consistent headers."""
    print("\n" + "="*60)
    print(f"PLOT: {title}")
    print("-" * 60)
    print(f"WHAT IT SHOWS: {what}")
    print(f"HOW TO READ:   {how}")
    if example:
        print(f"EXAMPLE:       {example}")
    print("="*60)


def plot_xgb_importance(model, max_num_features=20, importance_type='gain', title="XGBoost Feature Importance", print_expl=True):
    """
    Plots the built-in feature importance from XGBoost.
    """
    if print_expl:
        _print_header(
            title=f"Native Feature Importance ({importance_type})",
            what="The most 'useful' features for the model during training.",
            how="Longer bars = Higher contribution to reducing prediction error (loss).",
            example="Does NOT show direction (good vs. bad), only magnitude."
        )

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=max_num_features, importance_type=importance_type, height=0.6)
    plt.title(f"{title} ({importance_type})")
    plt.show()


def get_shap_explainer(model):
    """Returns a TreeExplainer for XGBoost/RF."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X: pd.DataFrame):
    """Computes SHAP values."""
    # Check if SHAP version returns an Explanation object or list/array
    shap_values = explainer(X)
    return shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, plot_type="dot", max_display=20, print_expl=True):
    """
    Global explainability: Beeswarm (dot) or Bar plot.
    """
    if print_expl and plot_type == "dot":
        _print_header(
            title="Global SHAP Summary (Beeswarm)",
            what="Feature importance + Direction of impact + Data distribution.",
            how="Y-Axis=Importance, X-Axis=Impact (Right=Phishing, Left=Legit), Color=Value.",
            example="Red dots on the RIGHT mean High Values increase Phishing risk."
        )
    elif print_expl and plot_type == "bar":
        _print_header(
            title="Global SHAP Importance (Bar)",
            what="Mean absolute impact of each feature.",
            how="Longer bars = More important feature overall."
        )

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type=plot_type, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()


def plot_local_waterfall(shap_values, index: int, max_display=10, print_expl=True):
    """
    Local explainability: Waterfall plot for a single instance.
    """
    if print_expl:
        print(f"\n--- Analyzing Instance Index: {index} ---")
        print("READING GUIDE (Bottom to Top):")
        print("  1. Start at E[f(x)]: Average dataset score.")
        print("  2. RED arrows:  Features pushing score HIGHER (Risk).")
        print("  3. BLUE arrows: Features pushing score LOWER (Safe).")
        print("  4. f(x):        Final prediction score.")

    plt.figure()
    # shap_values[index] is the Explanation object for that specific row
    shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()


def plot_feature_group_importance(shap_values, X_columns, title="Feature Importance by Group", print_expl=True):
    """
    Aggregates SHAP values by feature groups (defined in eda.py) 
    and plots the total importance per group.
    """
    if print_expl:
        _print_header(
            title="Feature Importance by Semantic Group",
            what="Aggregated impact of feature categories (URL, Domain, External, etc.).",
            how="Longer bars = The entire category has a stronger total impact."
        )

    # 1. Calculate mean absolute SHAP value for each feature
    if isinstance(shap_values, np.ndarray):
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create mapping
    feat_importance = dict(zip(X_columns, mean_abs_shap))
    
    # 2. Get groups
    groups = eda.get_feature_groups()
    
    # 3. Aggregate
    group_importance = {}
    for group_name, features in groups.items():
        total_imp = 0
        for f in features:
            if f in feat_importance:
                total_imp += feat_importance[f]
        group_importance[group_name] = total_imp

    # 4. Convert to DataFrame
    df_groups = pd.DataFrame(list(group_importance.items()), columns=['Group', 'Total Magnitude'])
    df_groups = df_groups.sort_values('Total Magnitude', ascending=True)

    # 5. Plot
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df_groups))
    bars = plt.barh(df_groups['Group'], df_groups['Total Magnitude'], color=colors)
    
    # Add labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (width*0.01), bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center', fontsize=10)

    plt.title(title, fontsize=14)
    plt.xlabel("Sum of Mean |SHAP| Values")
    plt.ylabel("Feature Group")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def find_interesting_indices(y_true, y_pred) -> dict:
    """Returns TP, TN, FP, FN indices."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return {
        "TP": np.where((y_true == 1) & (y_pred == 1))[0],
        "TN": np.where((y_true == 0) & (y_pred == 0))[0],
        "FP": np.where((y_true == 0) & (y_pred == 1))[0],
        "FN": np.where((y_true == 1) & (y_pred == 0))[0]
    }