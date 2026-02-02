# -*- coding: utf-8 -*-
"""
tuning.py

Grid search utilities (transparent and notebook-friendly).
All comments in English.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold


def run_grid_search(
    *,
    model_name: str,
    estimator,
    param_grid: Dict[str, Any],
    X,
    y,
    scoring: Optional[Dict[str, str]] = None,
    refit: str = "pr_auc",
    n_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[GridSearchCV, pd.DataFrame]:
    """
    Runs GridSearchCV with StratifiedKFold and returns:
      - fitted GridSearchCV object
      - a compact DataFrame summary (sorted by refit metric)

    Notes:
      - Use n_jobs=1 for GPU-based XGBoost to avoid parallel GPU fits.
      - scoring default returns PR-AUC and ROC-AUC for reporting.
    """
    if scoring is None:
        scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=False
    )

    print(f"\n=== GridSearch: {model_name} ===")
    print(f"CV: StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={random_state})")
    print(f"Refit metric: {refit}")
    print(f"Total combinations: {len(gs.get_params()['param_grid']) if isinstance(param_grid, list) else _count_grid(param_grid)}")

    gs.fit(X, y)

    df = pd.DataFrame(gs.cv_results_)
    keep_cols = [
        "rank_test_pr_auc",
        "mean_test_pr_auc", "std_test_pr_auc",
        "mean_test_roc_auc", "std_test_roc_auc",
        "mean_fit_time", "mean_score_time",
        "params"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].sort_values(by=f"mean_test_{refit}", ascending=False).reset_index(drop=True)

    return gs, df


def _count_grid(param_grid: Dict[str, Any]) -> int:
    """Counts combinations in a sklearn param_grid dict."""
    total = 1
    for k, v in param_grid.items():
        total *= len(v)
    return total
