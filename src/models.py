# -*- coding: utf-8 -*-
"""
models.py

Model builders ("factory") for phishing detection.

Models:
- Logistic Regression (Elastic Net) with scaling
- Linear SVM (LinearSVC) with scaling + calibration (predict_proba)
- Random Forest
- XGBoost (GPU if available)

All comments in this file are in English by design.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import os
import json
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def build_logreg_elasticnet(*, random_state: int = 42, **logreg_kwargs) -> Pipeline:
    """Logistic Regression (ElasticNet) with scaling. kwargs override defaults."""
    params = dict(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state
    )
    params.update(logreg_kwargs)

    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params))
    ])



def build_linear_svm_calibrated(
    *,
    random_state: int = 42,
    svm_kwargs: Optional[Dict[str, Any]] = None,
    calib_kwargs: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Linear SVM with scaling + probability calibration.
    svm_kwargs/calib_kwargs override defaults.
    """
    svm_params = dict(
        C=1.0,
        class_weight="balanced",
        random_state=random_state
    )
    if svm_kwargs:
        svm_params.update(svm_kwargs)

    base = LinearSVC(**svm_params)

    cal_params = dict(
        method="sigmoid",
        cv=3,
        n_jobs=-1
    )
    if calib_kwargs:
        cal_params.update(calib_kwargs)

    calibrated = CalibratedClassifierCV(
        estimator=base,
        **cal_params
    )

    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", calibrated)
    ])


def build_random_forest(*, random_state: int = 42, **rf_kwargs) -> RandomForestClassifier:
    """Random Forest baseline. kwargs override defaults."""
    params = dict(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state
    )
    params.update(rf_kwargs)
    return RandomForestClassifier(**params)


def _compute_scale_pos_weight(y: Optional[pd.Series]) -> Optional[float]:
    """Compute scale_pos_weight = #neg / #pos for XGBoost (helps imbalance)."""
    if y is None:
        return None
    yv = y.astype(int).values
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    if n_pos == 0:
        return None
    return float(n_neg / n_pos)


def build_xgboost(
    y_train: Optional[pd.Series] = None,
    *,
    random_state: int = 42,
    use_gpu: bool = True,
    **xgb_kwargs
):
    """
    XGBoost baseline. kwargs override defaults.
    If scale_pos_weight not provided, it is computed from y_train.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Install it or skip XGBoost.")

    spw = _compute_scale_pos_weight(y_train)

    params: Dict[str, Any] = dict(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )

    # Only set scale_pos_weight if the user didn't override it
    if spw is not None and "scale_pos_weight" not in xgb_kwargs:
        params["scale_pos_weight"] = spw

    # Apply user overrides
    params.update(xgb_kwargs)

    if use_gpu:
        # Prefer modern GPU config if not explicitly overridden
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda")
        try:
            return XGBClassifier(**params)
        except TypeError:
            # Fallback for older versions
            params.pop("device", None)
            if params.get("tree_method") == "hist":
                params["tree_method"] = "gpu_hist"
            return XGBClassifier(**params)

    # CPU mode
    params.pop("device", None)
    params.setdefault("tree_method", "hist")
    return XGBClassifier(**params)

##### SAVE METHODS #####

def save_model(model, path: str, meta: dict | None = None):
    """Save sklearn-compatible model with optional metadata json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    if meta is not None:
        meta_path = path.replace(".joblib", ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

def make_meta(model_name: str, stage: str, extra: dict | None = None):
    """Create standard metadata dict."""
    meta = {
        "model_name": model_name,
        "stage": stage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    return meta
