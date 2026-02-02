# -*- coding: utf-8 -*-
"""
eval_utils.py

Unified evaluation utilities for binary classification experiments.

Includes:
- score extraction (predict_proba / decision_function)
- evaluation at a given threshold
- threshold sweep
- run_multiple_models helper (train on train, eval on val)

All comments in this file are in English by design.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


@dataclass
class EvalResult:
    model_name: str
    threshold: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def get_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Return a continuous score for the positive class:
    - Prefer predict_proba[:,1]
    - Else use decision_function
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1].astype(float)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores, dtype=float)

    # Fallback (not recommended for AUC/PR-AUC)
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def evaluate_at_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    threshold: float = 0.5
) -> EvalResult:
    """Compute key metrics and confusion matrix at a given threshold."""
    y_true = y.astype(int).values
    scores = get_scores(model, X)

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    y_pred = (scores >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return EvalResult(
        model_name=model_name,
        threshold=float(threshold),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )

def evaluate_from_scores(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate metrics based on pre-calculated scores (prob_pos).
    Returns a dictionary suitable for pandas DataFrame construction.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    
    # Generate predictions based on threshold
    y_pred = (s >= threshold).astype(int)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(
    scores: np.ndarray,
    y_true: pd.Series,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Sweep thresholds and return a table of metrics.
    roc_auc/pr_auc are constant across thresholds and are included for convenience.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 17)]

    y = y_true.astype(int).values
    roc_auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)

    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def results_to_df(results: List[EvalResult]) -> pd.DataFrame:
    """Convert EvalResult list to DataFrame."""
    return pd.DataFrame([{
        "model": r.model_name,
        "threshold": r.threshold,
        "roc_auc": r.roc_auc,
        "pr_auc": r.pr_auc,
        "accuracy": r.accuracy,
        "precision": r.precision,
        "recall": r.recall,
        "f1": r.f1,
        "tn": r.tn,
        "fp": r.fp,
        "fn": r.fn,
        "tp": r.tp,
    } for r in results])


def _fit_with_optional_early_stopping(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
):
    """
    Fit model with optional early stopping for XGBoost (if provided).
    Safe fallback for non-XGB models.
    """
    if HAS_XGB and isinstance(model, XGBClassifier) and X_val is not None and y_val is not None:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50
            )
            return model
        except TypeError:
            # Some versions do not accept early_stopping_rounds in this wrapper config
            model.fit(X_train, y_train)
            return model

    model.fit(X_train, y_train)
    return model


def run_models_train_val(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit each model on train and evaluate on val.
    Returns:
      - results dataframe
      - fitted models dict
    """

    results: List[EvalResult] = []
    fitted: Dict[str, Any] = {}

    items = list(models.items())
    t_global = time.perf_counter()

    for idx, (name, model) in enumerate(items, start=1):
        if verbose:
            print(f"[{idx}/{len(items)}] Training: {name} ...", end=" ")

        t0 = time.perf_counter()
        _fit_with_optional_early_stopping(model, X_train, y_train, X_val, y_val)
        t_fit = time.perf_counter() - t0

        fitted[name] = model

        t1 = time.perf_counter()
        r = evaluate_at_threshold(model, X_val, y_val, model_name=name, threshold=threshold)
        t_eval = time.perf_counter() - t1
        results.append(r)

        if verbose:
            elapsed = time.perf_counter() - t_global
            avg_per_model = elapsed / idx
            eta = avg_per_model * (len(items) - idx)
            print(f"done (fit {t_fit:.1f}s, eval {t_eval:.1f}s). ETA ~{eta:.1f}s")

    df = results_to_df(results)
    df = df.sort_values(by=["pr_auc", "roc_auc", "f1"], ascending=False).reset_index(drop=True)
    return df, fitted

# =========================
# Plots & Threshold Utilities (single-model focused)
# =========================

def plot_roc_curve(y_true, scores, title="ROC Curve"):
    """Plot ROC curve for a single model."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_threshold_impact(
    df_sweep: pd.DataFrame,
    title: str = "Threshold Impact",
    metrics: Tuple[str, ...] = ("precision", "recall", "f1"),
    zoom: bool = True,
):
    """
    Plot metric-vs-threshold curves from a threshold_sweep() dataframe.
    zoom=True sets y-limits tightly to make small differences visible.
    """
    plt.figure()
    for m in metrics:
        if m not in df_sweep.columns:
            raise ValueError(f"Metric '{m}' not found in df_sweep columns.")
        plt.plot(df_sweep["threshold"], df_sweep[m], label=m.capitalize())

    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True)

    if zoom:
        vals = df_sweep[list(metrics)].to_numpy().ravel()
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        pad = max(0.002, 0.05 * (vmax - vmin + 1e-12))
        plt.ylim(vmin - pad, vmax + pad)

    plt.tight_layout()
    plt.show()


def pick_policy_thresholds(
    df_sweep: pd.DataFrame,
    *,
    recall_target: float = 0.98,
    precision_target: float = 0.98,
) -> Dict[str, float]:
    """
    Pick 3 thresholds:
    - alert_first: prioritize recall (>= recall_target), then maximize precision
    - block_first: prioritize precision (>= precision_target), then maximize recall
    - balanced: maximize F1
    """
    df = df_sweep.copy()

    # Balanced: best F1
    idx_bal = df["f1"].astype(float).idxmax()
    t_bal = float(df.loc[idx_bal, "threshold"])

    # Alert-first: high recall, best precision
    df_alert = df[df["recall"] >= recall_target]
    if len(df_alert) == 0:
        # Fallback: best recall overall, then best precision
        idx = df.sort_values(["recall", "precision"], ascending=[False, False]).index[0]
        t_alert = float(df.loc[idx, "threshold"])
    else:
        idx = df_alert.sort_values(["precision", "recall"], ascending=[False, False]).index[0]
        t_alert = float(df_alert.loc[idx, "threshold"])

    # Block-first: high precision, best recall
    df_block = df[df["precision"] >= precision_target]
    if len(df_block) == 0:
        # Fallback: best precision overall, then best recall
        idx = df.sort_values(["precision", "recall"], ascending=[False, False]).index[0]
        t_block = float(df.loc[idx, "threshold"])
    else:
        idx = df_block.sort_values(["recall", "precision"], ascending=[False, False]).index[0]
        t_block = float(df_block.loc[idx, "threshold"])

    return {
        "alert_first": t_alert,
        "balanced": t_bal,
        "block_first": t_block,
    }


def plot_confusion_matrix_from_scores(y_true, scores, threshold=0.5, title="Confusion Matrix"):
    """Plot confusion matrix at a threshold using raw scores/probabilities."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    y_pred = (s >= float(threshold)).astype(int)

    cm = confusion_matrix(y, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_three_policy_confusion_matrices(
    y_true,
    scores,
    thresholds: Dict[str, float],
):
    """
    Plot 3 confusion matrices:
    - Alert-first (high recall)
    - Balanced (best F1)
    - Block-first (high precision)
    """
    plot_confusion_matrix_from_scores(
        y_true, scores,
        threshold=thresholds["alert_first"],
        title=f"Alert-first (high recall) | thr={thresholds['alert_first']:.2f}"
    )
    plot_confusion_matrix_from_scores(
        y_true, scores,
        threshold=thresholds["balanced"],
        title=f"Balanced (best F1) | thr={thresholds['balanced']:.2f}"
    )
    plot_confusion_matrix_from_scores(
        y_true, scores,
        threshold=thresholds["block_first"],
        title=f"Block-first (high precision) | thr={thresholds['block_first']:.2f}"
    )


def plot_xgb_learning_curve_logloss(model, title="XGB Learning Curve - logloss"):
    """
    Plot train/val logloss from an XGBClassifier that was fit with eval_set and eval_metric including 'logloss'.
    """
    if not hasattr(model, "evals_result_"):
        raise ValueError("Model has no evals_result_. Fit XGB with eval_set=... and eval_metric including 'logloss'.")

    er = model.evals_result_
    if "validation_0" not in er or "validation_1" not in er:
        raise ValueError("evals_result_ missing validation_0/validation_1. Make sure you provided eval_set=[(train),(val)].")

    if "logloss" not in er["validation_0"] or "logloss" not in er["validation_1"]:
        raise ValueError("logloss not found in evals_result_. Set eval_metric to include 'logloss'.")

    tr = er["validation_0"]["logloss"]
    va = er["validation_1"]["logloss"]
    iters = np.arange(1, len(tr) + 1)

    plt.figure()
    plt.plot(iters, tr, label="Train logloss")
    plt.plot(iters, va, label="Validation logloss")
    plt.title(title)
    plt.xlabel("Boosting Rounds")
    plt.ylabel("logloss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
