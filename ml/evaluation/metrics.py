"""
ml/evaluation/metrics.py

Shared metric computation functions used by both Layer 2A and 2B evaluators.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, accuracy_score,
)

CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack"]


def anomaly_metrics(y_true: np.ndarray, scores: np.ndarray,
                    preds: np.ndarray) -> dict:
    """
    Metrics for one-class anomaly detection (Layer 2A).

    Parameters
    ----------
    y_true  : (N,) int — 0=normal, 1=attack
    scores  : (N,) float — anomaly scores (higher = more anomalous)
    preds   : (N,) int — binary predictions (1=anomaly, 0=normal)
    """
    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "auc":           round(float(auc), 4),
        "avg_precision": round(float(ap), 4),
        "fpr":           round(float(fpr), 4),
        "tpr":           round(float(tpr), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def classification_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict:
    """
    Metrics for multi-class classification (Layer 2B).

    Parameters
    ----------
    y_true : (N,) int — class labels 0-4
    preds  : (N,) int — predicted class labels
    """
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, preds)
    cm       = confusion_matrix(y_true, preds, labels=list(range(5))).tolist()

    report = classification_report(
        y_true, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = {cls: round(report[cls]["f1-score"], 4) for cls in CLASS_NAMES}

    return {
        "macro_f1":         round(float(macro_f1), 4),
        "accuracy":         round(float(accuracy), 4),
        "per_class_f1":     per_class,
        "confusion_matrix": cm,
    }