"""Evaluation metrics for rare-event fraud detection (Week 8). No accuracy."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pr_auc(y_true, scores):
    """Average precision (area under the precision-recall curve). Main metric."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores):
    """ROC-AUC, for reference only (looks optimistic on rare classes)."""
    return float(roc_auc_score(y_true, scores))


def recall_at_precision(y_true, scores, min_precision=0.90):
    """Highest recall reachable while keeping precision >= min_precision.

    Returns (recall, threshold). (0.0, None) if the precision is never reached.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= min_precision          # align with thresholds
    if not ok.any():
        return 0.0, None
    best = np.argmax(np.where(ok, recall[:-1], -1))
    return float(recall[:-1][best]), float(thresholds[best])


def precision_recall_at_threshold(y_true, scores, threshold=0.5):
    """Precision, recall and number of alerts when flagging scores >= threshold."""
    pred = (np.asarray(scores) >= threshold).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    return float(p), float(r), int(pred.sum())


def evaluate(y_true, scores, threshold=0.5, min_precision=0.90):
    """All metrics in one dict (one row of the results table)."""
    r90, _ = recall_at_precision(y_true, scores, min_precision)
    p, r, alerts = precision_recall_at_threshold(y_true, scores, threshold)
    return {
        "PR-AUC": pr_auc(y_true, scores),
        "ROC-AUC": roc_auc(y_true, scores),
        f"recall@{int(min_precision * 100)}%precision": r90,
        f"precision@{threshold}": p,
        f"recall@{threshold}": r,
        "alerts": alerts,
    }
