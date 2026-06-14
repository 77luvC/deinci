from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    pooled_var = ((len(positive) - 1) * np.var(positive, ddof=1) + (len(negative) - 1) * np.var(negative, ddof=1))
    pooled_var /= len(positive) + len(negative) - 2
    if pooled_var <= 0:
        return float("nan")
    return float((np.mean(positive) - np.mean(negative)) / math.sqrt(pooled_var))


def pair_metrics(scores: pd.DataFrame, score_col: str = "scincl_cosine") -> dict:
    labels = scores["label"].astype(int).to_numpy()
    values = scores[score_col].astype(float).to_numpy()
    positive = values[labels == 1]
    negative = values[labels == 0]

    out = {
        "n_pairs": len(scores),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "mean_positive_score": float(np.mean(positive)) if len(positive) else float("nan"),
        "mean_negative_score": float(np.mean(negative)) if len(negative) else float("nan"),
        "positive_negative_diff": float(np.mean(positive) - np.mean(negative)) if len(positive) and len(negative) else float("nan"),
        "cohens_d": cohens_d(positive, negative),
    }
    if len(set(labels)) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, values))
        out["average_precision"] = float(average_precision_score(labels, values))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out

