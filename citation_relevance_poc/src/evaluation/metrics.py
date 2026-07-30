from __future__ import annotations

import numpy as np
import pandas as pd


def pair_metrics(scores: pd.DataFrame, score_col: str = "scincl_cosine") -> dict:
    """Summarize the SciNCL score distribution for a set of observed citation pairs.

    This pipeline only carries actual/observed citation pairs (no sampled
    negatives), so these are plain descriptive statistics rather than a
    positive-vs-negative discrimination metric.
    """
    values = pd.to_numeric(scores[score_col], errors="coerce").dropna().to_numpy()
    out = {
        "n_pairs": len(scores),
        "n_scored_pairs": int(len(values)),
        "mean_score": float(np.mean(values)) if len(values) else float("nan"),
        "median_score": float(np.median(values)) if len(values) else float("nan"),
        "std_score": float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan"),
        "p25_score": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
        "p75_score": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
    }
    return out
