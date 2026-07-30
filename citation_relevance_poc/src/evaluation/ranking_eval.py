from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(labels: np.ndarray, k: int) -> float:
    gains = labels[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains * discounts))


def ranking_metrics(scores: pd.DataFrame, ks: tuple[int, ...] = (10, 50, 100, 300)) -> pd.DataFrame:
    rows = []
    for query_id, group in scores.groupby("query_id"):
        ranked = group.sort_values("scincl_cosine", ascending=False)
        labels = ranked["label"].astype(int).to_numpy()
        total_pos = int(labels.sum())
        if total_pos == 0:
            continue

        row = {"query_id": query_id, "n_candidates": len(ranked), "n_positive": total_pos}
        positive_positions = np.where(labels == 1)[0]
        row["mrr"] = float(1.0 / (positive_positions[0] + 1)) if len(positive_positions) else 0.0

        ideal = np.sort(labels)[::-1]
        for k in ks:
            row[f"recall@{k}"] = float(labels[:k].sum() / total_pos)
            ideal_dcg = _dcg(ideal, k)
            row[f"ndcg@{k}"] = float(_dcg(labels, k) / ideal_dcg) if ideal_dcg > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)

