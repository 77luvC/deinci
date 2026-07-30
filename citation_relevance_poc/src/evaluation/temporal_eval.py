from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import pair_metrics


def ai_era_group(year: float | int | None) -> str:
    if pd.isna(year):
        return "unknown"
    year = int(year)
    if year < 2022:
        return "non_ai"
    if year > 2023:
        return "ai"
    return "transition"


def add_temporal_groups(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["query_year"] = pd.to_numeric(out["query_year"], errors="coerce")
    out["candidate_year"] = pd.to_numeric(out["candidate_year"], errors="coerce")
    out["query_period"] = out["query_year"].apply(
        lambda y: "unknown" if pd.isna(y) else ("pre_2020" if int(y) <= 2020 else "post_2020")
    )
    out["query_ai_group"] = out["query_year"].apply(ai_era_group)
    out["query_genai_topic"] = out["query_is_genai"].apply(lambda x: "genai_topic" if bool(x) else "non_genai_topic")
    out["combined_group"] = out["query_period"] + "_" + out["query_ai_group"]
    out["same_field"] = out["query_field"].astype(str) == out["candidate_field"].astype(str)
    return out


def grouped_pair_metrics(scores: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_name, group in scores.groupby(group_col):
        metrics = pair_metrics(group)
        metrics[group_col] = group_name
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    cols = [group_col] + [col for col in rows[0].keys() if col != group_col]
    return pd.DataFrame(rows)[cols]
