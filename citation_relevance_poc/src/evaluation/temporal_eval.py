from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import pair_metrics

AI_ERA_WINDOWS = {
    "pre_ai": ("2019-01-01", "2021-12-31"),
    "post_ai": ("2023-01-01", "2026-06-30"),
}

_AI_ERA_WINDOWS_TS = {
    label: (pd.Timestamp(start), pd.Timestamp(end))
    for label, (start, end) in AI_ERA_WINDOWS.items()
}


def ai_era_group(year: float | int | None) -> str:
    if pd.isna(year):
        return "unknown"
    year = int(year)
    if 2019 <= year <= 2021:
        return "pre_ai"
    if 2023 <= year <= 2026:
        return "post_ai"
    return "outside_window"


def ai_era_group_from_date(publication_date: object, year: float | int | None) -> str:
    date = pd.to_datetime(publication_date, errors="coerce")
    if not pd.isna(date):
        for label, (start, end) in _AI_ERA_WINDOWS_TS.items():
            if start <= date <= end:
                return label
        return "outside_window"
    return ai_era_group(year)


def add_temporal_groups(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["query_year"] = pd.to_numeric(out["query_year"], errors="coerce")
    out["candidate_year"] = pd.to_numeric(out["candidate_year"], errors="coerce")
    out["query_period"] = out["query_year"].apply(
        lambda y: "unknown" if pd.isna(y) else ("pre_2020" if int(y) <= 2020 else "post_2020")
    )
    if "query_publication_date" in out.columns:
        out["query_ai_group"] = out.apply(
            lambda row: ai_era_group_from_date(row.get("query_publication_date"), row.get("query_year")),
            axis=1,
        )
    else:
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
