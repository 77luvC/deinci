from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from src.evaluation.temporal_eval import add_temporal_groups
from src.pipelines.run_cited_distribution_analysis import add_candidate_metadata, share_top
from src.utils.io import ensure_dir, read_table, write_table

ANALYSIS_GROUPS = ["pre_ai", "post_ai"]


BASE_METRICS = [
    "median_cited_age",
    "share_recent_refs_3yr",
    "median_log1p_cited_by_count",
    "share_low_citation_refs",
    "top10pct_share_of_citation_mass",
    "share_cross_field_refs",
    "cited_field_entropy",
]

RELEVANCE_METRICS = [
    "mean_scincl",
    "share_relevant_low_citation_refs",
]


def entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return float("nan")
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def mean_boolean(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.astype(float).mean())


def seed_level_metrics(
    actual: pd.DataFrame,
    *,
    low_citation_threshold: float,
    relevance_threshold: float,
) -> pd.DataFrame:
    rows = []
    actual = actual.copy()
    actual["valid_cited_age"] = actual["candidate_age_at_citation"].where(
        actual["candidate_age_at_citation"] >= 0
    )
    has_scincl = "scincl_cosine" in actual.columns and pd.to_numeric(
        actual["scincl_cosine"], errors="coerce"
    ).notna().any()
    if has_scincl:
        actual["scincl_cosine"] = pd.to_numeric(actual["scincl_cosine"], errors="coerce")
    actual["is_recent_ref_3yr"] = actual["valid_cited_age"].between(0, 3, inclusive="both")
    actual["is_low_citation_ref"] = actual["candidate_visibility_cited_by_count"] <= low_citation_threshold
    if has_scincl:
        actual["is_relevant_low_citation_ref"] = (
            actual["is_low_citation_ref"] & (actual["scincl_cosine"] >= relevance_threshold)
        )

    for query_id, group in actual.groupby("query_id"):
        row = {
            "query_id": query_id,
            "query_ai_group": group["query_ai_group"].iloc[0],
            "query_year": group["query_year"].iloc[0],
            "query_field": group["query_field"].iloc[0],
            "n_reference_pairs": len(group),
            "median_cited_age": group["valid_cited_age"].median(),
            "share_recent_refs_3yr": mean_boolean(group["is_recent_ref_3yr"]),
            "median_log1p_cited_by_count": group["candidate_log1p_visibility_cited_by_count"].median(),
            "share_low_citation_refs": mean_boolean(group["is_low_citation_ref"]),
            "top10pct_share_of_citation_mass": share_top(group["candidate_visibility_cited_by_count"], 0.10),
            "share_cross_field_refs": mean_boolean(
                group["query_field"].astype(str) != group["candidate_field"].astype(str)
            ),
            "cited_field_entropy": entropy(group["candidate_field"]),
        }
        if has_scincl:
            row["mean_scincl"] = group["scincl_cosine"].mean()
            row["share_relevant_low_citation_refs"] = mean_boolean(group["is_relevant_low_citation_ref"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["query_ai_group", "query_id"])


def available_metrics(seed_metrics: pd.DataFrame) -> list[str]:
    metrics = [metric for metric in BASE_METRICS if metric in seed_metrics.columns]
    metrics.extend(metric for metric in RELEVANCE_METRICS if metric in seed_metrics.columns)
    return metrics


def period_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in seed_metrics.groupby("query_ai_group"):
        for metric in available_metrics(seed_metrics):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "query_ai_group": group_name,
                    "metric": metric,
                    "n_seed_papers": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                    "sem": values.sem(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return out
    order = valid.sort_values().index
    ranked = valid.loc[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out.loc[order] = adjusted
    return out


def compare_groups(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("post_ai", "pre_ai"),
    ]
    rows = []
    for metric in available_metrics(seed_metrics):
        for group_a, group_b in comparisons:
            values_a = pd.to_numeric(
                seed_metrics.loc[seed_metrics["query_ai_group"] == group_a, metric],
                errors="coerce",
            ).dropna()
            values_b = pd.to_numeric(
                seed_metrics.loc[seed_metrics["query_ai_group"] == group_b, metric],
                errors="coerce",
            ).dropna()

            row = {
                "metric": metric,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(values_a),
                "n_b": len(values_b),
                "mean_a": values_a.mean(),
                "mean_b": values_b.mean(),
                "mean_diff_a_minus_b": values_a.mean() - values_b.mean(),
                "median_a": values_a.median(),
                "median_b": values_b.median(),
                "median_diff_a_minus_b": values_a.median() - values_b.median(),
            }

            if len(values_a) >= 2 and len(values_b) >= 2:
                t_result = ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
                u_result = mannwhitneyu(values_a, values_b, alternative="two-sided")
                row.update(
                    {
                        "welch_t_stat": float(t_result.statistic),
                        "welch_p_value": float(t_result.pvalue),
                        "mannwhitney_u_stat": float(u_result.statistic),
                        "mannwhitney_p_value": float(u_result.pvalue),
                    }
                )
            else:
                row.update(
                    {
                        "welch_t_stat": float("nan"),
                        "welch_p_value": float("nan"),
                        "mannwhitney_u_stat": float("nan"),
                        "mannwhitney_p_value": float("nan"),
                    }
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    out["welch_q_value_bh"] = benjamini_hochberg(out["welch_p_value"])
    out["mannwhitney_q_value_bh"] = benjamini_hochberg(out["mannwhitney_p_value"])
    return out


def metric_definitions(
    low_citation_threshold: float,
    relevance_threshold: float,
    *,
    include_relevance: bool,
) -> pd.DataFrame:
    definitions = [
        ("median_cited_age", "Median valid cited-work age per seed: query year minus cited-work year."),
        ("share_recent_refs_3yr", "Share of references with cited-work age between 0 and 3 years."),
        (
            "median_log1p_cited_by_count",
            "Median log(1 + OpenAlex cited-by count at the seed paper's citation year) of referenced works.",
        ),
        (
            "share_low_citation_refs",
            f"Share of references with citation-year cited-by count <= global bottom-quartile threshold ({low_citation_threshold}).",
        ),
        (
            "top10pct_share_of_citation_mass",
            "Share of each seed's referenced-work citation-count mass held by its top 10% most-cited references.",
        ),
        ("share_cross_field_refs", "Share of references whose OpenAlex field differs from the seed-paper field."),
        ("cited_field_entropy", "Shannon entropy of cited-work OpenAlex field distribution per seed."),
    ]
    if include_relevance:
        definitions.extend(
            [
                ("mean_scincl", "Mean SciNCL cosine score across actual references for the seed."),
                (
                    "share_relevant_low_citation_refs",
                    f"Share of references that are low-citation and have SciNCL cosine >= {relevance_threshold}.",
                ),
            ]
        )
    return pd.DataFrame(definitions, columns=["metric", "definition"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-citation-quantile", type=float, default=0.25)
    parser.add_argument("--relevance-threshold", type=float, default=0.80)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    scores = add_temporal_groups(read_table(args.scores))
    scores = scores[scores["query_ai_group"].isin(ANALYSIS_GROUPS)].copy()
    papers = read_table(args.papers)
    enriched = add_candidate_metadata(scores, papers)
    actual = enriched[enriched["label"].astype(int) == 1].copy()
    actual["candidate_visibility_cited_by_count"] = pd.to_numeric(
        actual["candidate_visibility_cited_by_count"], errors="coerce"
    )
    low_citation_threshold = actual["candidate_visibility_cited_by_count"].quantile(args.low_citation_quantile)

    seed_metrics = seed_level_metrics(
        actual,
        low_citation_threshold=low_citation_threshold,
        relevance_threshold=args.relevance_threshold,
    )
    include_relevance = any(metric in seed_metrics.columns for metric in RELEVANCE_METRICS)

    write_table(seed_metrics, output_dir / "seed_democratization_metrics.csv")
    write_table(period_summary(seed_metrics), output_dir / "period_democratization_summary.csv")
    write_table(compare_groups(seed_metrics), output_dir / "period_democratization_comparisons.csv")
    write_table(
        metric_definitions(
            low_citation_threshold,
            args.relevance_threshold,
            include_relevance=include_relevance,
        ),
        output_dir / "metric_definitions.csv",
    )
    print(f"Saved democratization outputs in {output_dir}")


if __name__ == "__main__":
    main()
