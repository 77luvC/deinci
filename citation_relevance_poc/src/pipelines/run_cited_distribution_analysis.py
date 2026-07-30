from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.temporal_eval import add_temporal_groups
from src.utils.io import ensure_dir, read_table, write_table


def gini(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[x >= 0]
    if len(x) == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def share_top(values: pd.Series, frac: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(len(x) * frac)))
    return float(np.sort(x)[-n_top:].sum() / x.sum())


def add_candidate_metadata(scores: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    candidates = papers.add_prefix("candidate_").rename(columns={"candidate_paper_id": "candidate_id"})
    queries = papers.add_prefix("query_").rename(columns={"query_paper_id": "query_id"})
    out = scores.merge(candidates, on="candidate_id", how="left", suffixes=("", "_from_papers"))
    out = out.merge(
        queries[["query_id", "query_publication_date", "query_work_type", "query_venue"]],
        on="query_id",
        how="left",
        suffixes=("", "_from_papers"),
    )
    out["query_year"] = pd.to_numeric(out["query_year"], errors="coerce")
    out["candidate_year"] = pd.to_numeric(out["candidate_year"], errors="coerce")
    out["candidate_age_at_citation"] = out["query_year"] - out["candidate_year"]
    out["candidate_current_cited_by_count"] = pd.to_numeric(
        out["candidate_openalex_current_cited_by_count"], errors="coerce"
    )
    out["candidate_log1p_current_cited_by_count"] = np.log1p(out["candidate_current_cited_by_count"])
    out["same_field"] = out["query_field"].astype(str) == out["candidate_field"].astype(str)
    return out


def summarize_actual_references(actual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in actual.groupby("query_ai_group"):
        citation_count = group["candidate_current_cited_by_count"]
        rows.append(
            {
                "query_ai_group": group_name,
                "n_reference_pairs": len(group),
                "n_query_papers": group["query_id"].nunique(),
                "mean_scincl": group["scincl_cosine"].mean(),
                "median_scincl": group["scincl_cosine"].median(),
                "mean_cited_age": group["candidate_age_at_citation"].mean(),
                "median_cited_age": group["candidate_age_at_citation"].median(),
                "mean_current_cited_by_count": citation_count.mean(),
                "median_current_cited_by_count": citation_count.median(),
                "p25_current_cited_by_count": citation_count.quantile(0.25),
                "p75_current_cited_by_count": citation_count.quantile(0.75),
                "mean_log1p_current_cited_by_count": group["candidate_log1p_current_cited_by_count"].mean(),
                "gini_current_cited_by_count": gini(citation_count),
                "top10pct_share_of_current_citation_mass": share_top(citation_count, 0.10),
                "top25pct_share_of_current_citation_mass": share_top(citation_count, 0.25),
                "share_open_access": group["candidate_is_oa"].astype("boolean").mean(),
                "share_same_field": group["same_field"].mean(),
                "share_article": (group["candidate_work_type"] == "article").mean(),
                "share_book": (group["candidate_work_type"] == "book").mean(),
            }
        )
    return pd.DataFrame(rows)


def distribution_table(actual: pd.DataFrame, column: str, output_col: str) -> pd.DataFrame:
    table = (
        actual.groupby(["query_ai_group", column], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={column: output_col})
    )
    totals = table.groupby("query_ai_group")["n"].transform("sum")
    table["share"] = table["n"] / totals
    return table.sort_values(["query_ai_group", "n"], ascending=[True, False])


def visibility_bins(actual: pd.DataFrame) -> pd.DataFrame:
    out = actual.copy()
    counts = out["candidate_current_cited_by_count"]
    try:
        out["visibility_quartile"] = pd.qcut(
            counts.rank(method="first"),
            q=4,
            labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
        )
    except ValueError:
        out["visibility_quartile"] = "unclear"
    table = out.groupby(["query_ai_group", "visibility_quartile"]).size().reset_index(name="n")
    table["share"] = table["n"] / table.groupby("query_ai_group")["n"].transform("sum")
    return table


def write_plots(actual: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = ensure_dir(output_dir / "figures")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=actual, x="query_ai_group", y="candidate_log1p_current_cited_by_count")
    plt.title("Cited Work Visibility by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "cited_visibility_by_ai_group.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=actual, x="query_ai_group", y="candidate_age_at_citation")
    plt.title("Cited Work Age by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "cited_age_by_ai_group.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=actual, x="query_ai_group", y="scincl_cosine")
    plt.title("SciNCL Relevance Proxy for Actual References")
    plt.tight_layout()
    plt.savefig(fig_dir / "actual_reference_scincl_by_ai_group.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    scores = add_temporal_groups(read_table(args.scores))
    papers = read_table(args.papers)
    enriched = add_candidate_metadata(scores, papers)
    actual = enriched[enriched["label"].astype(int) == 1].copy()

    write_table(enriched, output_dir / "all_pairs_with_candidate_metadata.parquet")
    write_table(actual, output_dir / "actual_references_with_candidate_metadata.parquet")
    write_table(summarize_actual_references(actual), output_dir / "cited_distribution_summary.csv")
    write_table(visibility_bins(actual), output_dir / "cited_visibility_quartiles.csv")
    write_table(distribution_table(actual, "candidate_work_type", "candidate_work_type"), output_dir / "cited_work_type_distribution.csv")
    write_table(distribution_table(actual, "candidate_field", "candidate_field"), output_dir / "cited_field_distribution.csv")
    write_table(distribution_table(actual, "candidate_venue", "candidate_venue"), output_dir / "cited_venue_distribution.csv")
    write_plots(actual, output_dir)
    print(f"Saved cited distribution outputs in {output_dir}")


if __name__ == "__main__":
    main()
