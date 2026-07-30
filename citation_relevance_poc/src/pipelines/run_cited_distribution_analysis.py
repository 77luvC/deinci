from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.temporal_eval import add_temporal_groups
from src.utils.io import ensure_dir, read_table, write_table

ANALYSIS_GROUPS = ["pre_ai", "post_ai"]

# This analysis describes the cited works behind actual references only. It uses
# citation counts as of the query paper's citation year when available, falling
# back to current OpenAlex counts otherwise.


def gini(values: pd.Series) -> float:
    """Compute the Gini coefficient for non-negative numeric values.

    Missing and negative values are dropped. Empty inputs return NaN, while an
    all-zero distribution has zero inequality by definition.
    """
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
    """Return the share of total mass held by the top fraction of values.

    Values are coerced to non-negative numeric observations. Empty or zero-mass
    inputs return NaN because no concentration share is defined.
    """
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(len(x) * frac)))
    return float(np.sort(x)[-n_top:].sum() / x.sum())


def add_candidate_metadata(scores: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    """Attach query and candidate paper metadata to scored pair rows.

    Expects `scores` to include query/candidate IDs and basic year/field columns,
    and `papers` to use the canonical paper schema. Citation-year counts are
    preferred when present; otherwise current OpenAlex cited-by counts become the
    visibility fallback.
    """
    # Prefix the full paper table twice so candidate and query metadata can
    # coexist in a single analysis row without column-name collisions. The
    # prefix naturally reproduces the pairs table's own FK names (`paper_id`
    # -> `candidate_paper_id`/`query_paper_id`), so no rename is needed here.
    candidates = papers.add_prefix("candidate_")
    queries = papers.add_prefix("query_")
    out = scores.merge(candidates, on="candidate_paper_id", how="left", suffixes=("", "_from_papers"))
    out = out.merge(
        queries[["query_paper_id", "query_publication_date", "query_work_type", "query_venue"]],
        on="query_paper_id",
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
    if "candidate_cited_by_count_at_query_year" in out.columns:
        # Historical counts are the preferred visibility measure because they
        # avoid crediting a cited work for citations accumulated after the query.
        out["candidate_visibility_cited_by_count"] = pd.to_numeric(
            out["candidate_cited_by_count_at_query_year"], errors="coerce"
        )
        out["candidate_visibility_source"] = "citation_year"
    else:
        # Metadata-only runs may lack historical counts; keep analyses available
        # but mark the provenance of the fallback.
        out["candidate_visibility_cited_by_count"] = out["candidate_current_cited_by_count"]
        out["candidate_visibility_source"] = "current_fallback"
    out["candidate_log1p_visibility_cited_by_count"] = np.log1p(out["candidate_visibility_cited_by_count"])
    out["same_field"] = out["query_field"].astype(str) == out["candidate_field"].astype(str)
    return out


def summarize_actual_references(actual: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual-reference visibility and metadata by AI-era group."""
    rows = []
    for group_name, group in actual.groupby("query_ai_group"):
        citation_count = group["candidate_visibility_cited_by_count"]
        scincl = pd.to_numeric(group.get("scincl_cosine", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "query_ai_group": group_name,
                "n_reference_pairs": len(group),
                "n_query_papers": group["query_paper_id"].nunique(),
                "mean_score": scincl.mean(),
                "median_score": scincl.median(),
                "mean_cited_age": group["candidate_age_at_citation"].mean(),
                "median_cited_age": group["candidate_age_at_citation"].median(),
                "mean_visibility_cited_by_count": citation_count.mean(),
                "median_visibility_cited_by_count": citation_count.median(),
                "p25_visibility_cited_by_count": citation_count.quantile(0.25),
                "p75_visibility_cited_by_count": citation_count.quantile(0.75),
                "mean_log1p_visibility_cited_by_count": group["candidate_log1p_visibility_cited_by_count"].mean(),
                "gini_visibility_cited_by_count": gini(citation_count),
                "top10pct_share_of_visibility_citation_mass": share_top(citation_count, 0.10),
                "top25pct_share_of_visibility_citation_mass": share_top(citation_count, 0.25),
                "share_open_access": group["candidate_is_oa"].astype("boolean").mean(),
                "share_same_field": group["same_field"].mean(),
                "share_article": (group["candidate_work_type"] == "article").mean(),
                "share_book": (group["candidate_work_type"] == "book").mean(),
            }
        )
    return pd.DataFrame(rows)


def distribution_table(actual: pd.DataFrame, column: str, output_col: str) -> pd.DataFrame:
    """Count and share a categorical cited-work column within each AI-era group.

    Missing categories are retained with `dropna=False`, which is important for
    auditing metadata coverage rather than silently omitting unknown venues or
    fields.
    """
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
    """Assign cited works to global visibility quartiles and summarize shares.

    Ranking before qcut avoids duplicate-edge failures when many counts are tied.
    If qcut still fails, all rows are marked `unclear` so the output table keeps
    a visible edge-case signal.
    """
    out = actual.copy()
    counts = out["candidate_visibility_cited_by_count"]
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


def write_plots(actual: pd.DataFrame, output_dir: Path, tables_dir: Path) -> None:
    """Write diagnostic plots for cited-work visibility distributions.

    Expects `actual` to contain only positive/actual-reference rows enriched by
    `add_candidate_metadata`. Plotting is descriptive; statistical tests live in
    the democratization analysis pipeline.
    """
    fig_dir = ensure_dir(output_dir / "figures")
    # Group-level boxplots provide compact comparisons of visibility and age.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=actual, x="query_ai_group", y="candidate_log1p_visibility_cited_by_count")
    plt.title("Cited Work Citation-Year Visibility by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "cited_visibility_by_ai_group.png", dpi=160)
    plt.close()

    # SciNCL may be absent in metadata-only pair files, so only draw this plot
    # when at least one numeric score is available.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=actual, x="query_ai_group", y="candidate_age_at_citation")
    plt.title("Cited Work Age by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "cited_age_by_ai_group.png", dpi=160)
    plt.close()

    # Raw distribution views expose skew and heavy tails that boxplots can hide.
    plt.figure(figsize=(8, 5))
    scincl = pd.to_numeric(actual.get("scincl_cosine", pd.Series(dtype=float)), errors="coerce")
    if scincl.notna().any():
        plot_df = actual.copy()
        plot_df["scincl_cosine"] = scincl
        sns.boxplot(data=plot_df, x="query_ai_group", y="scincl_cosine")
        plt.title("SciNCL Relevance Proxy for Actual References")
        plt.tight_layout()
        plt.savefig(fig_dir / "actual_reference_scincl_by_ai_group.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=actual,
        x="candidate_log1p_visibility_cited_by_count",
        hue="query_ai_group",
        bins=12,
        kde=True,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.xlabel("log(1 + citation-year cited-by count)")
    plt.ylabel("Number of actual reference pairs")
    plt.title("Row-Level Distribution of Cited-Work Visibility")
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_visibility_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.ecdfplot(
        data=actual,
        x="candidate_visibility_cited_by_count",
        hue="query_ai_group",
        complementary=False,
    )
    plt.xscale("log")
    plt.xlabel("Citation-year cited-by count, log scale")
    plt.ylabel("Cumulative share of actual reference pairs")
    plt.title("ECDF of Cited-Work Visibility")
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_visibility_ecdf.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=actual,
        x="query_ai_group",
        y="candidate_log1p_visibility_cited_by_count",
        inner="quartile",
        cut=0,
        color="#d7e8f7",
    )
    sns.stripplot(
        data=actual,
        x="query_ai_group",
        y="candidate_log1p_visibility_cited_by_count",
        color="#1f2937",
        alpha=0.65,
        jitter=0.18,
        size=4,
    )
    plt.xlabel("")
    plt.ylabel("log(1 + citation-year cited-by count)")
    plt.title("Every Actual Reference Pair by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_visibility_violin_points.png", dpi=160)
    plt.close()

    # Rank-size curves make citation concentration visible without binning.
    ranked = []
    for group_name, group in actual.groupby("query_ai_group"):
        values = group.sort_values("candidate_visibility_cited_by_count", ascending=False).reset_index(drop=True)
        values = values.assign(rank=np.arange(1, len(values) + 1))
        ranked.append(values)
    ranked_df = pd.concat(ranked, ignore_index=True)
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=ranked_df,
        x="rank",
        y="candidate_visibility_cited_by_count",
        hue="query_ai_group",
        marker="o",
    )
    plt.yscale("log")
    plt.xlabel("Rank within group, sorted by citation-year cited-by count")
    plt.ylabel("Citation-year cited-by count, log scale")
    plt.title("Rank-Size Plot of Actual Cited Works")
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_visibility_rank_size.png", dpi=160)
    plt.close()

    # Per-query summaries prevent queries with many references from dominating
    # every row-level visualization.
    per_query = (
        actual.groupby(["query_ai_group", "query_paper_id"], as_index=False)
        .agg(
            n_reference_pairs=("candidate_paper_id", "count"),
            median_visibility_cited_by_count=("candidate_visibility_cited_by_count", "median"),
            mean_score=("scincl_cosine", "mean"),
        )
        .sort_values(["query_ai_group", "median_visibility_cited_by_count"])
    )
    plt.figure(figsize=(8, 5))
    sns.stripplot(
        data=per_query,
        x="query_ai_group",
        y="median_visibility_cited_by_count",
        hue="n_reference_pairs",
        jitter=0.18,
        size=7,
    )
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Per-query median cited-work citation-year cited-by count")
    plt.title("Per-Query Distribution of Actual References")
    plt.tight_layout()
    plt.savefig(fig_dir / "raw_per_query_visibility_points.png", dpi=160)
    plt.close()

    # This is tabular data (not an image), so it belongs in `tables/` even
    # though it's derived alongside the per-query figure above.
    per_query.to_csv(tables_dir / "raw_per_query_reference_distribution.csv", index=False)


def main() -> None:
    """Create cited-work distribution tables and plots for actual references."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    scores = add_temporal_groups(read_table(args.scores))
    # Match the rest of the field-scan analysis by excluding transition and
    # unknown temporal groups.
    scores = scores[scores["query_ai_group"].isin(ANALYSIS_GROUPS)].copy()
    papers = read_table(args.papers)
    enriched = add_candidate_metadata(scores, papers)
    # Distribution analysis is about what papers actually cite, so sampled
    # negative/comparison candidates are removed before summaries.
    actual = enriched[enriched["label"].astype(int) == 1].copy()
    if "scincl_cosine" in actual.columns:
        actual["scincl_cosine"] = pd.to_numeric(actual["scincl_cosine"], errors="coerce")

    write_table(enriched, tables_dir / "all_pairs_with_candidate_metadata.parquet")
    write_table(actual, tables_dir / "actual_references_with_candidate_metadata.parquet")
    write_table(summarize_actual_references(actual), tables_dir / "cited_distribution_summary.csv")
    write_table(visibility_bins(actual), tables_dir / "cited_visibility_quartiles.csv")
    write_table(distribution_table(actual, "candidate_work_type", "candidate_work_type"), tables_dir / "cited_work_type_distribution.csv")
    write_table(distribution_table(actual, "candidate_field", "candidate_field"), tables_dir / "cited_field_distribution.csv")
    write_table(distribution_table(actual, "candidate_venue", "candidate_venue"), tables_dir / "cited_venue_distribution.csv")
    write_plots(actual, output_dir, tables_dir)
    print(f"Saved cited distribution outputs in {output_dir}")


if __name__ == "__main__":
    main()
