from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.temporal_eval import add_temporal_groups, grouped_pair_metrics
from src.utils.io import ensure_dir, read_table, write_table

ANALYSIS_GROUPS = ["pre_ai", "post_ai"]


def _maybe_regression(scores: pd.DataFrame, output_path: Path) -> None:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        pd.DataFrame([{"status": "statsmodels unavailable"}]).to_csv(output_path, index=False)
        return

    df = scores.copy()
    df["post_2020"] = (df["query_period"] == "post_2020").astype(int)
    df["post_ai"] = (df["query_ai_group"] == "post_ai").astype(int)
    df["same_field_int"] = df["same_field"].astype(int)
    model = smf.ols(
        "scincl_cosine ~ post_2020 + post_ai + post_2020:post_ai + same_field_int",
        data=df,
    ).fit()
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "p_value": model.pvalues.values,
        }
    )
    table.to_csv(output_path, index=False)


def _plot_temporal(scores: pd.DataFrame, output_dir: Path) -> None:
    # Figures live under this field's own `figures/` dir (not a shared
    # `results/` path) so runs across multiple fields/datasets, and different
    # analysis stages within one field, do not overwrite each other's plots.
    fig_dir = ensure_dir(output_dir / "figures")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=scores, x="query_period", y="scincl_cosine")
    plt.title("SciNCL Score by Query Period")
    plt.tight_layout()
    plt.savefig(fig_dir / "score_by_year.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=scores, x="query_ai_group", y="scincl_cosine")
    plt.title("SciNCL Score by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "score_by_ai_era.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    scores = add_temporal_groups(read_table(args.scores))
    scores = scores[scores["query_ai_group"].isin(ANALYSIS_GROUPS)].copy()
    write_table(grouped_pair_metrics(scores, "query_period"), tables_dir / "temporal_metrics.csv")
    write_table(grouped_pair_metrics(scores, "query_ai_group"), tables_dir / "ai_era_metrics.csv")
    write_table(grouped_pair_metrics(scores, "query_genai_topic"), tables_dir / "genai_topic_metrics.csv")
    write_table(grouped_pair_metrics(scores, "combined_group"), tables_dir / "combined_group_metrics.csv")
    _maybe_regression(scores, tables_dir / "did_regression.csv")
    _plot_temporal(scores, output_dir)
    print(f"Saved temporal outputs in {output_dir}")


if __name__ == "__main__":
    main()
