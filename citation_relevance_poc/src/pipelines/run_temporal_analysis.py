from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.temporal_eval import add_temporal_groups, grouped_pair_metrics
from src.utils.io import ensure_dir, read_table, write_table


def _maybe_regression(scores: pd.DataFrame, output_path: Path) -> None:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        pd.DataFrame([{"status": "statsmodels unavailable"}]).to_csv(output_path, index=False)
        return

    df = scores.copy()
    df["post_2020"] = (df["query_period"] == "post_2020").astype(int)
    df["ai_era"] = (df["query_ai_group"] == "ai").astype(int)
    df["same_field_int"] = df["same_field"].astype(int)
    model = smf.ols("scincl_cosine ~ post_2020 + ai_era + post_2020:ai_era + label + same_field_int", data=df).fit()
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
    fig_dir = ensure_dir(output_dir.parent / "figures")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=scores, x="query_period", y="scincl_cosine", hue="label")
    plt.title("SciNCL Score by Query Period")
    plt.tight_layout()
    plt.savefig(fig_dir / "score_by_year.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=scores, x="query_ai_group", y="scincl_cosine", hue="label")
    plt.title("SciNCL Score by AI Era Group")
    plt.tight_layout()
    plt.savefig(fig_dir / "score_by_genai_status.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    scores = add_temporal_groups(read_table(args.scores))
    write_table(grouped_pair_metrics(scores, "query_period"), output_dir / "temporal_metrics.csv")
    ai_era_metrics = grouped_pair_metrics(scores, "query_ai_group")
    write_table(ai_era_metrics, output_dir / "ai_era_metrics.csv")
    write_table(ai_era_metrics, output_dir / "genai_metrics.csv")
    write_table(grouped_pair_metrics(scores, "query_genai_topic"), output_dir / "genai_topic_metrics.csv")
    write_table(grouped_pair_metrics(scores, "combined_group"), output_dir / "combined_group_metrics.csv")
    _maybe_regression(scores, output_dir / "did_regression.csv")
    _plot_temporal(scores, output_dir)
    print(f"Saved temporal outputs in {output_dir}")


if __name__ == "__main__":
    main()
