from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import pair_metrics
from src.evaluation.ranking_eval import ranking_metrics
from src.utils.io import ensure_dir, read_table, write_table

LIMITATION_TEXT = (
    "SciNCL is used here as a citation-informed relevance proxy, not as a citation quality judge. "
    "A high SciNCL score suggests topical or scholarly relatedness between two papers, but it does "
    "not prove that a citation supports a specific claim, uses the best available evidence, or is "
    "non-hallucinated. The pretrained SciNCL model is based on historical citation information from "
    "S2ORC 20200705v1, so scores for post-2020 and GenAI-era papers should be interpreted with caution."
)


def _plot_score_distribution(scores: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = ensure_dir(output_dir.parent / "figures")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=scores, x="scincl_cosine", hue="label", bins=20, kde=True)
    plt.title("SciNCL Score Distribution by Label")
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution_by_label.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=scores, x="label_type", y="scincl_cosine")
    plt.xticks(rotation=30, ha="right")
    plt.title("SciNCL Score by Label Type")
    plt.tight_layout()
    plt.savefig(fig_dir / "score_distribution_by_label_type.png", dpi=160)
    plt.close()


def _write_report(scores: pd.DataFrame, metrics: dict, ranking: pd.DataFrame, output_dir: Path) -> None:
    report_path = output_dir.parent / "demo_report.md"
    ranking_means = ranking.drop(columns=["query_id"], errors="ignore").mean(numeric_only=True)
    lines = [
        "# Demo SciNCL POC Report",
        "",
        "## Dataset",
        "",
        f"- Number of scored pairs: {len(scores)}",
        f"- Number of query papers: {scores['query_id'].nunique()}",
        f"- Number of positive pairs: {int(scores['label'].sum())}",
        f"- Number of negative pairs: {int((scores['label'] == 0).sum())}",
        "- SciNCL model version: `malteos/scincl`",
        "",
        "## Pair Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Ranking Metrics Mean", ""])
    for key, value in ranking_means.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Limitations", "", LIMITATION_TEXT, ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    scores = read_table(args.scores)
    metrics = pair_metrics(scores)
    ranking = ranking_metrics(scores)

    write_table(pd.DataFrame([metrics]), output_dir / "metrics_summary.csv")
    write_table(ranking, output_dir / "ranking_metrics.csv")
    _plot_score_distribution(scores, output_dir)
    _write_report(scores, metrics, ranking, output_dir)
    print(f"Saved evaluation outputs in {output_dir}")


if __name__ == "__main__":
    main()
