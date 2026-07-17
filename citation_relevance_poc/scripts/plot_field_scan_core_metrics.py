"""Plot core field-scan metrics with post-pre differences and p-values."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


FIELD_ORDER = [
    "CS",
    "Health Informatics",
    "Public Health",
    "Psychology",
    "Social Sciences",
]

METRICS = {
    "median_log1p_cited_by_count": {
        "label": "Median log(1 + historical citations)",
        "short": "Median log citations",
        "scale": 1.0,
        "unit": "log points",
    },
    "share_low_citation_refs": {
        "label": "Share of low-citation references",
        "short": "Low-citation share",
        "scale": 100.0,
        "unit": "percentage points",
    },
    "top10pct_share_of_citation_mass": {
        "label": "Top 10% share of citation mass",
        "short": "Top-10% concentration",
        "scale": 100.0,
        "unit": "percentage points",
    },
    "share_recent_refs_3yr": {
        "label": "Share of references from past 3 years",
        "short": "Recent-reference share",
        "scale": 100.0,
        "unit": "percentage points",
    },
    "share_cross_field_refs": {
        "label": "Share of cross-field references",
        "short": "Cross-field share",
        "scale": 100.0,
        "unit": "percentage points",
    },
    "cited_field_entropy": {
        "label": "Cited-field entropy",
        "short": "Field entropy",
        "scale": 1.0,
        "unit": "entropy points",
    },
}

GROUPS = {
    "visibility": [
        "median_log1p_cited_by_count",
        "share_low_citation_refs",
        "top10pct_share_of_citation_mass",
    ],
    "recency": [
        "share_recent_refs_3yr",
    ],
    "breadth": [
        "share_cross_field_refs",
        "cited_field_entropy",
    ],
}


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<.001"
    if p_value < 0.1:
        return f"p={p_value:.3f}".replace("0.", ".")
    return f"p={p_value:.2f}".replace("0.", ".")


def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset"] = pd.Categorical(df["dataset"], categories=FIELD_ORDER, ordered=True)
    df = df.sort_values(["dataset", "metric"]).reset_index(drop=True)
    missing = sorted(set(METRICS) - set(df["metric"]))
    if missing:
        raise ValueError(f"Missing metrics: {missing}")
    return df


def plot_metric_group(df: pd.DataFrame, group_name: str, metrics: list[str], out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
        }
    )

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.7 * len(metrics), 5.25),
        sharey=True,
        constrained_layout=False,
    )
    if len(metrics) == 1:
        axes = [axes]

    y = np.arange(len(FIELD_ORDER))
    for ax, metric in zip(axes, metrics):
        spec = METRICS[metric]
        sub = (
            df[df["metric"] == metric]
            .set_index("dataset")
            .loc[FIELD_ORDER]
            .reset_index()
        )
        values = sub["post_minus_pre"].to_numpy() * spec["scale"]
        p_values = sub["welch_p"].to_numpy()

        colors = []
        for value, p_value in zip(values, p_values):
            if p_value < 0.05 and value >= 0:
                colors.append("#2F6FB0")
            elif p_value < 0.05 and value < 0:
                colors.append("#B84A4A")
            else:
                colors.append("#C7CDD4")

        ax.barh(y, values, color=colors, edgecolor="#5B626B", linewidth=0.6)
        ax.axvline(0, color="#222222", linewidth=0.9)
        ax.set_title(spec["label"])
        ax.set_xlabel(f"Post-AI minus pre-AI ({spec['unit']})")
        ax.set_yticks(y)
        ax.set_yticklabels(FIELD_ORDER)
        ax.grid(axis="x", color="#E8EAED", linewidth=0.8)
        ax.set_axisbelow(True)

        max_abs = max(np.max(np.abs(values)), 1e-9)
        pad = max_abs * 0.18
        ax.set_xlim(min(values.min() - pad, -pad), max(values.max() + pad, pad))

        x_min, x_max = ax.get_xlim()
        offset = (x_max - x_min) * 0.012
        for yi, value, p_value in zip(y, values, p_values):
            ha = "left" if value >= 0 else "right"
            x = value + offset if value >= 0 else value - offset
            label = f"{format_p(p_value)}{stars(p_value)}"
            ax.text(x, yi, label, va="center", ha=ha, fontsize=8.8, color="#202124")

    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.11, right=0.985, top=0.80, bottom=0.20, wspace=0.08)
    fig.suptitle(
        f"Core metrics by field: {group_name.capitalize()} changes",
        fontsize=14,
        fontweight="bold",
        y=0.94,
    )
    fig.text(
        0.01,
        0.04,
        "Bars show post-AI minus pre-AI seed-level means; p-values are Welch tests. * p<.05, ** p<.01, *** p<.001.",
        fontsize=8.5,
        color="#4D535A",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"core_metrics_{group_name}_post_minus_pre.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = list(METRICS)
    value_matrix = np.zeros((len(FIELD_ORDER), len(metrics)))
    labels = [["" for _ in metrics] for _ in FIELD_ORDER]

    indexed = df.set_index(["dataset", "metric"])
    for i, field in enumerate(FIELD_ORDER):
        for j, metric in enumerate(metrics):
            row = indexed.loc[(field, metric)]
            value = float(row["post_minus_pre"]) * METRICS[metric]["scale"]
            p_value = float(row["welch_p"])
            value_matrix[i, j] = value
            if abs(value) >= 10:
                value_text = f"{value:.1f}"
            elif abs(value) >= 1:
                value_text = f"{value:.2f}"
            else:
                value_text = f"{value:.3f}"
            labels[i][j] = f"{value_text}\n{format_p(p_value)}{stars(p_value)}"

    color_matrix = value_matrix.copy()
    for j in range(color_matrix.shape[1]):
        col_max = float(np.nanmax(np.abs(color_matrix[:, j])))
        if col_max > 0:
            color_matrix[:, j] = color_matrix[:, j] / col_max

    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)

    fig, ax = plt.subplots(figsize=(13.8, 6.4), constrained_layout=False)
    im = ax.imshow(color_matrix, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRICS[m]["short"] for m in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(FIELD_ORDER)))
    ax.set_yticklabels(FIELD_ORDER)
    ax.set_title("Post-AI minus pre-AI changes across fields", fontsize=14, fontweight="bold")

    for i in range(len(FIELD_ORDER)):
        for j in range(len(metrics)):
            color = "white" if abs(color_matrix[i, j]) > 0.58 else "#202124"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8.4, color=color)

    fig.subplots_adjust(left=0.13, right=0.90, top=0.86, bottom=0.38)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Within-metric normalized direction")
    fig.text(
        0.13,
        0.05,
        "Cell values are true post-pre differences: share metrics in percentage points; other metrics in native units. Colors are normalized within each metric. P-values are Welch tests.",
        fontsize=8.5,
        color="#4D535A",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"core_metrics_all_fields_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/field_scan_200_summary_comparisons.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/field_scan_200_figures_python"),
    )
    args = parser.parse_args()

    df = load_summary(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for group_name, metrics in GROUPS.items():
        plot_metric_group(df, group_name, metrics, args.out_dir)
    plot_heatmap(df, args.out_dir)

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
