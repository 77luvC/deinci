"""Plot core field-scan metrics with post-pre differences and p-values.

The script expects a field-scan summary CSV produced by the analysis pipeline.
Each row should describe one `(field_label, metric)` comparison and include a
post-minus-pre estimate plus a Welch-test p-value. The plotting code preserves
the raw comparison values in labels while using grouped plots and a normalized
heatmap to make cross-field patterns easier to scan.
"""

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

# Plotting assumes these field labels match the `field_label` values in the summary
# CSV. Fields outside this list are ignored by the figure builders, while missing
# listed fields will surface as lookup errors when the relevant metric is drawn.

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

# Each metric spec defines the public label, compact heatmap label, display
# scaling, and unit. Share metrics are stored as proportions in the CSV and are
# converted to percentage points for figures; log and entropy metrics remain in
# their native units.

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

# Groups determine which bar-chart panels are exported. A metric listed here is
# expected to also exist in METRICS and in the input summary; otherwise plotting
# will fail early enough to reveal a stale group definition.


def format_p(p_value: float) -> str:
    """Format a p-value for compact figure annotations.

    Assumes `p_value` is numeric or NA-like. NA values are rendered explicitly so
    missing statistical tests are visible in the figure instead of being hidden.
    Very small p-values use a threshold label to avoid visually noisy decimals.
    """
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<.001"
    if p_value < 0.1:
        return f"p={p_value:.3f}".replace("0.", ".")
    return f"p={p_value:.2f}".replace("0.", ".")


def stars(p_value: float) -> str:
    """Return conventional significance stars for a p-value.

    Assumes smaller values indicate stronger evidence under the Welch test used
    upstream. NA values receive no stars because there is no statistical result
    to encode. The function intentionally stops at p<.05 for visible marking.
    """
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
    """Load and order the summary table used by all figures.

    Expected input columns include `field_label`, `metric`, `mean_diff_a_minus_b`,
    and `welch_p_value`. This function verifies that every configured metric appears at
    least once, but it relies on downstream lookups to expose missing field-level
    rows or duplicate `(field_label, metric)` combinations. The field_label column is
    made categorical so all figures follow FIELD_ORDER rather than CSV order.
    """
    # Read the pipeline output as a flat comparison table. The script assumes the
    # CSV already contains one row per field/metric comparison and does not
    # recompute statistics here.
    df = pd.read_csv(path)

    # Apply the canonical visual order before sorting. Unknown datasets become
    # NaN categories and naturally sort after known fields; the plotting steps
    # later select FIELD_ORDER explicitly, so extra rows are not displayed.
    df["field_label"] = pd.Categorical(df["field_label"], categories=FIELD_ORDER, ordered=True)
    df = df.sort_values(["field_label", "metric"]).reset_index(drop=True)

    # Fail fast when a configured metric is absent from the whole file. Edge
    # cases such as a metric present for only some fields are left to the exact
    # `.loc` calls in the plotting functions, which produce a missing-key error.
    missing = sorted(set(METRICS) - set(df["metric"]))
    if missing:
        raise ValueError(f"Missing metrics: {missing}")
    return df


def plot_metric_group(df: pd.DataFrame, group_name: str, metrics: list[str], out_dir: Path) -> None:
    """Write bar charts for one conceptual group of metrics.

    `df` is expected to contain rows for each metric in `metrics` and for every
    field in FIELD_ORDER. Values are interpreted as post-AI minus pre-AI
    differences; metric specs determine whether those differences are displayed
    in native units or percentage points. The function writes both PNG and PDF
    outputs and closes the figure to avoid leaking Matplotlib state.
    """
    # Set local publication-style defaults for these figures. Updating rcParams
    # here keeps the script self-contained, but callers should expect global
    # Matplotlib settings in this process to be affected after the first plot.
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

    # Create one horizontal bar panel per metric. A single-metric group returns a
    # lone Axes object from Matplotlib, so normalize it to a list for the loop.
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.7 * len(metrics), 5.25),
        sharey=True,
        constrained_layout=False,
    )
    if len(metrics) == 1:
        axes = [axes]

    # Use fixed row positions so each panel aligns field names consistently.
    # The code assumes FIELD_ORDER contains at least one field.
    y = np.arange(len(FIELD_ORDER))
    for ax, metric in zip(axes, metrics):
        spec = METRICS[metric]

        # Select the exact metric and reindex by FIELD_ORDER. Missing fields or
        # duplicate field/metric rows are edge cases that should fail loudly here
        # rather than produce a misleading partial chart.
        sub = (
            df[df["metric"] == metric]
            .set_index("field_label")
            .loc[FIELD_ORDER]
            .reset_index()
        )
        values = sub["mean_diff_a_minus_b"].to_numpy() * spec["scale"]
        p_values = sub["welch_p_value"].to_numpy()

        # Encode statistically significant movement in blue/red by direction.
        # Non-significant or NA p-values remain neutral gray; comparisons with
        # NA values are expected to have been handled before plotting.
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

        # Build symmetric-enough padding around the observed range. The 1e-9
        # floor avoids a zero-width axis when all displayed differences are 0;
        # non-finite values are not supported and should be validated upstream.
        max_abs = max(np.max(np.abs(values)), 1e-9)
        pad = max_abs * 0.18
        ax.set_xlim(min(values.min() - pad, -pad), max(values.max() + pad, pad))

        # Place p-value labels just outside the bar end so positive and negative
        # effects remain readable. Labels may extend into the margin for very
        # small values, so the x-limit padding above is part of this assumption.
        x_min, x_max = ax.get_xlim()
        offset = (x_max - x_min) * 0.012
        for yi, value, p_value in zip(y, values, p_values):
            ha = "left" if value >= 0 else "right"
            x = value + offset if value >= 0 else value - offset
            label = f"{format_p(p_value)}{stars(p_value)}"
            ax.text(x, yi, label, va="center", ha=ha, fontsize=8.8, color="#202124")

    # Final shared annotations explain statistical encoding once per exported
    # figure. The y-axis is inverted so FIELD_ORDER reads top-to-bottom.
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

    # Save reproducible figure formats for manuscript/export workflows. The
    # caller must ensure `out_dir` exists before invoking this function.
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"core_metrics_{group_name}_post_minus_pre.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a cross-field heatmap of all configured core metrics.

    The heatmap displays true scaled post-minus-pre values as text, while colors
    are normalized within each metric column. This keeps direction and relative
    strength visible across metrics with incompatible units. The input must have
    exactly one usable row for every `(field, metric)` pair in FIELD_ORDER and
    METRICS.
    """
    metrics = list(METRICS)

    # Preallocate numeric values and text labels to keep table geometry stable.
    # Empty labels are overwritten below; any missing row should fail at lookup
    # time rather than silently leave a blank cell.
    value_matrix = np.zeros((len(FIELD_ORDER), len(metrics)))
    labels = [["" for _ in metrics] for _ in FIELD_ORDER]

    indexed = df.set_index(["field_label", "metric"])
    for i, field in enumerate(FIELD_ORDER):
        for j, metric in enumerate(metrics):
            # Extract the raw comparison and p-value for one cell. The value is
            # scaled for display according to METRICS, but the underlying sign is
            # preserved for both label text and heatmap coloring.
            row = indexed.loc[(field, metric)]
            value = float(row["mean_diff_a_minus_b"]) * METRICS[metric]["scale"]
            p_value = float(row["welch_p_value"])
            value_matrix[i, j] = value

            # Use more precision for small absolute changes so near-zero effects
            # do not all collapse visually to 0.0. Very large values are kept
            # short to fit inside heatmap cells.
            if abs(value) >= 10:
                value_text = f"{value:.1f}"
            elif abs(value) >= 1:
                value_text = f"{value:.2f}"
            else:
                value_text = f"{value:.3f}"
            labels[i][j] = f"{value_text}\n{format_p(p_value)}{stars(p_value)}"

    # Normalize each metric column independently for color only. This assumes
    # the viewer compares color direction/intensity within a metric, while the
    # printed cell labels carry the actual units for cross-metric interpretation.
    color_matrix = value_matrix.copy()
    for j in range(color_matrix.shape[1]):
        col_max = float(np.nanmax(np.abs(color_matrix[:, j])))
        if col_max > 0:
            color_matrix[:, j] = color_matrix[:, j] / col_max

    norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)

    # Draw the heatmap with a diverging palette centered on zero. Because colors
    # are already normalized to [-1, 1] within each metric, the colorbar names the
    # direction scale rather than claiming shared measurement units.
    fig, ax = plt.subplots(figsize=(13.8, 6.4), constrained_layout=False)
    im = ax.imshow(color_matrix, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRICS[m]["short"] for m in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(FIELD_ORDER)))
    ax.set_yticklabels(FIELD_ORDER)
    ax.set_title("Post-AI minus pre-AI changes across fields", fontsize=14, fontweight="bold")

    # Switch annotation color on high-intensity cells for contrast. The threshold
    # is visual, not statistical; significance is still communicated by p-values
    # and stars inside each label.
    for i in range(len(FIELD_ORDER)):
        for j in range(len(metrics)):
            color = "white" if abs(color_matrix[i, j]) > 0.58 else "#202124"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8.4, color=color)

    # Layout leaves enough bottom margin for rotated metric labels and a shared
    # footnote that explains the mixed-unit label convention.
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

    # Save both raster and vector-style outputs for downstream use. The caller
    # must create `out_dir`; this function only writes files inside it.
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"core_metrics_all_fields_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and generate every configured core-metric figure.

    Defaults assume the command is run from the `citation_relevance_poc` project
    directory, because the default paths are relative. Supplying absolute paths
    or running from the project root avoids ambiguity in notebooks, CI jobs, or
    parent workspaces.
    """
    # Keep the CLI minimal: the script is a plotting endpoint, while upstream
    # pipelines are responsible for producing the summary CSV with the expected
    # schema and time-window definitions.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/field_scan_200_summary_comparisons.csv"),
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=Path("results/field_scan_200_figures_python"),
    )
    parser.add_argument(
        "--field-order",
        default=None,
        help="Comma-separated field labels. Defaults to the manuscript field order.",
    )
    args = parser.parse_args()

    global FIELD_ORDER
    if args.field_order:
        FIELD_ORDER = [field.strip() for field in args.field_order.split(",") if field.strip()]
        if not FIELD_ORDER:
            raise ValueError("--field-order was provided but no usable field labels were parsed.")

    # Load once and reuse the same validated/ordered frame for all figures so
    # grouped charts and the all-metric heatmap are guaranteed to be consistent.
    df = load_summary(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Emit one file pair per metric group, then a compact all-field/all-metric
    # heatmap for overview comparison.
    for group_name, metrics in GROUPS.items():
        plot_metric_group(df, group_name, metrics, args.out_dir)
    plot_heatmap(df, args.out_dir)

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
