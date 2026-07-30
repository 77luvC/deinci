from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from src.pipelines.run_democratization_analysis import BASE_METRICS, RELEVANCE_METRICS, benjamini_hochberg
from src.utils.io import ensure_dir, write_table


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected input as label=path, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Input label cannot be empty: {value}")
    return label, Path(path)


def read_seed_metrics(inputs: list[str]) -> pd.DataFrame:
    frames = []
    for item in inputs:
        label, path = parse_input(item)
        frame = pd.read_csv(path)
        frame["field_label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ols_interaction_pvalue(df: pd.DataFrame, metric: str, treatment_field_label: str, post_period: str, base_period: str) -> dict:
    subset = df[df["query_ai_group"].isin([base_period, post_period])].copy()
    subset = subset[["field_label", "query_ai_group", metric]].dropna()
    if subset["field_label"].nunique() < 2 or len(subset) < 8:
        return {"interaction_coef": np.nan, "interaction_t": np.nan, "interaction_p_value": np.nan}

    y = subset[metric].astype(float).to_numpy()
    treated = (subset["field_label"] == treatment_field_label).astype(float).to_numpy()
    post = (subset["query_ai_group"] == post_period).astype(float).to_numpy()
    x = np.column_stack([np.ones(len(subset)), treated, post, treated * post])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    df_resid = len(y) - x.shape[1]
    if df_resid <= 0:
        return {"interaction_coef": float(beta[3]), "interaction_t": np.nan, "interaction_p_value": np.nan}

    sigma2 = float((residuals @ residuals) / df_resid)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(cov[3, 3]))
    if se <= 0 or not np.isfinite(se):
        return {"interaction_coef": float(beta[3]), "interaction_t": np.nan, "interaction_p_value": np.nan}
    t_stat = float(beta[3] / se)
    p_value = float(2 * t.sf(abs(t_stat), df_resid))
    return {"interaction_coef": float(beta[3]), "interaction_t": t_stat, "interaction_p_value": p_value}


def did_comparisons(
    seed_metrics: pd.DataFrame,
    *,
    treatment_field_label: str,
    control_field_labels: list[str],
) -> pd.DataFrame:
    # `seed_metrics["field_label"]` holds the `label=` half of each --input
    # (e.g. "CS"), not a data directory name like "openalex_cs_1000" -- so
    # treatment_field_label/control_field_labels must match one of those
    # labels. Passing a directory-style value here used to fail silently: the
    # `.isin([...])` filter below would match zero rows and every DID
    # estimate would come out NaN with no error raised.
    available = sorted(seed_metrics["field_label"].unique())
    unknown = [
        value for value in [treatment_field_label, *control_field_labels] if value not in available
    ]
    if unknown:
        raise ValueError(
            f"treatment_field_label/control_field_labels must match an --input label, "
            f"got unknown value(s) {unknown!r}; available labels: {available!r}"
        )

    contrasts = [("post_ai", "pre_ai")]
    metrics = [metric for metric in BASE_METRICS + RELEVANCE_METRICS if metric in seed_metrics.columns]
    rows = []
    for control_field_label in control_field_labels:
        pair = seed_metrics[seed_metrics["field_label"].isin([treatment_field_label, control_field_label])].copy()
        for metric in metrics:
            for post_period, base_period in contrasts:
                means = (
                    pair[pair["query_ai_group"].isin([base_period, post_period])]
                    .groupby(["field_label", "query_ai_group"])[metric]
                    .mean()
                )
                treatment_post = means.get((treatment_field_label, post_period), np.nan)
                treatment_base = means.get((treatment_field_label, base_period), np.nan)
                control_post = means.get((control_field_label, post_period), np.nan)
                control_base = means.get((control_field_label, base_period), np.nan)
                treatment_change = treatment_post - treatment_base
                control_change = control_post - control_base
                did = treatment_change - control_change
                ols = ols_interaction_pvalue(pair, metric, treatment_field_label, post_period, base_period)
                rows.append(
                    {
                        "metric": metric,
                        "treatment_field_label": treatment_field_label,
                        "control_field_label": control_field_label,
                        "post_period": post_period,
                        "base_period": base_period,
                        "treatment_post_mean": treatment_post,
                        "treatment_base_mean": treatment_base,
                        "control_post_mean": control_post,
                        "control_base_mean": control_base,
                        "treatment_change": treatment_change,
                        "control_change": control_change,
                        "did_estimate": did,
                        "interaction_coef": ols["interaction_coef"],
                        "interaction_t": ols["interaction_t"],
                        "interaction_p_value": ols["interaction_p_value"],
                    }
                )
    out = pd.DataFrame(rows)
    out["interaction_q_value_bh"] = benjamini_hochberg(out["interaction_p_value"])
    return out


def period_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [metric for metric in BASE_METRICS + RELEVANCE_METRICS if metric in seed_metrics.columns]
    for (field_label, period), group in seed_metrics.groupby(["field_label", "query_ai_group"]):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "field_label": field_label,
                    "query_ai_group": period,
                    "metric": metric,
                    "n_seed_papers": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="label=seed_metrics_csv")
    parser.add_argument("--treatment-field-label", required=True)
    parser.add_argument("--control-field-label", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    seed_metrics = read_seed_metrics(args.input)
    write_table(seed_metrics, output_dir / "combined_seed_democratization_metrics.csv")
    write_table(period_summary(seed_metrics), output_dir / "combined_period_summary.csv")
    write_table(
        did_comparisons(
            seed_metrics,
            treatment_field_label=args.treatment_field_label,
            control_field_labels=args.control_field_label,
        ),
        output_dir / "did_democratization_comparisons.csv",
    )
    print(f"Saved cross-field DID outputs in {output_dir}")


if __name__ == "__main__":
    main()
