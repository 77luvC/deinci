from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FIELD_REQUIRED_KEYS = {"dataset", "label"}
FIELD_STEPS = {
    "build_raw",
    "load_pairs",
    "pair_metadata",
    "historical_metadata",
    "relevance_scoring",
    "historical_scores",
    "cited_distribution",
    "democratization",
    "temporal_relevance",
}


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML run-parameter document.

    Date-stamped YAML files are the source of truth for reproducible runs. This
    function keeps validation light and focused on fields needed by the runner.
    """
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config.get("fields"), list) or not config["fields"]:
        raise ValueError("Config must define a non-empty `fields` list.")
    for field in config["fields"]:
        missing = FIELD_REQUIRED_KEYS - set(field)
        if missing:
            raise ValueError(f"Field entry is missing required keys {sorted(missing)}: {field}")
        if not field.get("field_id") and not field.get("subfield_ids"):
            raise ValueError(f"Field entry must define `field_id` or `subfield_ids`: {field}")
    return config


def resolve_project_dir(config_path: Path, config: dict[str, Any]) -> Path:
    """Resolve the project directory for subprocess calls.

    Relative `run.project_dir` values are interpreted relative to the config
    file, which makes SCC jobs robust when launched outside the repository.
    """
    run_config = config.get("run", {})
    project_dir = Path(run_config.get("project_dir", ".."))
    if not project_dir.is_absolute():
        project_dir = config_path.parent / project_dir
    return project_dir.resolve()


def shell_join(parts: list[str]) -> str:
    """Render a command for logging without changing how it is executed."""
    return " ".join(shlex.quote(str(part)) for part in parts)


class RunLog:
    """Markdown record of one pipeline invocation: parameters used plus per-stage timing.

    Rewritten to disk after every stage (not just at the end) so a run that
    fails or is killed partway still leaves a readable log behind. Each
    invocation gets its own timestamped file, since `run_name` in the config
    is normally reused across resumed/partial reruns of the same experiment.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_name: str,
        config: dict[str, Any],
        config_path: Path,
        command: list[str],
        field_labels: list[str],
        only: set[str] | None,
        skip: set[str],
        start_time: datetime,
    ) -> None:
        self.path = path
        self.run_name = run_name
        self.config = config
        self.config_path = config_path
        self.command_str = shell_join(command)
        self.field_labels = field_labels
        self.only = only
        self.skip = skip
        self.start_time = start_time
        self.end_time: datetime | None = None
        self.rows: list[dict[str, Any]] = []
        self._write()

    def record(self, *, dataset: str | None, stage: str, duration_seconds: float, status: str) -> None:
        self.rows.append(
            {
                "dataset": dataset or "(run-level)",
                "stage": stage,
                "duration_seconds": duration_seconds,
                "status": status,
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        self._write()

    def finalize(self) -> None:
        """Stamp the end time, write the final log, and print a console summary."""
        self.end_time = datetime.now(timezone.utc)
        self._write()

        if self.rows:
            per_stage: dict[str, dict[str, float]] = {}
            for row in self.rows:
                entry = per_stage.setdefault(row["stage"], {"n": 0.0, "total": 0.0})
                entry["n"] += 1
                entry["total"] += row["duration_seconds"]
            print("\n=== Stage timing summary ===", flush=True)
            for stage, entry in sorted(per_stage.items(), key=lambda kv: kv[1]["total"], reverse=True):
                mean = entry["total"] / entry["n"]
                print(
                    f"  {stage:<40s} n={int(entry['n']):>3d}  total={entry['total']:>8.1f}s  mean={mean:>7.1f}s",
                    flush=True,
                )
            total = sum(row["duration_seconds"] for row in self.rows)
            print(f"\nTotal recorded wall-clock time: {total:.1f}s", flush=True)
        print(f"Run log: {self.path}", flush=True)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self._render_markdown(), encoding="utf-8")

    def _render_markdown(self) -> str:
        lines: list[str] = [f"# Run Log: {self.run_name}", ""]
        lines.append(f"- Config file: `{self.config_path}`")
        lines.append(f"- Command: `{self.command_str}`")
        lines.append(f"- Started (UTC): {self.start_time.isoformat(timespec='seconds')}")
        if self.end_time is not None:
            elapsed = (self.end_time - self.start_time).total_seconds()
            lines.append(f"- Finished (UTC): {self.end_time.isoformat(timespec='seconds')}")
            lines.append(f"- Wall-clock elapsed: {elapsed:.1f}s")
        else:
            lines.append("- Finished (UTC): _(run in progress or interrupted)_")
        lines.append(f"- Fields run: {', '.join(self.field_labels) if self.field_labels else '(none)'}")
        lines.append(f"- `--only`: {', '.join(sorted(self.only)) if self.only else '(all steps)'}")
        lines.append(f"- `--skip`: {', '.join(sorted(self.skip)) if self.skip else '(none)'}")
        lines.append("")

        lines.append("## Parameters")
        lines.append("")
        lines.append("```yaml")
        lines.append(yaml.safe_dump(self.config, sort_keys=False).rstrip())
        lines.append("```")
        lines.append("")

        lines.append("## Stage Timings")
        lines.append("")
        if self.rows:
            lines.append("| Dataset | Stage | Duration (s) | Status | Recorded (UTC) |")
            lines.append("|---|---|---|---|---|")
            for row in self.rows:
                lines.append(
                    f"| {row['dataset']} | {row['stage']} | {row['duration_seconds']:.1f} "
                    f"| {row['status']} | {row['recorded_at']} |"
                )
        else:
            lines.append("_No stages recorded yet._")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        if self.rows:
            per_stage: dict[str, dict[str, float]] = {}
            for row in self.rows:
                entry = per_stage.setdefault(row["stage"], {"n": 0.0, "total": 0.0})
                entry["n"] += 1
                entry["total"] += row["duration_seconds"]
            lines.append("| Stage | Runs | Total (s) | Mean (s) |")
            lines.append("|---|---|---|---|")
            for stage, entry in sorted(per_stage.items(), key=lambda kv: kv[1]["total"], reverse=True):
                mean = entry["total"] / entry["n"]
                lines.append(f"| {stage} | {int(entry['n'])} | {entry['total']:.1f} | {mean:.1f} |")
            total = sum(row["duration_seconds"] for row in self.rows)
            lines.append("")
            lines.append(f"**Total recorded wall-clock time:** {total:.1f}s")
        else:
            lines.append("_No stages recorded yet._")
        lines.append("")
        return "\n".join(lines)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    env: dict[str, str],
    run_log: RunLog | None = None,
    stage: str | None = None,
    dataset: str | None = None,
) -> None:
    """Run one pipeline command or print it in dry-run mode.

    When `run_log` is provided, wall-clock duration for this invocation is
    recorded under `stage`/`dataset` even if the command later raises, so a
    failed run still reports how long each completed stage took.
    """
    print(f"\n$ {shell_join(cmd)}", flush=True)
    if dry_run:
        return
    label = stage or Path(cmd[1]).stem
    start = time.time()
    status = "ok"
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.time() - start
        print(f"  [{label}] {dataset or ''} took {duration:.1f}s ({status})", flush=True)
        if run_log is not None:
            run_log.record(dataset=dataset, stage=label, duration_seconds=duration, status=status)


def step_is_enabled(step: str, config: dict[str, Any], only: set[str] | None, skip: set[str]) -> bool:
    """Return whether a named step should run for this invocation."""
    if only is not None and step not in only:
        return False
    if step in skip:
        return False
    return bool(config.get("steps", {}).get(step, True))


def make_run_id(run_name: str, timestamp: datetime) -> str:
    """Build the unique per-invocation run identifier used as a results/ subdirectory name."""
    return f"{run_name}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def field_paths(project_dir: Path, run_dir: Path, dataset: str) -> dict[str, Path]:
    """Build canonical paths for one dataset within one run.

    `data/raw/<dataset>` and `data/processed/<dataset>` (including the SciNCL
    embedding cache and the historical-citation-count cache) are intentionally
    *not* under `run_dir`: they are expensive-to-build inputs/caches meant to be
    reused across runs, not per-run outputs. Everything under `results/` is
    per-run: it lives under `run_dir/<dataset>/` so that no run can overwrite
    another run's figures, tables, or reports.
    """
    field_dir = run_dir / dataset
    tables_dir = field_dir / "tables"
    return {
        "raw": project_dir / "data" / "raw" / dataset,
        "processed": project_dir / "data" / "processed" / dataset,
        "papers_raw": project_dir / "data" / "raw" / dataset / "papers.csv",
        "pairs_raw": project_dir / "data" / "raw" / dataset / "pairs.csv",
        "papers": project_dir / "data" / "processed" / dataset / "papers.parquet",
        "pairs": project_dir / "data" / "processed" / dataset / "pairs.parquet",
        "embeddings": project_dir / "data" / "processed" / dataset / "scincl_embeddings",
        "historical_cache": project_dir / "data" / "processed" / dataset / "historical_citation_counts_yearly_cache.csv",
        "field_dir": field_dir,
        "tables_dir": tables_dir,
        "figures_dir": field_dir / "figures",
        "reports_dir": field_dir / "reports",
        "metadata": tables_dir / "pair_metadata.parquet",
        "metadata_historical": tables_dir / "pair_metadata_historical.parquet",
        "scores": tables_dir / "pair_scores.parquet",
        "scores_historical": tables_dir / "pair_scores_historical.parquet",
        "cited_distribution": field_dir,
        "democratization": field_dir,
        "temporal": field_dir,
    }


def add_optional(cmd: list[str], flag: str, value: Any) -> None:
    """Append a CLI flag only when a config value is present."""
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def run_field_pipeline(
    field: dict[str, Any],
    *,
    config: dict[str, Any],
    project_dir: Path,
    run_dir: Path,
    python_bin: str,
    dry_run: bool,
    only: set[str] | None,
    skip: set[str],
    env: dict[str, str],
    force_recompute_cli: bool = False,
    run_log: RunLog | None = None,
) -> None:
    """Run all enabled steps for a single configured field."""
    dataset = str(field["dataset"])
    paths = field_paths(project_dir, run_dir, dataset)
    sampling = config.get("sampling", {})
    openalex = config.get("openalex", {})
    historical = config.get("historical_citations", {})
    core = config.get("core_metrics", {})
    relevance = config.get("relevance_scoring", {})

    print(f"\n=== Field: {field['label']} ({dataset}) -> {paths['field_dir']} ===", flush=True)
    if not dry_run:
        # Pre-create the full figures/reports/tables shape up front so it's
        # visible even before any stage that populates it has run — nothing
        # currently writes a narrative report, but the folder is reserved for
        # one (e.g. a future per-field summary write-up).
        paths["tables_dir"].mkdir(parents=True, exist_ok=True)
        paths["figures_dir"].mkdir(parents=True, exist_ok=True)
        paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    if step_is_enabled("build_raw", config, only, skip):
        cmd = [
            python_bin,
            "-m",
            "src.data_loaders.openalex_expanded_builder",
            "--output-dir",
            str(paths["raw"]),
            "--language",
            str(field.get("language", sampling.get("language", "en"))),
            "--source-dataset",
            dataset,
            "--seeds-per-period",
            str(field.get("seeds_per_period", sampling.get("seeds_per_period", 200))),
            "--candidate-sample-size",
            str(field.get("candidate_sample_size", sampling.get("candidate_sample_size", 3000))),
            "--min-references",
            str(field.get("min_references", sampling.get("min_references", 8))),
            "--seed",
            str(field.get("random_seed", sampling.get("random_seed", 42))),
            "--sleep-seconds",
            str(openalex.get("sleep_seconds", 0.15)),
        ]
        add_optional(cmd, "--field-id", field.get("field_id"))
        add_optional(cmd, "--subfield-ids", field.get("subfield_ids"))
        add_optional(cmd, "--mailto", openalex.get("mailto"))
        run_command(
            cmd,
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="openalex_expanded_builder",
            dataset=dataset,
        )

    if step_is_enabled("load_pairs", config, only, skip):
        run_command(
            [
                python_bin,
                "-m",
                "src.data_loaders.generic_pairs_loader",
                "--papers-csv",
                str(paths["papers_raw"]),
                "--pairs-csv",
                str(paths["pairs_raw"]),
                "--output-dir",
                str(paths["processed"]),
            ],
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="generic_pairs_loader",
            dataset=dataset,
        )

    relevance_enabled = bool(relevance.get("enabled", True))
    model_name = str(relevance.get("model", "malteos/scincl"))
    force_recompute = bool(force_recompute_cli or relevance.get("force_recompute", False))

    # Metadata-only pair files always carry an empty `scincl_cosine` column
    # (see run_pair_metadata.py). They are only useful as the analysis input
    # when relevance scoring is off or explicitly not preferred; otherwise
    # `analysis_scores` below resolves to the scored file and these metadata
    # files would just be unused, redundant output. An explicit `--only` still
    # forces them to run for ad hoc metadata-only inspection.
    metadata_fallback_needed = (not relevance_enabled) or not core.get("use_relevance_scores_when_available", True)
    run_pair_metadata_step = step_is_enabled("pair_metadata", config, only, skip) and (
        metadata_fallback_needed or (only is not None and "pair_metadata" in only)
    )
    run_historical_metadata_step = step_is_enabled("historical_metadata", config, only, skip) and (
        metadata_fallback_needed or (only is not None and "historical_metadata" in only)
    )

    if run_pair_metadata_step:
        run_command(
            [
                python_bin,
                "-m",
                "src.pipelines.run_pair_metadata",
                "--papers",
                str(paths["papers"]),
                "--pairs",
                str(paths["pairs"]),
                "--output",
                str(paths["metadata"]),
            ],
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="run_pair_metadata",
            dataset=dataset,
        )
    elif step_is_enabled("pair_metadata", config, only, skip):
        print(
            f"  [run_pair_metadata] {dataset} skipped: relevance scores will be used for analysis "
            "(pass --only pair_metadata to force).",
            flush=True,
        )

    if run_historical_metadata_step:
        run_historical(
            python_bin,
            scores=paths["metadata"],
            papers=paths["papers"],
            output=paths["metadata_historical"],
            cache=paths["historical_cache"],
            config=config,
            project_dir=project_dir,
            dry_run=dry_run,
            env=env,
            dataset=dataset,
            stage="run_historical_citation_counts:metadata",
            run_log=run_log,
        )
    elif step_is_enabled("historical_metadata", config, only, skip):
        print(
            f"  [run_historical_citation_counts:metadata] {dataset} skipped: relevance scores will be used "
            "for analysis (pass --only historical_metadata to force).",
            flush=True,
        )

    if relevance_enabled and step_is_enabled("relevance_scoring", config, only, skip):
        embed_cmd = [
            python_bin,
            "-m",
            "src.pipelines.run_embed",
            "--papers",
            str(paths["papers"]),
            "--output",
            str(paths["embeddings"]),
            "--model",
            model_name,
        ]
        if force_recompute:
            embed_cmd.append("--force-recompute")
        run_command(
            embed_cmd,
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="run_embed",
            dataset=dataset,
        )
        scoring_cmd = [
            python_bin,
            "-m",
            "src.pipelines.run_pair_scoring",
            "--papers",
            str(paths["papers"]),
            "--pairs",
            str(paths["pairs"]),
            "--embeddings",
            str(paths["embeddings"]),
            "--output",
            str(paths["scores"]),
            "--model",
            model_name,
        ]
        if force_recompute:
            scoring_cmd.append("--force-recompute")
        run_command(
            scoring_cmd,
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="run_pair_scoring",
            dataset=dataset,
        )

    if relevance_enabled and step_is_enabled("historical_scores", config, only, skip):
        run_historical(
            python_bin,
            scores=paths["scores"],
            papers=paths["papers"],
            output=paths["scores_historical"],
            cache=paths["historical_cache"],
            config=config,
            project_dir=project_dir,
            dry_run=dry_run,
            env=env,
            dataset=dataset,
            stage="run_historical_citation_counts:scores",
            run_log=run_log,
        )

    analysis_scores = (
        paths["scores_historical"]
        if relevance_enabled and core.get("use_relevance_scores_when_available", True)
        else paths["metadata_historical"]
    )

    if step_is_enabled("cited_distribution", config, only, skip):
        run_command(
            [
                python_bin,
                "-m",
                "src.pipelines.run_cited_distribution_analysis",
                "--scores",
                str(analysis_scores),
                "--papers",
                str(paths["papers"]),
                "--output-dir",
                str(paths["cited_distribution"]),
            ],
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="run_cited_distribution_analysis",
            dataset=dataset,
        )

    if step_is_enabled("democratization", config, only, skip):
        run_command(
            [
                python_bin,
                "-m",
                "src.pipelines.run_democratization_analysis",
                "--scores",
                str(analysis_scores),
                "--papers",
                str(paths["papers"]),
                "--output-dir",
                str(paths["democratization"]),
                "--low-citation-quantile",
                str(core.get("low_citation_quantile", 0.25)),
                "--relevance-threshold",
                str(core.get("relevance_threshold", 0.80)),
            ],
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="run_democratization_analysis",
            dataset=dataset,
        )

    if relevance_enabled and step_is_enabled("temporal_relevance", config, only, skip):
        run_command(
            [
                python_bin,
                "-m",
                "src.pipelines.run_temporal_analysis",
                "--scores",
                str(paths["scores_historical"]),
                "--output-dir",
                str(paths["temporal"]),
            ],
            cwd=project_dir,
            dry_run=dry_run,
            env=env,
            run_log=run_log,
            stage="run_temporal_analysis",
            dataset=dataset,
        )


def run_historical(
    python_bin: str,
    *,
    scores: Path,
    papers: Path,
    output: Path,
    cache: Path,
    config: dict[str, Any],
    project_dir: Path,
    dry_run: bool,
    env: dict[str, str],
    dataset: str | None = None,
    stage: str = "run_historical_citation_counts",
    run_log: RunLog | None = None,
) -> None:
    """Run historical citation enrichment using config-level OpenAlex settings."""
    openalex = config.get("openalex", {})
    historical = config.get("historical_citations", {})
    cmd = [
        python_bin,
        "-m",
        "src.pipelines.run_historical_citation_counts",
        "--scores",
        str(scores),
        "--papers",
        str(papers),
        "--output",
        str(output),
        "--cache",
        str(cache),
        "--method",
        str(historical.get("method", "yearly")),
        "--sleep-seconds",
        str(openalex.get("sleep_seconds", 0.15)),
        "--max-retries",
        str(historical.get("max_retries", 3)),
        "--checkpoint-every",
        str(historical.get("checkpoint_every", 100)),
    ]
    add_optional(cmd, "--mailto", openalex.get("mailto"))
    run_command(
        cmd,
        cwd=project_dir,
        dry_run=dry_run,
        env=env,
        run_log=run_log,
        stage=stage,
        dataset=dataset,
    )


def aggregate_core_outputs(
    config: dict[str, Any], fields: list[dict[str, Any]], project_dir: Path, run_dir: Path, dry_run: bool
) -> None:
    """Create field-scan comparison tables consumed by the plotting script."""
    comparison_rows = []
    cited_rows = []
    for field in fields:
        dataset = str(field["dataset"])
        label = str(field["label"])
        paths = field_paths(project_dir, run_dir, dataset)

        comparison_path = paths["democratization"] / "tables" / "period_democratization_comparisons.csv"
        cited_path = paths["cited_distribution"] / "tables" / "cited_distribution_summary.csv"
        if dry_run:
            print(f"Would aggregate {comparison_path}")
            print(f"Would aggregate {cited_path}")
            continue

        comparisons = pd.read_csv(comparison_path)
        comparisons.insert(0, "field_label", label)
        comparisons.insert(1, "source_dataset", dataset)
        comparison_rows.append(comparisons)

        cited = pd.read_csv(cited_path)
        cited.insert(0, "field_label", label)
        cited.insert(1, "source_dataset", dataset)
        cited_rows.append(cited)

    if dry_run:
        return

    aggregate_tables_dir = run_dir / "aggregate" / "tables"
    aggregate_tables_dir.mkdir(parents=True, exist_ok=True)
    comparisons_out = aggregate_tables_dir / "summary_comparisons.csv"
    cited_out = aggregate_tables_dir / "cited_distribution_summary.csv"
    pd.concat(comparison_rows, ignore_index=True).to_csv(comparisons_out, index=False)
    pd.concat(cited_rows, ignore_index=True).to_csv(cited_out, index=False)
    print(f"Wrote {comparisons_out}")
    print(f"Wrote {cited_out}")


def plot_core_outputs(
    config: dict[str, Any],
    *,
    project_dir: Path,
    run_dir: Path,
    python_bin: str,
    dry_run: bool,
    env: dict[str, str],
    run_log: RunLog | None = None,
) -> None:
    """Run the existing field-scan plotting script for aggregate outputs."""
    run_command(
        [
            python_bin,
            "scripts/plot_field_scan_core_metrics.py",
            "--summary",
            str(run_dir / "aggregate" / "tables" / "summary_comparisons.csv"),
            "--output-dir",
            str(run_dir / "aggregate" / "figures"),
            "--field-order",
            ",".join(str(field["label"]) for field in config["fields"]),
        ],
        cwd=project_dir,
        dry_run=dry_run,
        env=env,
        run_log=run_log,
        stage="plot_core",
    )


def run_cross_field_did(
    config: dict[str, Any],
    *,
    project_dir: Path,
    run_dir: Path,
    python_bin: str,
    dry_run: bool,
    env: dict[str, str],
    run_log: RunLog | None = None,
) -> None:
    """Run DID comparisons from configured seed-level metric files."""
    did = config.get("did", {})
    output_dir_name = Path(str(did.get("output_dir", "did_democratization"))).name
    cmd = [
        python_bin,
        "-m",
        "src.pipelines.run_cross_field_did_analysis",
        "--treatment-field-label",
        str(did["treatment_field_label"]),
        "--output-dir",
        str(run_dir / "aggregate" / "tables" / output_dir_name),
    ]
    for field in config["fields"]:
        paths = field_paths(project_dir, run_dir, str(field["dataset"]))
        cmd.extend(
            ["--input", f"{field['label']}={paths['democratization'] / 'tables' / 'seed_democratization_metrics.csv'}"]
        )
    for control in did.get("control_field_labels", []):
        cmd.extend(["--control-field-label", str(control)])
    run_command(cmd, cwd=project_dir, dry_run=dry_run, env=env, run_log=run_log, stage="cross_field_did")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Date-stamped YAML parameter document.")
    parser.add_argument("--field", action="append", help="Dataset name to run; repeat to select multiple fields.")
    parser.add_argument("--only", action="append", help="Step name to run; repeat for multiple steps.")
    parser.add_argument("--skip", action="append", default=[], help="Step name to skip; repeat for multiple steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute SciNCL embeddings and pair scores even if a matching cache exists.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Reuse an existing results/<run-id>/ directory instead of creating a new one "
            "(e.g. to resume a partial run with --only/--field). Defaults to a fresh "
            "'<run.name>_<UTC timestamp>' directory, so every invocation without this flag "
            "is a new, self-contained attempt that cannot overwrite a previous one."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_dir = resolve_project_dir(config_path, config)
    python_bin = str(config.get("run", {}).get("python", sys.executable))
    only = set(args.only) if args.only else None
    skip = set(args.skip or [])
    selected_fields = set(args.field or [])
    env = os.environ.copy()

    api_key_env = config.get("openalex", {}).get("api_key_env")
    if api_key_env and api_key_env in os.environ:
        env["OPENALEX_API_KEY"] = os.environ[api_key_env]

    fields = [
        field
        for field in config["fields"]
        if not selected_fields or field["dataset"] in selected_fields or field["label"] in selected_fields
    ]
    if not fields:
        raise ValueError(f"No configured fields matched --field values: {sorted(selected_fields)}")

    run_name = config.get("run", {}).get("name", "pipeline")
    started_at = datetime.now(timezone.utc)
    run_id = args.run_id or make_run_id(run_name, started_at)
    run_dir = project_dir / "results" / run_id
    print(f"Run directory: {run_dir}" + (" (dry run; nothing will be written)" if args.dry_run else ""), flush=True)

    # The run log is skipped in dry-run mode since no stage actually executes.
    run_log = None
    if not args.dry_run:
        log_path = run_dir / "run_logs" / f"{run_id}.md"
        run_log = RunLog(
            log_path,
            run_name=run_name,
            config=config,
            config_path=config_path,
            command=sys.argv,
            field_labels=[str(field["label"]) for field in fields],
            only=only,
            skip=skip,
            start_time=started_at,
        )

    try:
        should_run_field_steps = only is None or bool(only & FIELD_STEPS)
        if should_run_field_steps:
            for field in fields:
                run_field_pipeline(
                    field,
                    config=config,
                    project_dir=project_dir,
                    run_dir=run_dir,
                    python_bin=python_bin,
                    dry_run=args.dry_run,
                    only=only,
                    skip=skip,
                    env=env,
                    force_recompute_cli=args.force_recompute,
                    run_log=run_log,
                )

        full_field_set = len(fields) == len(config["fields"])
        if step_is_enabled("aggregate_core", config, only, skip):
            force_aggregate = only is not None and "aggregate_core" in only
            if full_field_set or force_aggregate:
                start = time.time()
                status = "ok"
                try:
                    aggregate_core_outputs(config, fields, project_dir, run_dir, args.dry_run)
                except Exception:
                    status = "failed"
                    raise
                finally:
                    if run_log is not None:
                        run_log.record(
                            dataset=None, stage="aggregate_core", duration_seconds=time.time() - start, status=status
                        )
            else:
                print(
                    "Skipping aggregate_core for a partial field run; rerun without --field after all fields finish.",
                    flush=True,
                )

        if step_is_enabled("plot_core", config, only, skip):
            force_plot = only is not None and "plot_core" in only
            if full_field_set or force_plot:
                plot_core_outputs(
                    config,
                    project_dir=project_dir,
                    run_dir=run_dir,
                    python_bin=python_bin,
                    dry_run=args.dry_run,
                    env=env,
                    run_log=run_log,
                )
            else:
                print("Skipping plot_core for a partial field run; rerun without --field after all fields finish.", flush=True)

        if step_is_enabled("cross_field_did", config, only, skip):
            run_cross_field_did(
                config,
                project_dir=project_dir,
                run_dir=run_dir,
                python_bin=python_bin,
                dry_run=args.dry_run,
                env=env,
                run_log=run_log,
            )
    finally:
        if run_log is not None:
            run_log.finalize()


if __name__ == "__main__":
    main()
