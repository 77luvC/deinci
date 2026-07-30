# Citation Relevance POC

OpenAlex citation-behavior pipeline for comparing pre-AI and post-AI reference
patterns across field-specific 200-seed samples.

## Current Scope

The fixed core sample size is 200 seed papers per field:

- `pre_ai`: query paper publication date from 2019-01-01 through 2021-12-31
- `post_ai`: query paper publication date from 2023-01-01 through 2026-06-30
- 2022 is skipped as a transition year

Current field-scan datasets:

- Computer Science: OpenAlex field ID `17`
- Health Informatics: OpenAlex subfield/topic filter ID `2718`
- Public Health: OpenAlex subfield/topic filter ID `2739`
- Psychology: OpenAlex field ID `32`
- Social Sciences: OpenAlex field ID `33`

The old demo, toy, 50-seed expanded, Math, Pure Math, and broad Medicine result
folders are intentionally removed from the tracked project state.

## Metrics Registry

See [`docs/metrics_registry.md`](docs/metrics_registry.md) and
[`docs/metrics_registry_en.md`](docs/metrics_registry_en.md) for notation,
metric formulas, thresholds, output tables, and interpretation rules.

## Naming Conventions

See [`docs/NAMING.md`](docs/NAMING.md) for the target naming system for
data dirs, config files, run/results dirs, and logs (separator grammar,
field-code vocabulary, templates) and the known deviations from it in the
current tree.

## Config-Driven Pipeline

The preferred way to rerun the project from scratch is through a date-stamped
parameter file:

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_2026-07-27.yaml
```

Use `--dry-run` before a large SCC run to print all commands without making
OpenAlex calls or writing outputs:

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_2026-07-27.yaml \
  --dry-run
```

The parameter file controls field list, seed count, OpenAlex sampling size,
historical citation-count method, core metrics, relevance scoring, and optional
DID. See [`docs/pipeline_runbook.md`](docs/pipeline_runbook.md) for SCC usage
and resume examples.

## Manual Core Steps

Run the commands below from this directory. Replace `<dataset>`, `<field-id>` or
`<subfield-id>` with one of the configured field-scan datasets, and `<run-id>`
with a unique identifier for this attempt (e.g. `manual_2026-07-27`) — every
run should use its own `<run-id>` so its figures/tables/reports never
overwrite a previous attempt's; see
[`docs/pipeline_runbook.md`](docs/pipeline_runbook.md#run-directories).
`data/raw/<dataset>` and `data/processed/<dataset>` are *not* run-scoped —
they're reused across runs for the same dataset.

```bash
python -m src.data_loaders.openalex_expanded_builder \
  --output-dir data/raw/<dataset> \
  --field-id <field-id> \
  --language en \
  --seeds-per-period 200 \
  --candidate-sample-size 3000 \
  --min-references 8 \
  --seed 42

python -m src.data_loaders.generic_pairs_loader \
  --papers-csv data/raw/<dataset>/papers.csv \
  --pairs-csv data/raw/<dataset>/pairs.csv \
  --output-dir data/processed/<dataset>

python -m src.pipelines.run_pair_metadata \
  --papers data/processed/<dataset>/papers.parquet \
  --pairs data/processed/<dataset>/pairs.parquet \
  --output results/<run-id>/<dataset>/tables/pair_metadata.parquet

python -m src.pipelines.run_historical_citation_counts \
  --scores results/<run-id>/<dataset>/tables/pair_metadata.parquet \
  --papers data/processed/<dataset>/papers.parquet \
  --output results/<run-id>/<dataset>/tables/pair_metadata_historical.parquet \
  --cache data/processed/<dataset>/historical_citation_counts_yearly_cache.csv \
  --method yearly

python -m src.pipelines.run_cited_distribution_analysis \
  --scores results/<run-id>/<dataset>/tables/pair_metadata_historical.parquet \
  --papers data/processed/<dataset>/papers.parquet \
  --output-dir results/<run-id>/<dataset>

python -m src.pipelines.run_democratization_analysis \
  --scores results/<run-id>/<dataset>/tables/pair_metadata_historical.parquet \
  --papers data/processed/<dataset>/papers.parquet \
  --output-dir results/<run-id>/<dataset> \
  --low-citation-quantile 0.25
```

`run_cited_distribution_analysis` and `run_democratization_analysis` each take
the field's whole output directory (`results/<run-id>/<dataset>`) and sort
their own outputs into `tables/` and `figures/` subdirectories.

For Health Informatics and Public Health, pass `--subfield-ids <id>` instead of
`--field-id <field-id>`.

## Relevance And DID Steps

The embedding and relevance-evaluation path is kept for the next round of
analysis:

```bash
python -m src.pipelines.run_embed \
  --papers data/processed/<dataset>/papers.parquet \
  --output data/processed/<dataset>/scincl_embeddings \
  --model malteos/scincl

python -m src.pipelines.run_pair_scoring \
  --papers data/processed/<dataset>/papers.parquet \
  --pairs data/processed/<dataset>/pairs.parquet \
  --embeddings data/processed/<dataset>/scincl_embeddings \
  --output results/<run-id>/<dataset>/tables/pair_scores.parquet \
  --model malteos/scincl

python -m src.pipelines.run_temporal_analysis \
  --scores results/<run-id>/<dataset>/tables/pair_scores.parquet \
  --output-dir results/<run-id>/<dataset>
```

`run_embed` writes to a stable, dataset-scoped (not run-scoped) cache and
skips recomputation whenever the model and paper inputs are unchanged;
`run_pair_scoring` does the same for its own run-scoped output. Pass
`--force-recompute` to override either. See
[`docs/pipeline_runbook.md`](docs/pipeline_runbook.md#embedding-and-pair-score-persistence)
for where these caches live and how reuse is decided.

This pipeline only carries observed/actual citation pairs (no sampled negative
pairs), so it does not include a positive-vs-negative discrimination
evaluation (ROC-AUC, Cohen's d, retrieval/ranking metrics). That comparison
was validated separately and is intentionally out of scope here.

Cross-field DID remains available through
`src.pipelines.run_cross_field_did_analysis` after the relevant field-level
`seed_democratization_metrics.csv` files are produced.

## Field-Scan Figures

```bash
python scripts/plot_field_scan_core_metrics.py \
  --summary results/<run-id>/aggregate/tables/summary_comparisons.csv \
  --output-dir results/<run-id>/aggregate/figures
```
