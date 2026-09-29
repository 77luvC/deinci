# Citation Relevance POC

This pipeline compares the references used by OpenAlex-indexed research articles
published before and after the widespread availability of generative AI. It
collects actual citing-to-cited pairs, estimates each cited work's visibility
at the citing paper's publication year, scores title-and-abstract similarity
with SciNCL, and compares seed-paper-level reference patterns within fields.
The period comparison is descriptive: publication after 2022 is not a measure
of whether an author used AI.

## Choose a run

Run commands from `citation_relevance_poc/`. Each config specifies its own
field list, target seeds **per field and period**, candidate pool, and enabled
steps.

| Purpose | Config | Target sample |
| --- | --- | --- |
| Small end-to-end check | [`pipeline_sanity_check_cs_seed1.yaml`](configs/pipeline_sanity_check_cs_seed1.yaml) | 1 Computer Science seed per period |
| Five-field 200-seed run | [`pipeline_2026-07-27.yaml`](configs/pipeline_2026-07-27.yaml) | 200 per field and period; includes Social Sciences |
| Five-field 1,000-seed run | [`pipeline_field_scan_5fields_1000_2026-07-29.yaml`](configs/pipeline_field_scan_5fields_1000_2026-07-29.yaml) | 1,000 per field and period; includes Education |

The completed 1,000-seed run is archived under
[`results_1000/field_scan_5fields_1000_2026-07-29_20260730T002823Z/`](results_1000/field_scan_5fields_1000_2026-07-29_20260730T002823Z/).
Its five fields are Computer Science (OpenAlex field 17), Health Informatics
(subfield 2718), Public Health (subfield 2739), Psychology (field 32), and
Education (subfield 3304). The 200-seed config uses Social Sciences (field 33)
instead of Education. Do not mix their outputs.

Both substantive configs compare citing articles from **2019-01-01 to
2021-12-31** (`pre_ai`) with **2023-01-01 to 2026-06-30** (`post_ai`).
The pipeline excludes 2022 as a transition year. Seed candidates must be
English-language articles with a reconstructable abstract and at least eight
OpenAlex references. The 200-seed config samples up to 3,000 candidates per
field-period; the 1,000-seed config samples up to 5,000. The builder selects
seed papers across citation-count tertiles, then retains referenced works
with usable English titles and abstracts. The number of seed papers with
usable reference pairs in the analysis can be below the selection target.

## Install

```bash
git clone https://github.com/77luvC/deinci.git
cd deinci/citation_relevance_poc
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The data-building stages need access to OpenAlex. SciNCL scoring downloads
`malteos/scincl` on first use; a GPU is useful for larger runs, although CPU
execution is possible. You may set `OPENALEX_API_KEY` in the shell. Before
reusing a dated config for new OpenAlex requests, copy it and replace its
`openalex.mailto` value with your own contact email or `null`. Do not commit
an API key to YAML or Git.

## Run the pipeline

First print the commands without making API calls or writing output:

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_sanity_check_cs_seed1.yaml \
  --dry-run
```

Then run the small end-to-end check:

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_sanity_check_cs_seed1.yaml
```

For a new five-field 1,000-seed run **from scratch**, copy the archived
settings, then edit `run.name` and `openalex.mailto` in the copy:

```bash
cp configs/pipeline_field_scan_5fields_1000_2026-07-29.yaml \
  configs/pipeline_my_1000_run.yaml
# Edit run.name and openalex.mailto in configs/pipeline_my_1000_run.yaml

python scripts/run_configured_pipeline.py \
  --config configs/pipeline_my_1000_run.yaml \
  --dry-run

python scripts/run_configured_pipeline.py \
  --config configs/pipeline_my_1000_run.yaml
```

This full run collects raw data for all five fields and can take substantial
time and OpenAlex requests. The archived 1,000-seed aggregate run used
`--skip build_raw` **because its five raw datasets had already been built in
earlier per-field runs**. Those raw 1,000-seed datasets are not tracked in this
repository. A fresh clone must build them first; copying the archived command
with `--skip build_raw` will not recreate the archive.

To run the smaller five-field config, substitute
`configs/pipeline_2026-07-27.yaml`. For a new experiment, copy a config,
change `run.name` and any parameters, and keep that config with the results.
OpenAlex is a live source, so a new run with the same settings need not
reproduce the exact historical sample.

### What each stage does

1. `build_raw`: sample seed papers from OpenAlex and hydrate their listed
   references into `papers.csv` and `pairs.csv`.
2. `load_pairs`: convert those files to processed Parquet tables.
3. `relevance_scoring`: cache SciNCL paper embeddings and score each observed
   citation pair by cosine similarity. With the supplied configs, the
   metadata-only `pair_metadata` and `historical_metadata` stages are skipped
   because scored pairs are the analysis input.
4. `historical_scores`: estimate the cited work's citation count through the
   citing paper's publication year.
5. `cited_distribution`, `democratization`, and `temporal_relevance`:
   create field-level tables and figures.
6. `aggregate_core` and `plot_core`: combine the configured fields into
   comparison tables and figures. `cross_field_did` is disabled in both
   five-field configs.

The core analysis has only **observed citations**. It does not create
noncited negative pairs or report a positive-vs-negative ROC-AUC. See the
[`metrics registry`](docs/metrics_registry_en.md) for definitions,
thresholds, statistical tests, and interpretation cautions.

## Find the outputs

The runner prints `Run directory: ...`. Unless `--run-id` is supplied, each
invocation creates a new
`results/<run.name>_<UTC timestamp>/`. The main paths are:

```text
data/raw/<dataset>/papers.csv
data/raw/<dataset>/pairs.csv
data/processed/<dataset>/papers.parquet
data/processed/<dataset>/pairs.parquet
data/processed/<dataset>/scincl_embeddings__<model-slug>.npy
results/<run-id>/<dataset>/tables/seed_democratization_metrics.csv
results/<run-id>/<dataset>/tables/period_democratization_comparisons.csv
results/<run-id>/aggregate/tables/summary_comparisons.csv
results/<run-id>/aggregate/tables/cited_distribution_summary.csv
results/<run-id>/aggregate/figures/
results/<run-id>/run_logs/<run-id>.md
```

The `data/raw/` and `data/processed/` directories are reusable per dataset;
the `results/<run-id>/` directory separates one attempt's outputs from
another's. The log records the config, command, selected fields, stage status,
and timing. The tracked `results_1000/` tree is an archive of the completed
run, not the runner's default destination. Its aggregate
[`summary_comparisons.csv`](results_1000/field_scan_5fields_1000_2026-07-29_20260730T002823Z/aggregate/tables/summary_comparisons.csv)
and
[`cited_distribution_summary.csv`](results_1000/field_scan_5fields_1000_2026-07-29_20260730T002823Z/aggregate/tables/cited_distribution_summary.csv)
can be inspected without rerunning the pipeline. The completed run contains
9,966 usable seed papers and 256,662 retained citation pairs.

## Resume or run selected steps

Reuse the **exact run ID** printed by the initial run. Without `--run-id`,
the command starts a new results directory and later stages may not find
earlier outputs.

```bash
RUN_ID="paste-the-run-id-printed-by-your-first-run"

python scripts/run_configured_pipeline.py \
  --config configs/pipeline_my_1000_run.yaml \
  --run-id "$RUN_ID" \
  --field openalex_cs_1000 \
  --only historical_scores \
  --only cited_distribution \
  --only democratization

python scripts/run_configured_pipeline.py \
  --config configs/pipeline_my_1000_run.yaml \
  --run-id "$RUN_ID" \
  --only aggregate_core \
  --only plot_core
```

`--field` accepts a configured dataset name or field label. Repeat `--field`,
`--only`, or `--skip` for multiple values. Partial field runs skip aggregate
outputs by default; rebuild them after the field outputs are ready. Use
`--skip build_raw` only when the matching raw dataset exists, and also skip
`load_pairs` only when the processed Parquet tables exist. `--force-recompute`
ignores matching SciNCL embedding and pair-score caches.

For SCC, activate a Python environment and run
[`scripts/scc_run_configured_pipeline.sh`](scripts/scc_run_configured_pipeline.sh).
Its `CONFIG`, `PIPELINE_FIELD`, `PIPELINE_ONLY`, `PIPELINE_SKIP`,
`DRY_RUN`, `RUN_ID`, and `FORCE_RECOMPUTE` environment variables map to
the runner options. See the
[`pipeline runbook`](docs/pipeline_runbook.md) and
[`RUN_ON_SCC.md`](RUN_ON_SCC.md) for detailed examples.

## Reading the results

The aggregate comparison is post-period minus pre-period in seed-paper
means, with each usable seed paper weighted equally. The reported Welch
q-values use Benjamini-Hochberg correction within each field. Historical
citation counts are year-level approximations and can fall back to current
counts when history is unavailable. SciNCL cosine is a topical-similarity
proxy, not a claim-level judgment of whether a citation is appropriate.
Neither a period difference nor the optional DID code alone identifies the
causal effect of AI use.

For detailed output conventions, see [`docs/NAMING.md`](docs/NAMING.md);
for measures and caveats, see
[`docs/metrics_registry_en.md`](docs/metrics_registry_en.md).
