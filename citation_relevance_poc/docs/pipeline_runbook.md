# Config-Driven Pipeline Runbook

This project now uses date-stamped YAML parameter files as the source of truth
for full pipeline runs. A run can change sample size, field list, OpenAlex
filters, historical-count settings, or relevance-scoring options without editing
the pipeline code.

## Plain-English Overview

One command, `run_configured_pipeline.py --config <file>`, walks through every
field listed in the config one at a time and, for each, does roughly this:

1. **Get the data.** Pull a balanced sample of pre-AI (2019-2021) and post-AI
   (2023-2026) seed papers from OpenAlex for that field, plus every reference
   they actually cite. Convert that into a standard papers/pairs table pair.
2. **Score topical relevance (optional, but on by default).** Embed every
   paper with SciNCL and compute a cosine-similarity score for each citing
   pair. This is the expensive step, so it's cached — if you rerun the
   pipeline and nothing about the papers or the model changed, it's skipped
   and the previous result is reused instead of recomputed. See
   [Embedding And Pair-Score Persistence](#embedding-and-pair-score-persistence)
   below for exactly how that works.
3. **Backfill historical citation counts**, so a cited paper's "visibility" is
   judged by how cited it was *at the time*, not how cited it is today.
4. **Run the analyses**: cited-work distribution, seed-level democratization
   metrics (pre-AI vs. post-AI), and temporal/grouped relevance trends.
5. **Move to the next field** and repeat, writing everything into its own
   dataset-named subtree inside this invocation's run directory, so nothing
   from one field overwrites another's results.

Once every field in the config has been processed, the runner optionally
aggregates the per-field summaries into one cross-field comparison table and
plots them, and (if configured) runs a difference-in-differences comparison
against control fields.

This pipeline only carries real, observed citation pairs — it does not
sample "negative" (non-cited) candidates and does not run any
positive-vs-negative discrimination evaluation. That's a separate,
already-completed validation of SciNCL and is out of scope here.

## Run Directories

Every invocation of `run_configured_pipeline.py` is a separate "attempt" and
gets its own directory, `results/<run-id>/`, so two runs (or a full run and an
exploratory partial one) can never overwrite each other's figures, tables, or
logs. By default `<run-id>` is `<run.name>_<UTC timestamp>`, e.g.
`sanity_check_cs_seed1_20260727T201511Z` — a fresh one every time you invoke
the script without extra flags.

Inside a run directory:

```text
results/<run-id>/
├── <dataset_1>/
│   ├── figures/   (PNG/PDF plots for that field)
│   ├── reports/   (reserved for narrative write-ups; currently unused)
│   └── tables/    (CSV/Parquet outputs + cache manifests for that field)
├── <dataset_2>/
│   └── ... same shape ...
├── aggregate/     (cross-field outputs, only written for a full-field run)
│   ├── figures/
│   └── tables/
└── run_logs/
    └── <run-id>.md
```

**Resuming a partial run** — `--only`/`--skip`/`--field` let you rerun just
part of a pipeline (e.g. redo `historical_scores` after fixing a bug), but by
default every invocation still gets a brand-new run directory, so a bare
resume command would land in an empty directory with nothing to build on.
Pass `--run-id <the-earlier-run-id>` to reopen that exact directory instead of
creating a new one — files for steps you didn't select are left untouched,
and steps you did select overwrite only their own files inside that run.
`data/raw/<dataset>/` and `data/processed/<dataset>/` (including the SciNCL
embedding cache) are **not** run-scoped — they live outside `results/` and are
reused across every run for the same dataset, regardless of `--run-id`, since
rebuilding them is the expensive, rate-limited part (OpenAlex fetches, SciNCL
encoding).

## Parameter Files

The current template is:

```text
configs/pipeline_2026-07-27.yaml
```

For a new run, copy the latest file and update the date in the filename and
`run.name`. The most important knobs are:

- `sampling.seeds_per_period`: seed papers per AI-era group.
- `sampling.candidate_sample_size`: OpenAlex random candidate pool size.
- `fields`: field labels and OpenAlex `field_id` or `subfield_ids`.
- `relevance_scoring.enabled`: whether to run SciNCL embeddings and pair scores.
- `core_metrics.use_relevance_scores_when_available`: whether core
  democratization outputs should include relevance-adjusted metrics.
- `historical_citations.method`: `yearly` for faster SCC-scale runs, `exact` for
  slower candidate/cutoff count queries.

## Local Run

From `citation_relevance_poc/`:

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_2026-07-27.yaml
```

Useful resume modes (note `--run-id`, which points back at the run directory
created by the original invocation — see [Run Directories](#run-directories)
above; without it, each command below would instead start a fresh, empty run):

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_2026-07-27.yaml \
  --run-id field_scan_200_2026-07-27_20260727T120000Z \
  --field openalex_cs_200 \
  --only historical_scores

python scripts/run_configured_pipeline.py \
  --config configs/pipeline_2026-07-27.yaml \
  --run-id field_scan_200_2026-07-27_20260727T120000Z \
  --skip build_raw \
  --skip load_pairs
```

Use `--dry-run` first to print the commands (and the run directory they'd
use) without making API calls or writing outputs.

When `--field` selects only part of the configured field list, the runner skips
aggregate plots by default so a partial rerun does not overwrite the full
field-scan summaries. After all fields finish, rebuild aggregate outputs with:

```bash
python scripts/run_configured_pipeline.py \
  --config configs/pipeline_2026-07-27.yaml \
  --run-id field_scan_200_2026-07-27_20260727T120000Z \
  --only aggregate_core \
  --only plot_core
```

## SCC Run

The wrapper `scripts/scc_run_configured_pipeline.sh` is intentionally scheduler
neutral. It can be called from an interactive compute node or wrapped in the
school's `qsub`/`sbatch` template.

```bash
CONFIG=/path/to/citation_relevance_poc/configs/pipeline_2026-07-27.yaml \
VENV_PATH=/path/to/venv \
OPENALEX_API_KEY="$OPENALEX_API_KEY" \
bash scripts/scc_run_configured_pipeline.sh
```

Environment overrides:

- `PIPELINE_FIELD="openalex_cs_200 openalex_psychology_200"` limits fields.
- `PIPELINE_ONLY="relevance_scoring historical_scores"` runs selected steps.
- `PIPELINE_SKIP="build_raw load_pairs"` skips steps.
- `DRY_RUN=1` prints commands only.
- `FORCE_RECOMPUTE=1` ignores cached embeddings/pair scores and recomputes them
  (see [Embedding And Pair-Score Persistence](#embedding-and-pair-score-persistence)).
- `RUN_ID=<existing-run-id>` resumes into that run directory instead of
  starting a new one (see [Run Directories](#run-directories)).

For relevance scoring on SCC, request a GPU if available. If using CPU only,
expect the embedding step to be the slowest part.

## Output Shape

Outside `results/` (shared across runs, keyed only by `<dataset>`):

```text
data/raw/<dataset>/
data/processed/<dataset>/
data/processed/<dataset>/scincl_embeddings__<model-slug>.npy
data/processed/<dataset>/scincl_embeddings__<model-slug>_metadata.parquet
data/processed/<dataset>/scincl_embeddings__<model-slug>.manifest.json
data/processed/<dataset>/historical_citation_counts_yearly_cache.csv
```

Inside `results/<run-id>/` (unique per invocation — see
[Run Directories](#run-directories)):

```text
results/<run-id>/<dataset>/tables/pair_metadata.parquet            (only when relevance scores are not the analysis input)
results/<run-id>/<dataset>/tables/pair_metadata_historical.parquet (only when relevance scores are not the analysis input)
results/<run-id>/<dataset>/tables/pair_scores.parquet
results/<run-id>/<dataset>/tables/pair_scores.parquet.manifest.json
results/<run-id>/<dataset>/tables/pair_scores_historical.parquet
results/<run-id>/<dataset>/tables/*.csv          (cited-distribution, democratization, temporal tables)
results/<run-id>/<dataset>/figures/*.png         (cited-distribution, temporal figures)
results/<run-id>/<dataset>/reports/              (reserved for narrative write-ups; currently unused)
results/<run-id>/aggregate/tables/summary_comparisons.csv
results/<run-id>/aggregate/tables/cited_distribution_summary.csv
results/<run-id>/aggregate/tables/did_democratization/       (cross-field DID, if configured)
results/<run-id>/aggregate/figures/*.png
results/<run-id>/run_logs/<run-id>.md
```

Every per-dataset output is namespaced by `<dataset>` *within* one run
directory, and every run gets its own `<run-id>` directory, so running the
pipeline across multiple fields — or running the pipeline again later — never
overwrites another field's or another run's results. All of it remains
available on disk afterward.

This pipeline only carries observed/actual citation pairs (positive `label=1`
rows). It does not construct sampled negative pairs and does not include a
positive-vs-negative discrimination evaluation (ROC-AUC, Cohen's d,
retrieval/ranking metrics) — that comparison is a separate, already-completed
validation of SciNCL and is intentionally out of scope for this analysis
pipeline.

Each invocation of `run_configured_pipeline.py` (not `--dry-run`) writes a
Markdown log at `results/<run-id>/run_logs/<run-id>.md`. The log records the
config path, the exact command line, the resolved parameters (the full
config, dumped as YAML), which fields/`--only`/`--skip` were selected, and a
stage-by-stage timing table — e.g. `generic_pairs_loader`, `run_embed`,
`run_pair_scoring`, `run_historical_citation_counts:metadata` vs `:scores`,
`run_cited_distribution_analysis` — per field, plus run-level stages
(`aggregate_core`, `plot_core`, `cross_field_did`). It is rewritten after
every stage, so a run that fails or is interrupted partway still leaves a
readable log with parameters and completed-stage timings, and a summary
(total/mean seconds per stage) prints to stdout at the end of the run.

## Embedding And Pair-Score Persistence

SciNCL embeddings and pair-level similarity scores are the most expensive
outputs to (re)compute, so both are cached and reused by default.

**In plain English:** each field's embeddings live in one predictable file
under `data/processed/<dataset>/`, named after the model that produced them.
Before running SciNCL, the pipeline checks "have I already embedded these
exact papers with this exact model?" — if yes, it reuses the saved vectors
instead of reloading the model and recomputing; if no (new papers, changed
text, or a different model), it recomputes and overwrites the cache. The same
check-before-compute pattern applies to the pair-similarity scores. The only
way to force a redo of unchanged data is to explicitly ask for it
(`--force-recompute`).

**Where embeddings live** — `run_embed.py` writes three files per
`(dataset, model)` pair under `data/processed/<dataset>/`:

- `scincl_embeddings__<model-slug>.npy` — the embedding matrix
- `scincl_embeddings__<model-slug>_metadata.parquet` — the paper rows the
  embeddings correspond to, in the same row order as the `.npy` file
- `scincl_embeddings__<model-slug>.manifest.json` — `model_name`, an
  `input_hash` over each paper's `paper_id`/`title`/`abstract`, `n_papers`,
  `embedding_dim`, `cache_schema_version`, and `created_at`

`<model-slug>` is the configured model name (e.g. `malteos/scincl` ->
`malteos-scincl`), so switching embedding models never silently reuses another
model's vectors, and the dataset directory itself already scopes the cache to
one field/dataset.

**How reuse is decided** — before encoding, `run_embed.py` hashes the current
`papers.parquet` the same way and compares it against the manifest. If the
model name and input hash match, and the `.npy`/`_metadata.parquet` files
still exist, it prints a "Reusing cached embeddings" message and returns
without loading the model. SciNCL is only rerun when:

- no manifest/cache exists yet for that dataset+model,
- the papers table changed (different `paper_id`, `title`, or `abstract`
  content — row order does not matter),
- a different model is configured, or
- `--force-recompute` is passed (config: `relevance_scoring.force_recompute:
  true`; CLI: `--force-recompute` on `run_configured_pipeline.py`; SCC wrapper:
  `FORCE_RECOMPUTE=1`).

**Where pair scores live and how their reuse works** — `run_pair_scoring.py`
writes `results/<run-id>/<dataset>/tables/pair_scores.parquet` plus a sibling
`.manifest.json` recording the model name, the embeddings manifest's
`input_hash`, and a hash of the pairs table (`query_paper_id`/`candidate_paper_id`/
`label`). A rerun reuses the existing scores file whenever that manifest
still matches (i.e. neither the pairs, the embeddings, nor the model
changed); otherwise it recomputes and rewrites both files. This is also the
point that guarantees `scincl_cosine` is always populated — the scoring step
raises rather than writing a file with missing scores.

Because this file lives under the per-run directory, a brand-new run (a
fresh `<run-id>`) always recomputes it — there's nothing to reuse yet at that
new path. That's fine: joining cached embeddings into pair scores is a cheap
local dot-product, not a model run, so the cost that actually matters
(SciNCL encoding) still only happens when the embedding cache misses. The
pair-score cache's reuse benefit shows up when you resume *the same* run with
`--run-id` (see [Run Directories](#run-directories)) — e.g. rerunning
`historical_scores` after `relevance_scoring` already completed in that run
skips recomputing the join.

**Metadata-only files stay out of the way** — `run_pair_metadata.py` (and its
historical-counts pass) always emit an empty `scincl_cosine` column by design,
for the case where relevance scoring is off. `run_configured_pipeline.py` only
runs those two steps when they will actually be used as the analysis input —
i.e. when `relevance_scoring.enabled: false`, or
`core_metrics.use_relevance_scores_when_available: false`. When relevance
scoring is on and preferred (the default), the runner skips
`pair_metadata`/`historical_metadata` and prints why, so a full run does not
keep producing an unused file with a blank `scincl_cosine` column. Pass
`--only pair_metadata` (or `--only historical_metadata`) to force one of these
steps to run anyway, e.g. for ad hoc auditing.
