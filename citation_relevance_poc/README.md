# Citation Relevance POC

Minimal SciNCL proof-of-concept pipeline for paper-pair relevance scoring.

## Metrics Registry

See [`docs/metrics_registry.md`](docs/metrics_registry.md) and
[`docs/metrics_registry_en.md`](docs/metrics_registry_en.md) for the fixed registry of analysis
dimensions, metric definitions, formulas, output tables, and current interpretation rules.

## Demo Run

Temporal AI-era grouping uses this convention:

- `pre_ai`: query paper publication date from 2019-01-01 through 2021-12-31
- `post_ai`: query paper publication date from 2023-01-01 through 2026-06-30

Rows outside these windows are labeled `outside_window`. When publication dates are unavailable,
the grouping falls back to publication year.

The keyword-based GenAI topic flag is kept separately as `query_genai_topic`.

```bash
python -m src.data_loaders.generic_pairs_loader \
  --papers-csv data/raw/demo/papers.csv \
  --pairs-csv data/raw/demo/pairs.csv \
  --output-dir data/processed/demo

python -m src.pipelines.run_embed \
  --papers data/processed/demo/papers.parquet \
  --output data/processed/demo/scincl_embeddings \
  --model malteos/scincl

python -m src.pipelines.run_pair_scoring \
  --papers data/processed/demo/papers.parquet \
  --pairs data/processed/demo/pairs.parquet \
  --embeddings data/processed/demo/scincl_embeddings \
  --output results/demo_pair_scores.parquet

python -m src.pipelines.run_retrieval_eval \
  --scores results/demo_pair_scores.parquet \
  --output-dir results/demo_eval

python -m src.pipelines.run_temporal_analysis \
  --scores results/demo_pair_scores.parquet \
  --output-dir results/demo_temporal
```

## OpenAlex Toy Run

This builds a larger toy sample from real OpenAlex records. Positive pairs are actual OpenAlex
references from the query paper. Negative pairs are sampled from other referenced works in the toy
corpus and should be treated only as comparison candidates, not verified irrelevant papers.

```bash
python -m src.data_loaders.openalex_toy_builder \
  --output-dir data/raw/openalex_toy \
  --search "machine learning" \
  --queries-per-period 10 \
  --positives-per-query 3 \
  --negatives-per-positive 1 \
  --seed 7

python -m src.data_loaders.generic_pairs_loader \
  --papers-csv data/raw/openalex_toy/papers.csv \
  --pairs-csv data/raw/openalex_toy/pairs.csv \
  --output-dir data/processed/openalex_toy

python -m src.pipelines.run_embed \
  --papers data/processed/openalex_toy/papers.parquet \
  --output data/processed/openalex_toy/scincl_embeddings \
  --model malteos/scincl

python -m src.pipelines.run_pair_scoring \
  --papers data/processed/openalex_toy/papers.parquet \
  --pairs data/processed/openalex_toy/pairs.parquet \
  --embeddings data/processed/openalex_toy/scincl_embeddings \
  --output results/openalex_toy_pair_scores.parquet

python -m src.pipelines.run_retrieval_eval \
  --scores results/openalex_toy_pair_scores.parquet \
  --output-dir results/openalex_toy_eval

python -m src.pipelines.run_temporal_analysis \
  --scores results/openalex_toy_pair_scores.parquet \
  --output-dir results/openalex_toy_temporal

python -m src.pipelines.run_cited_distribution_analysis \
  --scores results/openalex_toy_pair_scores.parquet \
  --papers data/processed/openalex_toy/papers.parquet \
  --output-dir results/openalex_toy_cited_distribution
```

## OpenAlex Expanded Computer Science Run

This builds the next-phase OpenAlex sample from Computer Science seed papers using
OpenAlex primary topic field ID `17`. It samples English-language seed and reference
papers, 50 seed papers in each AI-period band
(`pre_ai`, `post_ai`), stratifies seed papers by cited-by-count tertiles,
and keeps all usable OpenAlex references with title and abstract as positive pairs.

```bash
python -m src.data_loaders.openalex_expanded_builder \
  --output-dir data/raw/openalex_expanded \
  --field-id 17 \
  --language en \
  --seeds-per-period 50 \
  --candidate-sample-size 750 \
  --min-references 8 \
  --seed 42

python -m src.data_loaders.generic_pairs_loader \
  --papers-csv data/raw/openalex_expanded/papers.csv \
  --pairs-csv data/raw/openalex_expanded/pairs.csv \
  --output-dir data/processed/openalex_expanded

python -m src.pipelines.run_embed \
  --papers data/processed/openalex_expanded/papers.parquet \
  --output data/processed/openalex_expanded/scincl_embeddings \
  --model malteos/scincl

python -m src.pipelines.run_pair_scoring \
  --papers data/processed/openalex_expanded/papers.parquet \
  --pairs data/processed/openalex_expanded/pairs.parquet \
  --embeddings data/processed/openalex_expanded/scincl_embeddings \
  --output results/openalex_expanded_pair_scores.parquet

python -m src.pipelines.run_historical_citation_counts \
  --scores results/openalex_expanded_pair_scores.parquet \
  --papers data/processed/openalex_expanded/papers.parquet \
  --output results/openalex_expanded_pair_scores_historical.parquet \
  --cache data/processed/openalex_expanded/historical_citation_counts_cache.csv

python -m src.pipelines.run_retrieval_eval \
  --scores results/openalex_expanded_pair_scores_historical.parquet \
  --output-dir results/openalex_expanded_eval

python -m src.pipelines.run_temporal_analysis \
  --scores results/openalex_expanded_pair_scores_historical.parquet \
  --output-dir results/openalex_expanded_temporal

python -m src.pipelines.run_cited_distribution_analysis \
  --scores results/openalex_expanded_pair_scores_historical.parquet \
  --papers data/processed/openalex_expanded/papers.parquet \
  --output-dir results/openalex_expanded_cited_distribution

python -m src.pipelines.run_democratization_analysis \
  --scores results/openalex_expanded_pair_scores_historical.parquet \
  --papers data/processed/openalex_expanded/papers.parquet \
  --output-dir results/openalex_expanded_democratization \
  --low-citation-quantile 0.25 \
  --relevance-threshold 0.80
```
