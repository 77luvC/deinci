# Citation Relevance POC

Minimal SciNCL proof-of-concept pipeline for paper-pair relevance scoring.

## Demo Run

Temporal AI-era grouping uses this convention:

- `non_ai`: query paper year before 2022
- `transition`: query paper year 2022 or 2023
- `ai`: query paper year after 2023

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
