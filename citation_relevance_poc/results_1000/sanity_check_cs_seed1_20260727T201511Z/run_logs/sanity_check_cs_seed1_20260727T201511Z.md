# Run Log: sanity_check_cs_seed1

- Config file: `/project/d2d/citation_relevance_poc_scc_2026-07-27/configs/pipeline_sanity_check_cs_seed1.yaml`
- Command: `scripts/run_configured_pipeline.py --config configs/pipeline_sanity_check_cs_seed1.yaml`
- Started (UTC): 2026-07-27T20:15:11+00:00
- Finished (UTC): 2026-07-27T20:16:55+00:00
- Wall-clock elapsed: 103.8s
- Fields run: CS
- `--only`: (all steps)
- `--skip`: (none)

## Parameters

```yaml
run:
  name: sanity_check_cs_seed1
  project_dir: ..
  python: python
openalex:
  mailto: ran0925@bu.edu
  api_key_env: OPENALEX_API_KEY
  sleep_seconds: 0.15
sampling:
  language: en
  seeds_per_period: 1
  candidate_sample_size: 50
  min_references: 8
  random_seed: 42
historical_citations:
  method: yearly
  max_retries: 3
  checkpoint_every: 100
core_metrics:
  low_citation_quantile: 0.25
  relevance_threshold: 0.8
  use_relevance_scores_when_available: true
relevance_scoring:
  enabled: true
  model: malteos/scincl
steps:
  build_raw: true
  load_pairs: true
  pair_metadata: true
  historical_metadata: true
  relevance_scoring: true
  historical_scores: true
  cited_distribution: true
  democratization: true
  retrieval_eval: true
  temporal_relevance: true
  aggregate_core: true
  plot_core: true
  cross_field_did: false
fields:
- dataset: openalex_cs_sanity1
  label: CS
  field_id: '17'
```

## Stage Timings

| Dataset | Stage | Duration (s) | Status | Recorded (UTC) |
|---|---|---|---|---|
| openalex_cs_sanity1 | openalex_expanded_builder | 2.3 | ok | 2026-07-27T20:15:13+00:00 |
| openalex_cs_sanity1 | generic_pairs_loader | 0.8 | ok | 2026-07-27T20:15:14+00:00 |
| openalex_cs_sanity1 | run_pair_metadata | 0.7 | ok | 2026-07-27T20:15:15+00:00 |
| openalex_cs_sanity1 | run_historical_citation_counts:metadata | 1.3 | ok | 2026-07-27T20:15:16+00:00 |
| openalex_cs_sanity1 | run_embed | 71.0 | ok | 2026-07-27T20:16:27+00:00 |
| openalex_cs_sanity1 | run_pair_scoring | 4.0 | ok | 2026-07-27T20:16:31+00:00 |
| openalex_cs_sanity1 | run_historical_citation_counts:scores | 1.4 | ok | 2026-07-27T20:16:32+00:00 |
| openalex_cs_sanity1 | run_cited_distribution_analysis | 8.7 | ok | 2026-07-27T20:16:41+00:00 |
| openalex_cs_sanity1 | run_democratization_analysis | 2.9 | ok | 2026-07-27T20:16:44+00:00 |
| openalex_cs_sanity1 | run_retrieval_eval | 2.7 | ok | 2026-07-27T20:16:47+00:00 |
| openalex_cs_sanity1 | run_temporal_analysis | 3.1 | ok | 2026-07-27T20:16:50+00:00 |
| (run-level) | aggregate_core | 0.0 | ok | 2026-07-27T20:16:50+00:00 |
| (run-level) | plot_core | 4.8 | ok | 2026-07-27T20:16:55+00:00 |

## Summary

| Stage | Runs | Total (s) | Mean (s) |
|---|---|---|---|
| run_embed | 1 | 71.0 | 71.0 |
| run_cited_distribution_analysis | 1 | 8.7 | 8.7 |
| plot_core | 1 | 4.8 | 4.8 |
| run_pair_scoring | 1 | 4.0 | 4.0 |
| run_temporal_analysis | 1 | 3.1 | 3.1 |
| run_democratization_analysis | 1 | 2.9 | 2.9 |
| run_retrieval_eval | 1 | 2.7 | 2.7 |
| openalex_expanded_builder | 1 | 2.3 | 2.3 |
| run_historical_citation_counts:scores | 1 | 1.4 | 1.4 |
| run_historical_citation_counts:metadata | 1 | 1.3 | 1.3 |
| generic_pairs_loader | 1 | 0.8 | 0.8 |
| run_pair_metadata | 1 | 0.7 | 0.7 |
| aggregate_core | 1 | 0.0 | 0.0 |

**Total recorded wall-clock time:** 103.7s
