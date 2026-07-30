# Metrics Registry

This document fixes the current Citation Relevance POC dimensions, metric definitions, formulas, and output locations. The current definitions are based on the pipeline code under `src/`. Some historical result files may still use older column names or older current-citation fallbacks; see "Scope Notes" at the end.

This pipeline carries only observed/actual citation pairs (`label == 1`). It
does not construct sampled negative pairs and does not include a
positive-vs-negative discrimination evaluation (ROC-AUC, Cohen's d,
retrieval/ranking metrics computed against negatives). That validation of
SciNCL was done separately, already completed, and is intentionally out of
scope for this pipeline; see the project memory / run notes for that
evaluation.

## Plain-English Overview

Skip this section if you just want the formulas below — this is the same
information, in prose, for anyone who wants the story before the notation.

**What question is this answering?** For a given research field (CS, Health
Informatics, Public Health, Psychology, Social Sciences), we sample ~200
"seed" papers published just before the ChatGPT era (2019-2021, `pre_ai`) and
~200 published well after it (2023-2026, `post_ai`), then look at everything
each seed paper actually cites. The question is whether *what* papers cite
changed after GenAI tools became widely available — are post-AI papers
citing more famous/incumbent work, or are they still reaching into the long
tail of less-cited papers? Is their attention more concentrated in fewer
fields, or more spread out?

**How does a paper move through the pipeline, field by field?**

1. **Sample papers from OpenAlex.** For one field, pull ~200 pre-AI and ~200
   post-AI seed papers (balanced across low/medium/high citation counts so we
   aren't only looking at already-famous papers), then fetch every reference
   each seed paper lists that has a usable title and abstract. Every
   (seed, reference) pair becomes one row with `label = 1` — it's a real,
   observed citation. Nothing is invented or sampled as a "non-citation"
   here.
2. **Normalize into a common table shape** (papers table + pairs table),
   regardless of whether the data came from the OpenAlex builder or a
   manually supplied CSV.
3. **(Optional) Score topical relevance with SciNCL.** Each paper's
   title+abstract is embedded with the `malteos/scincl` model, and each
   citing/cited pair gets a cosine-similarity score (`scincl_cosine`) — a
   rough proxy for "how topically related are these two papers?" This step
   is the slow one (a transformer model over every paper), so its results
   are cached and reused across runs — see [Embedding And Pair-Score
   Persistence](pipeline_runbook.md#embedding-and-pair-score-persistence).
4. **Backfill historical citation counts.** For every cited paper, ask
   OpenAlex "how many citations did this paper actually have as of the year
   the seed paper cited it?" — not its citation count today. This avoids
   crediting a paper for fame it only earned after the fact.
5. **Run the descriptive analyses**, each answering one piece of the overall
   question:
   - *Cited-work distribution* — who gets cited, in terms of visibility,
     field, venue, work type, and age at time of citation.
   - *Democratization metrics* — per seed paper, does it lean toward famous
     incumbents or the long tail, and how concentrated is its attention?
     Compared pre-AI vs. post-AI with significance tests.
   - *Temporal / grouped relevance* — how does average topical relevance
     (`scincl_cosine`) itself shift across time periods, with an optional
     regression controlling for same-field citations.
6. **(Optional) Cross-field DID.** If you also ran "control" fields that
   shouldn't be affected by GenAI adoption, compare the pre/post shift in CS
   (or whichever field is "treated") against the shift in those controls,
   to get a difference-in-differences estimate that's less likely to be
   confounded by an unrelated, field-wide trend.

**Why is there no "negative pairs" step?** An earlier version of this
pipeline also sampled non-cited candidate papers to test whether SciNCL could
even tell a real citation apart from a random one. That validation has
already been done, separately, and confirmed SciNCL is fit for purpose — so
this pipeline no longer needs to carry that machinery. Every pair you'll see
in these outputs is a real, observed citation.

## Analysis Units

The project treats the citation system at two levels:

| Analysis unit | Key | Meaning | Main outputs |
|---|---|---|---|
| pair-level | `query_paper_id`, `candidate_paper_id` | One seed/query paper paired with one actual cited paper | pair scores, temporal metrics |
| seed-level | `query_paper_id` | Aggregated metrics over one seed paper's actual reference set | `seed_democratization_metrics.csv` |
| group-level | `query_ai_group`, `dataset`, etc. | Aggregated comparisons by period, field, dataset, or treatment/control group | temporal outputs, democratization summaries, DID outputs |

Notation:

| Symbol | Definition |
|---|---|
| `i` | seed/query/citing paper, the paper making citations |
| `j` | candidate/cited paper, the paper actually cited |
| `R_i` | actual OpenAlex reference set of seed paper `i` |
| `s_ij` | SciNCL embedding cosine similarity between seed paper `i` and candidate paper `j` |
| `y_i`, `y_j` | publication years of seed paper `i` and candidate paper `j` |
| `c_ij` | historical visibility citation count of candidate paper `j` through the publication-year cutoff of seed paper `i` |
| `label_ij` / `label` | Pair label; always `1` in this pipeline (seed paper `i` actually cites paper `j`). Kept in the schema for compatibility with manually supplied pair files, not used to construct or evaluate negative pairs |

## Core Dimensions

| Dimension | Field | Values / definition | Generated by | Use |
|---|---|---|---|---|
| AI era | `query_ai_group` | `pre_ai`: 2019-01-01 to 2021-12-31; `post_ai`: 2023-01-01 to 2026-06-30; otherwise `outside_window` or `unknown` | `add_temporal_groups` | Main period comparison |
| Broad period | `query_period` | `pre_2020` if `query_year <= 2020`, otherwise `post_2020` | `add_temporal_groups` | Early temporal sanity check |
| GenAI topic | `query_genai_topic` | `genai_topic` if `query_is_genai`, otherwise `non_genai_topic` | `add_temporal_groups` | Topic heterogeneity |
| Combined group | `combined_group` | `query_period + "_" + query_ai_group` | `add_temporal_groups` | Crossed temporal group |
| Field | `query_field`, `candidate_field` | OpenAlex primary-topic field display name | pair scoring / pair metadata / metadata merge | Field match and cross-field share |
| Same field | `same_field` | `query_field == candidate_field` | temporal/cited-distribution metadata | Field homophily |
| Dataset / field sample | `dataset`, `source_dataset` | Examples: `openalex_cs_200`, `openalex_health_informatics_200` | builders / cross-field DID | Treatment/control grouping |
| Candidate work type | `candidate_work_type` | OpenAlex work type, e.g. `article`, `book`, `preprint` | metadata merge | Reference composition |
| Candidate venue | `candidate_venue` | OpenAlex primary source display name, fallback `unknown` | metadata merge | Venue concentration/composition |
| Candidate OA | `candidate_is_oa` | OpenAlex open-access boolean | metadata merge | Reference openness |
| Seed citation stratum | `seed_citation_stratum` | `low`, `medium`, `high` tertiles of seed cited-by count at sampling time | OpenAlex builders | Sampling balance and robustness |

## Shared Derived Fields

| Field | Definition / formula | Notes |
|---|---|---|
| `scincl_cosine` | `s_ij = dot(emb_i, emb_j)` | Embeddings come from `malteos/scincl`; the current encoder normalizes vectors, so the dot product is a cosine proxy |
| `scincl_cosine = NA` | Pair metadata placeholder when embeddings are not computed | `run_pair_metadata.py` can emit pair-level metadata without SciNCL scores; relevance metrics are then conditionally unavailable |
| `candidate_age_at_citation` | `y_i - y_j` | Negative ages are treated as invalid for seed-level age metrics |
| `valid_cited_age` | `candidate_age_at_citation` if `>= 0`, otherwise null | Avoids future-dated cited works contaminating age metrics |
| `candidate_current_cited_by_count` | OpenAlex current `cited_by_count` | Current citation totals can leak future information |
| `candidate_visibility_cited_by_count` | Prefer `candidate_cited_by_count_at_query_year`; if unavailable, fall back to current cited-by count | Recommended field for all visibility/prestige metrics |
| `candidate_visibility_source` | `citation_year` or `current_fallback` | Indicates whether visibility comes from historical counts or current-count fallback |
| `candidate_log1p_visibility_cited_by_count` | `log(1 + candidate_visibility_cited_by_count)` | Dampens heavy-tailed citation counts |
| `candidate_cited_by_count_at_query_year` | Historical count through query-year cutoff; actual reference pairs subtract the focal citation itself | Generated by `run_historical_citation_counts.py` |

Historical citation cutoff rule:

| Method | Formula / rule |
|---|---|
| exact | OpenAlex query `filter=cites:{candidate_paper_id},to_publication_date:{cutoff_date}`, where `cutoff_date = min(query publication year end, 2026-06-30)` |
| yearly | `current_cited_by_count - sum(counts_by_year for years after query publication year)` |
| focal-pair adjustment | If `label_ij == 1` (code column: `label`), `candidate_cited_by_count_at_query_year = max(raw_count - 1, 0)` |

## Pair-Level Relevance Metrics

These are plain descriptive statistics of `scincl_cosine` over observed citation pairs — not a positive-vs-negative discrimination measure, since this pipeline does not carry sampled negative pairs. (A separate, already-completed experiment validated that SciNCL scores separate actual citations from negative/comparison candidates; that evaluation is out of scope here.)

| Metric | Definition | Formula | Output |
|---|---|---|---|
| `n_pairs` | Total pair count | `N` | grouped temporal outputs (`temporal_metrics.csv`, `ai_era_metrics.csv`, etc.) |
| `n_scored_pairs` | Pairs with a non-missing `scincl_cosine` | `sum(notna(s_ij))` | same |
| `mean_score` | Mean SciNCL score | `mean(s_ij)` | same |
| `median_score` | Median SciNCL score | `median(s_ij)` | same |
| `std_score` | Sample standard deviation of SciNCL score | `std(s_ij, ddof=1)` | same |
| `p25_score` | 25th percentile of SciNCL score | `quantile(s_ij, .25)` | same |
| `p75_score` | 75th percentile of SciNCL score | `quantile(s_ij, .75)` | same |

## Temporal / Grouped Relevance Metrics

`run_temporal_analysis.py` computes pair-level relevance metrics by group:

| Output file | Grouping dimension | Metric set |
|---|---|---|
| `temporal_metrics.csv` | `query_period` | pair-level relevance metrics |
| `ai_era_metrics.csv` | `query_ai_group` | pair-level relevance metrics |
| `genai_topic_metrics.csv` | `query_genai_topic` | pair-level relevance metrics |
| `combined_group_metrics.csv` | `combined_group` | pair-level relevance metrics |

Optional DID-style regression:

```text
scincl_cosine ~ post_2020 + post_ai + post_2020:post_ai + same_field_int
```

Terms:

| Term | Meaning |
|---|---|
| `post_2020` | query in the broad post-2020 period |
| `post_ai` | query in the GenAI-era post window |
| `post_2020:post_ai` | interaction between broad period and AI-era group |
| `same_field_int` | whether query and candidate share the same field |

## Cited Distribution Metrics

These metrics use only actual references, i.e. `label == 1`. The current code aggregates by `query_ai_group`. If the input comes from `run_pair_metadata.py` rather than `run_pair_scoring.py`, `mean_score` and `median_score` remain `NaN`.

| Metric | Definition | Formula | Output |
|---|---|---|---|
| `n_reference_pairs` | Number of actual reference pairs | `|{(i,j): label_ij=1}|` | `cited_distribution_summary.csv` |
| `n_query_papers` | Number of seed papers with actual references | `nunique(query_paper_id)` | same |
| `mean_score` | Mean SciNCL score over actual references | `mean(s_ij)` | same |
| `median_score` | Median SciNCL score over actual references | `median(s_ij)` | same |
| `mean_cited_age` | Mean cited-work age | `mean(y_i - y_j)` | same |
| `median_cited_age` | Median cited-work age | `median(y_i - y_j)` | same |
| `mean_visibility_cited_by_count` | Mean cited-work visibility | `mean(c_ij)` | same |
| `median_visibility_cited_by_count` | Median cited-work visibility | `median(c_ij)` | same |
| `p25_visibility_cited_by_count` | 25th percentile of cited-work visibility | `quantile(c_ij, .25)` | same |
| `p75_visibility_cited_by_count` | 75th percentile of cited-work visibility | `quantile(c_ij, .75)` | same |
| `mean_log1p_visibility_cited_by_count` | Mean log visibility | `mean(log(1 + c_ij))` | same |
| `gini_visibility_cited_by_count` | Gini coefficient of cited-work visibility | `((2 * sum_k k*x_k) / (n*sum_k x_k)) - ((n+1)/n)`, where `x` is sorted ascending | same |
| `top10pct_share_of_visibility_citation_mass` | Citation mass share held by the top 10% most visible cited works | `sum(top ceil(.10*n) c_ij) / sum_all c_ij` | same |
| `top25pct_share_of_visibility_citation_mass` | Citation mass share held by the top 25% most visible cited works | `sum(top ceil(.25*n) c_ij) / sum_all c_ij` | same |
| `share_open_access` | Share of cited works that are OA | `mean(candidate_is_oa)` | same |
| `share_same_field` | Share of same-field references | `mean(query_field == candidate_field)` | same |
| `share_article` | Share of cited works with type `article` | `mean(candidate_work_type == "article")` | same |
| `share_book` | Share of cited works with type `book` | `mean(candidate_work_type == "book")` | same |

Distribution tables:

| Output file | Definition |
|---|---|
| `cited_visibility_quartiles.csv` | Ranks actual references by all-sample `candidate_visibility_cited_by_count`, bins them into Q1-Q4, then reports counts and shares by `query_ai_group` |
| `cited_work_type_distribution.csv` | Counts and shares by `query_ai_group` x `candidate_work_type` |
| `cited_field_distribution.csv` | Counts and shares by `query_ai_group` x `candidate_field` |
| `cited_venue_distribution.csv` | Counts and shares by `query_ai_group` x `candidate_venue` |

## Seed-Level Democratization Metrics

These are the primary research metrics. The analysis unit is seed paper `i` and its actual reference set `R_i`. The default comparison is `pre_ai` vs. `post_ai`. The current code separates always-on `BASE_METRICS` from conditional `RELEVANCE_METRICS`: `mean_score` and `share_relevant_low_citation_refs` appear in seed-level outputs, summaries, comparisons, and metric definitions only when the input contains usable numeric `scincl_cosine`. `median_cited_age` may still appear in some pipeline outputs, but it is no longer part of the current paper's primary metric set.

| Metric | Dimension | Definition | Formula |
|---|---|---|---|
| `n_reference_pairs` | coverage | Number of usable actual references for seed paper `i` | `|R_i|` |
| `share_recent_refs_3yr` | recency | Share of references to works from the past 3 years | `mean_{j in R_i}(0 <= y_i - y_j <= 3)` |
| `median_log1p_cited_by_count` | visibility / prestige | Median log citation-year visibility of cited works | `median_{j in R_i}(log(1 + c_ij))` |
| `share_low_citation_refs` | long-tail access | Share of references to low-citation works | `mean_{j in R_i}(c_ij <= T_low)` |
| `top10pct_share_of_citation_mass` | concentration | Share of the seed's cited-work visibility mass held by its top 10% most visible references | `sum_{j in Top10_i} c_ij / sum_{j in R_i} c_ij` |
| `share_cross_field_refs` | interdisciplinarity | Share of references crossing OpenAlex field boundaries | `mean_{j in R_i}(field_j != field_i)` |
| `cited_field_entropy` | diversity | Shannon entropy of the cited-field distribution | `-sum_f p_if * log(p_if)` |
| `mean_score` | relevance proxy | Mean SciNCL similarity over actual references | `mean_{j in R_i}(s_ij)` |
| `share_relevant_low_citation_refs` | relevance-adjusted long tail | Share of references that are both low-citation and high-SciNCL | `mean_{j in R_i}(c_ij <= T_low and s_ij >= T_rel)` |

Here, `Top10_i` is the set of the top `ceil(.10*|R_i|)` candidate papers in seed paper `i`'s references ranked by `c_ij`; `p_if` is the share of references in `R_i` that belong to field `f`.

Thresholds:

| Parameter | Default | Definition |
|---|---|---|
| `T_low` | `actual["candidate_visibility_cited_by_count"].quantile(0.25)` | Global bottom-quartile cutoff within the current actual-reference analysis sample |
| `T_rel` | `0.80` | SciNCL cosine relevance threshold |

Seed-level outputs:

| Output file | Content |
|---|---|
| `seed_democratization_metrics.csv` | Seed-level metrics for each `query_paper_id` |
| `period_democratization_summary.csv` | For each `query_ai_group` x `metric`: `n_seed_papers`, `mean`, `median`, `std`, `sem` |
| `period_democratization_comparisons.csv` | `post_ai` vs `pre_ai` mean differences, median differences, Welch t-test, Mann-Whitney U test, and BH q-values |
| `metric_definitions.csv` | Pipeline-generated short definitions and run-specific thresholds |

Interpretation for Matthew effect vs. democratization:

| Metric direction | More consistent with Matthew effect | More consistent with democratization |
|---|---|---|
| `median_log1p_cited_by_count` | Increase | Decrease |
| `share_low_citation_refs` | Decrease | Increase |
| `top10pct_share_of_citation_mass` | Increase | Decrease |
| `share_relevant_low_citation_refs` | Decrease | Increase |
| `cited_field_entropy` | Decrease or concentration | Increase |
| `share_cross_field_refs` | Decrease or field homophily | Increase |
| `mean_score` | Must be interpreted with visibility; higher alone does not imply democratization | If long-tail metrics improve and `mean_score` does not fall, this supports relevant long-tail discovery |

## Cross-Field DID Metrics

`run_cross_field_did_analysis.py` merges multiple seed-level metric files and estimates DID comparisons for one treatment dataset against one or more control datasets. DID automatically uses the `BASE_METRICS + RELEVANCE_METRICS` that are actually present in the input seed metrics, so field-scan datasets without SciNCL do not emit relevance DID.

| Metric / field | Definition | Formula |
|---|---|---|
| `treatment_post_mean` | Treatment dataset metric mean in the post period | `mean(metric | dataset=treat, period=post)` |
| `treatment_base_mean` | Treatment dataset metric mean in the base period | `mean(metric | dataset=treat, period=base)` |
| `control_post_mean` | Control dataset metric mean in the post period | `mean(metric | dataset=control, period=post)` |
| `control_base_mean` | Control dataset metric mean in the base period | `mean(metric | dataset=control, period=base)` |
| `treatment_change` | Treatment pre-post change | `treatment_post_mean - treatment_base_mean` |
| `control_change` | Control pre-post change | `control_post_mean - control_base_mean` |
| `did_estimate` | Difference-in-differences estimate | `(treatment_post - treatment_base) - (control_post - control_base)` |
| `interaction_coef` | OLS interaction coefficient | coefficient on `treated * post` |
| `interaction_t` | OLS interaction t-statistic | `coef / se` |
| `interaction_p_value` | Two-sided p-value | `2 * t.sf(abs(t), df_resid)` |
| `interaction_q_value_bh` | Benjamini-Hochberg adjusted q-value | BH correction over interaction p-values |

OLS implementation:

```text
metric ~ 1 + treated + post + treated:post
```

Current contrast:

| Post period | Base period |
|---|---|
| `post_ai` | `pre_ai` |

Some historical outputs may include `transition` as a base period; the current code fixes `post_ai` vs `pre_ai`.

## Output Map

Every path below is rooted at `results/<run-id>/`, where `<run-id>` is the
unique per-invocation directory described in
[Run Directories](pipeline_runbook.md#run-directories) (e.g.
`sanity_check_cs_seed1_20260727T201511Z/`) — no output is written directly
under `results/` anymore, so different runs and different fields never
collide on disk.

| Path pattern (relative to `results/<run-id>/`) | Pipeline / workflow | Analysis unit | Main content |
|---|---|---|---|
| `<dataset>/tables/pair_scores.parquet` | `run_pair_scoring.py` | pair | SciNCL scores plus query/candidate metadata |
| `<dataset>/tables/pair_scores.parquet.manifest.json` | `run_pair_scoring.py` | cache manifest | model, embeddings input hash, pairs hash used to decide reuse |
| `<dataset>/tables/pair_metadata.parquet` | `run_pair_metadata.py` | pair | query/candidate metadata with `scincl_cosine = NA`; only produced when relevance scores are not the analysis input (see [Embedding And Pair-Score Persistence](pipeline_runbook.md#embedding-and-pair-score-persistence)) |
| `<dataset>/tables/pair_scores_historical.parquet` | `run_historical_citation_counts.py` | pair | pair scores plus citation-year visibility counts |
| `<dataset>/tables/pair_metadata_historical.parquet` | `run_historical_citation_counts.py` | pair | pair metadata plus citation-year visibility counts, without usable SciNCL scores |
| `data/processed/<dataset>/scincl_embeddings__<model-slug>.npy` + `_metadata.parquet` + `.manifest.json` | `run_embed.py` | paper | cached SciNCL embeddings; lives outside `results/` (not run-scoped) and is reused across runs unless inputs/model change |
| `<dataset>/tables/*.csv` | `run_temporal_analysis.py` | grouped pair | relevance metrics by temporal dimensions |
| `<dataset>/figures/*.png` | `run_temporal_analysis.py` | grouped pair | score-by-period, score-by-AI-era plots |
| `<dataset>/tables/*.csv`, `<dataset>/tables/*.parquet` | `run_cited_distribution_analysis.py` | grouped actual references | visibility, composition, field/venue/work-type distributions |
| `<dataset>/figures/*.png` | `run_cited_distribution_analysis.py` | grouped actual references | visibility/age/relevance distribution plots |
| `<dataset>/tables/*.csv` | `run_democratization_analysis.py` | seed/group | seed-level democratization metrics and post/pre comparisons |
| `<dataset>/reports/` | (reserved) | — | currently unused; reserved for a future narrative per-field report |
| `aggregate/tables/did_democratization/*.csv` | `run_cross_field_did_analysis.py` | field_label-period-metric | cross-field DID summaries and estimates |
| `aggregate/tables/summary_comparisons.csv` | field-scan aggregation / plotting workflow | field_label-metric | post-AI minus pre-AI differences plus Welch and Mann-Whitney p-values |
| `aggregate/tables/cited_distribution_summary.csv` | field-scan aggregation workflow | field_label-group | compact cited-distribution summary: refs, queries, visibility, age, same-field share |
| `aggregate/figures/*` | `scripts/plot_field_scan_core_metrics.py` | figure | grouped plots for visibility, recency, breadth, and all-field heatmap |
| `run_logs/<run-id>.md` | `run_configured_pipeline.py` | run | config used, command line, per-stage timings |

Field-scan plotting metrics:

| Metric | Group | Displayed unit |
|---|---|---|
| `median_log1p_cited_by_count` | visibility | log points |
| `share_low_citation_refs` | visibility | percentage points |
| `top10pct_share_of_citation_mass` | visibility | percentage points |
| `share_recent_refs_3yr` | recency | percentage points |
| `share_cross_field_refs` | breadth | percentage points |
| `cited_field_entropy` | breadth | entropy points |

## Current Primary Metric Set

For the paper/research narrative, treat these as the primary metrics:

| Research question | Primary metrics | Secondary controls/checks |
|---|---|---|
| Are GenAI-era papers citing more incumbent/high-visibility work? | `median_log1p_cited_by_count`, `top10pct_share_of_citation_mass`, `gini_visibility_cited_by_count` | `p25/p75_visibility_cited_by_count`, visibility quartiles |
| Are GenAI-era papers opening attention to the long tail? | `share_low_citation_refs`, `share_relevant_low_citation_refs` | `mean_score`, `median_score` |
| Is attention becoming more concentrated or diverse? | `cited_field_entropy`, `share_cross_field_refs`, venue distribution | work type distribution, OA share |
| Are differences driven by recency? | `share_recent_refs_3yr` | query year/period stratification |
| Is CS changing differently from math/control fields? | DID estimates for seed-level democratization metrics | interaction p/q-values |

## Scope Notes

1. `SciNCL` is a citation-informed relevance proxy, not a citation quality judge. A high score suggests topical or scholarly relatedness, but it does not prove that the citation supports a specific claim, uses the best evidence, is non-decorative, or is non-hallucinated.
2. `malteos/scincl` was trained on historical citation data, so post-2020 and GenAI-era scores should be interpreted cautiously.
3. Citation visibility metrics should use `candidate_visibility_cited_by_count` whenever possible. If the input lacks historical counts, the code falls back to OpenAlex current cited-by totals.
4. Some older output files may still use `current_cited_by_count` column names; current code has migrated to `visibility_cited_by_count`. Formal runs should prefer historical enriched pair scores or pair metadata.
5. This pipeline only carries `label == 1` observed citation pairs; it does not construct negative pairs and does not compute positive-vs-negative discrimination metrics (ROC-AUC, average precision, Cohen's d, retrieval/ranking metrics). That evaluation was done separately as a one-time validation of SciNCL and is out of scope here.
6. `T_low` is the sample-internal global bottom-quartile threshold, not a fixed cross-dataset threshold. For cross-field DID, decide explicitly whether thresholds should be estimated separately by dataset or fixed across datasets.
7. The year 2022 is excluded from the `pre_ai` and `post_ai` windows to avoid mixing the transition around ChatGPT's release into either group.
8. The current OpenAlex observation end is fixed at 2026-06-30. If the data window changes, the historical cutoff rule and AI-era end date should be updated together.
