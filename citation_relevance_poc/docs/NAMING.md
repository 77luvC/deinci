# Naming Conventions

This project has no enforced naming system today — `run.name` in a config
file is free text, and every other name (data dirs, results dirs, config
files, root-level logs) has been typed by hand to loosely match it. That has
already produced drift (see [Known Deviations](#known-deviations) below).
This doc defines the target convention so new names stay decodable without
tribal knowledge, per general data-science / research-data-management
practice (snake_case identifiers, ISO 8601 dates, one separator per role —
see the dbt style guide and RDM file-naming guides such as Harvard's
[File Naming Conventions](https://datamanagement.hms.harvard.edu/plan-design/file-naming-conventions)).

It is descriptive, not yet enforced by code — `scripts/run_configured_pipeline.py`
still accepts arbitrary `run.name` values. Treat this as the target to write
new names against by hand until (if ever) it's worth encoding as validation.

## Separator grammar

Pick one job per separator and never swap them:

| Separator | Role |
|---|---|
| `_` | joins words *within* one token (e.g. `pure_math`, `health_informatics`) |
| `-` | joins token groups (source, fields, n, date) in a compound name |
| `+` | joins multiple field codes when one run spans several fields |
| `T…Z` | reserved for ISO 8601 timestamps only — never used elsewhere |

## Field codes (controlled vocabulary)

Every field has exactly one spelling, used everywhere — configs, data dirs,
results dirs, logs. Do not abbreviate or concatenate ad hoc (today's tree has
both `pure_math` and `puremath`, and `health_informatics` colliding with a
`health` used elsewhere as a short form of a different field — both are
exactly the drift this table exists to prevent):

| Code | Field |
|---|---|
| `cs` | Computer Science |
| `pure_math` | Pure Math |
| `health_informatics` | Health Informatics |
| `public_health` | Public Health |
| `psychology` | Psychology |
| `social_sciences` | Social Sciences |

Add a row here the same day you add a new field — don't let a shorthand get
invented at the call site.

## Templates

```
data dir:      data/{raw,processed}/{source}-{fields}-n{n}[-{qualifier}]
                 openalex-cs-n1000
                 openalex-cs+health_informatics+pure_math-n1000
                 openalex-cs-sanity-n1

config file:   configs/pipeline-{source}-{fields}-n{n}-{YYYY-MM-DD}.yaml
                 pipeline-openalex-cs-n1000-2026-07-27.yaml

run id:        {fields}-n{n}-{YYYYMMDDTHHMMSSZ}
                 cs-n1000-20260728T003245Z
               (the timestamp is generated once, at run time, by
               make_run_id() — it must not also be typed into run.name;
               today's `run.name: cs_1000_2026-07-27` plus the appended
               run timestamp is how two different dates end up in one
               directory name, e.g. `cs_1000_2026-07-27_20260728T003245Z`)

results dir:   results/{run_id}/{dataset}/{figures,reports,tables}/
                 (dataset here is the data-dir name above, so a run
                 directory's children are traceable back to their source
                 data without cross-referencing anything)

logs:          always results/{run_id}/run_logs/{run_id}.md
                 never at the project root
```

## Columns

Already followed correctly in this project's output tables — keep doing
this, don't change it:

- snake_case, full words, no invented abbreviations (`mean_score`, not `avg_scr`)
- a stat-prefix pattern for aggregates: `mean_`, `median_`, `std_`, `p25_`, `p75_` + `{metric}`
- the same column name means the same thing in every table it appears in
  (e.g. `n_pairs`, `n_scored_pairs` are identical across
  `genai_metrics.csv`, `combined_group_metrics.csv`, `temporal_metrics.csv`)

## Known deviations

Existing paths that predate this doc and don't match it. Left as-is for now
— nothing has been renamed. Fix opportunistically, or in a dedicated rename
pass if it's ever worth the churn:

- `configs/pipeline_cs_puremath_did_1000_2026-07-27.yaml`,
  `run_cs_puremath_did_1000_*.log` — `puremath` should be `pure_math`, and
  multi-field runs should use `+` (`cs+pure_math`) instead of concatenation,
  which is ambiguous (`cs_health_puremath` could parse as 2 or 3 fields).
- `results/cs_1000_2026-07-27_20260728T003245Z` and siblings — the date is
  duplicated (once from `run.name`, once from the appended run timestamp),
  and the two can disagree, as they do here (07-27 vs 07-28).
- `data/raw/openalex_cs_sanity1`, `configs/pipeline_sanity_check_cs_seed1.yaml`
  — sanity/test runs skip the `-n{n}` and dated-config parts of the template.
- Root-level `run_*.log` files (e.g. `run_cs_1000_buildraw.log`) — should live
  under `results/{run_id}/run_logs/`, matching every other run's logs.
- `scincl_embeddings__malteos-scincl.npy` — mixes `__` and `-` as separators
  with no assigned role for either.
- The `_1000` / `_200` sample-size suffix historically meant "seeds per
  period" but isn't guaranteed to match `sampling.seeds_per_period` in the
  config — see the comment in `configs/pipeline_cs_1000_2026-07-27.yaml`.
  Under this convention `n{n}` should always be read from the config value
  actually used, not typed independently.
