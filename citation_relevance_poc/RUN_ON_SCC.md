# Run On SCC

This folder is a clean upload package for rerunning the citation pipeline from
scratch on SCC. It intentionally does not include local `data/` or `results/`
outputs; those folders will be created/populated by the run.

## 1. Upload And Unpack

```bash
tar -xzf citation_relevance_poc_scc_2026-07-27.tar.gz
cd citation_relevance_poc_scc_2026-07-27
```

## 2. Create Or Activate Environment

Use your SCC Python environment. One common pattern is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For SciNCL relevance scoring, request a GPU node if available; CPU runs are
possible but slower.

## 3. Configure Secrets

Do not write API keys into the YAML file. Export them in the shell:

```bash
export OPENALEX_API_KEY="..."
```

Optionally edit `configs/pipeline_2026-07-27.yaml` to set `openalex.mailto`.

## 4. Check Commands

```bash
DRY_RUN=1 bash scripts/scc_run_configured_pipeline.sh
```

## 5. Run

```bash
bash scripts/scc_run_configured_pipeline.sh
```

Useful partial reruns:

```bash
PIPELINE_FIELD="openalex_cs_200" bash scripts/scc_run_configured_pipeline.sh
PIPELINE_ONLY="historical_scores" PIPELINE_FIELD="openalex_cs_200" bash scripts/scc_run_configured_pipeline.sh
PIPELINE_SKIP="build_raw load_pairs" bash scripts/scc_run_configured_pipeline.sh
```

After partial field runs, rebuild aggregate outputs:

```bash
PIPELINE_ONLY="aggregate_core plot_core" bash scripts/scc_run_configured_pipeline.sh
```
