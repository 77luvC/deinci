# Demo SciNCL POC Report

## Dataset

- Number of scored pairs: 120
- Number of query papers: 20
- Number of positive pairs: 60
- Number of negative pairs: 60
- SciNCL model version: `malteos/scincl`

## Pair Metrics

- n_pairs: 120
- n_positive: 60
- n_negative: 60
- mean_positive_score: 0.8725359658400218
- mean_negative_score: 0.6972044398387273
- positive_negative_diff: 0.17533152600129454
- cohens_d: 2.518875692425268
- roc_auc: 0.9508333333333333
- average_precision: 0.9547315286333741

## Ranking Metrics Mean

- n_candidates: 6.0
- n_positive: 3.0
- mrr: 0.975
- recall@10: 1.0
- ndcg@10: 0.9776612234513807
- recall@50: 1.0
- ndcg@50: 0.9776612234513807
- recall@100: 1.0
- ndcg@100: 0.9776612234513807
- recall@300: 1.0
- ndcg@300: 0.9776612234513807

## Limitations

SciNCL is used here as a citation-informed relevance proxy, not as a citation quality judge. A high SciNCL score suggests topical or scholarly relatedness between two papers, but it does not prove that a citation supports a specific claim, uses the best available evidence, or is non-hallucinated. The pretrained SciNCL model is based on historical citation information from S2ORC 20200705v1, so scores for post-2020 and GenAI-era papers should be interpreted with caution.
