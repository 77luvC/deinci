# Metrics Registry

本文档固化当前 Citation Relevance POC 的分析维度、指标定义、计算公式和落盘位置。这里的“当前口径”以 `src/` 中 pipeline 代码为准；历史结果文件可能仍保留旧列名或旧口径，见文末“口径注意事项”。

## 研究对象

本项目把 citation system 拆成三层分析单位：

| 分析单位 | 主键 | 含义 | 主要输出 |
|---|---|---|---|
| pair-level | `query_id`, `candidate_id` | 一篇种子论文和一个候选/被引论文构成的一条引用候选边 | pair scores, retrieval eval, temporal metrics |
| seed-level | `query_id` | 一篇种子论文实际引用集合的聚合指标 | `seed_democratization_metrics.csv` |
| group-level | `query_ai_group`, `dataset`, etc. | 按时期、领域、数据集等维度聚合后的比较结果 | temporal, democratization summary, DID outputs |

基本符号：

| 符号 | 定义 |
|---|---|
| `i` | seed/query/citing paper，即发出引用的论文 |
| `j` | candidate/cited paper，即候选或实际被引论文 |
| `R_i` | seed paper `i` 的实际 OpenAlex reference 集合 |
| `C_i` | seed paper `i` 的 candidate set；主要用于 retrieval/ranking metrics |
| `s_ij` | seed paper `i` 与 candidate paper `j` 的 SciNCL embedding cosine similarity |
| `y_i`, `y_j` | seed paper `i` 和 candidate paper `j` 的 publication year |
| `c_ij` | candidate paper `j` 截至 seed paper `i` 发表年份 cutoff 的 historical visibility citation count |
| `label_ij` / `label` | pair label；`1` 表示 `i` 实际引用 `j`，`0` 表示负样本/比较候选 |

## 核心维度

| 维度 | 字段 | 取值/定义 | 生成位置 | 用途 |
|---|---|---|---|---|
| AI era | `query_ai_group` | `pre_ai`: 2019-01-01 至 2021-12-31；`post_ai`: 2023-01-01 至 2026-06-30；其他为 `outside_window` 或 `unknown` | `add_temporal_groups` | 主时期比较 |
| Broad period | `query_period` | `pre_2020` if `query_year <= 2020`; else `post_2020` | `add_temporal_groups` | 早期 temporal sanity check |
| GenAI topic | `query_genai_topic` | `genai_topic` if `query_is_genai` else `non_genai_topic` | `add_temporal_groups` | 主题异质性 |
| Combined group | `combined_group` | `query_period + "_" + query_ai_group` | `add_temporal_groups` | 交叉分组 |
| Field | `query_field`, `candidate_field` | OpenAlex primary topic field display name | pair scoring / pair metadata / metadata merge | field match, cross-field share |
| Same field | `same_field` | `query_field == candidate_field` | temporal/cited distribution metadata | field homophily |
| Dataset/field sample | `dataset`, `source_dataset` | e.g. `computer_science`, `broad_math`, `openalex_expanded` | builders / cross-field DID | DID treatment/control grouping |
| Candidate work type | `candidate_work_type` | OpenAlex work type, e.g. `article`, `book`, `preprint` | metadata merge | reference composition |
| Candidate venue | `candidate_venue` | OpenAlex primary source display name, fallback `unknown` | metadata merge | venue concentration/composition |
| Candidate OA | `candidate_is_oa` | OpenAlex open access boolean | metadata merge | reference openness |
| Seed citation stratum | `seed_citation_stratum` | `low`, `medium`, `high` tertiles of seed cited-by count at sampling time | OpenAlex builders | sampling balance, robustness |

## Shared Derived Fields

| 字段 | 定义/公式 | 说明 |
|---|---|---|
| `scincl_cosine` | `s_ij = dot(emb_i, emb_j)` | Embeddings 来自 `malteos/scincl`，当前实现假设向量已归一化，因此点积即 cosine proxy |
| `scincl_cosine = NA` | 未计算 embeddings 的 pair metadata 占位值 | `run_pair_metadata.py` 可生成无 SciNCL 分数的 pair-level metadata；此时 relevance metrics 条件性缺失 |
| `candidate_age_at_citation` | `y_i - y_j` | 负值会在 seed-level age metric 中视为无效 |
| `valid_cited_age` | `candidate_age_at_citation` if `>= 0`, else null | 用于避免 future-dated cited works 影响 age 指标 |
| `candidate_current_cited_by_count` | OpenAlex 当前 `cited_by_count` | 当前总引用数，会有未来信息穿越风险 |
| `candidate_visibility_cited_by_count` | 优先 `candidate_cited_by_count_at_query_year`；若没有 historical count，则 fallback 为 current cited-by count | 推荐用于所有 visibility / prestige 指标 |
| `candidate_visibility_source` | `citation_year` 或 `current_fallback` | 标识 visibility 指标来自 historical count 还是 current cited-by fallback |
| `candidate_log1p_visibility_cited_by_count` | `log(1 + candidate_visibility_cited_by_count)` | 降低重尾分布影响 |
| `candidate_cited_by_count_at_query_year` | historical count through query-year cutoff；actual reference pair 再减去 focal citation 自身 | 由 `run_historical_citation_counts.py` 生成 |

Historical citation cutoff rule：

| 方法 | 公式/规则 |
|---|---|
| exact | OpenAlex query `filter=cites:{candidate_id},to_publication_date:{cutoff_date}`，`cutoff_date = min(query publication year end, 2026-06-30)` |
| yearly | `current_cited_by_count - sum(counts_by_year for years after query publication year)` |
| focal-pair adjustment | 若 `label_ij == 1`（代码列名为 `label`），`candidate_cited_by_count_at_query_year = max(raw_count - 1, 0)` |

## Pair-Level Relevance Metrics

这些指标评估 `scincl_cosine` 能否把实际引用边和比较候选区分开。注意：如果一个数据集只有正样本，区分性指标会是 `NaN`。

| 指标 | 定义 | 公式 | 输出 |
|---|---|---|---|
| `n_pairs` | pair 总数 | `N` | `metrics_summary.csv`, grouped temporal outputs |
| `n_positive` | actual reference pair 数 | `sum(label_ij == 1)` | same |
| `n_negative` | negative/comparison pair 数 | `sum(label_ij == 0)` | same |
| `mean_positive_score` | 正样本平均 SciNCL | `mean(s_ij | label_ij=1)` | same |
| `mean_negative_score` | 负样本平均 SciNCL | `mean(s_ij | label_ij=0)` | same |
| `positive_negative_diff` | 正负样本均值差 | `mean_positive_score - mean_negative_score` | same |
| `cohens_d` | 标准化正负均值差 | `(mean_pos - mean_neg) / sqrt(pooled_var)` | same |
| `roc_auc` | ROC AUC | `roc_auc_score(label_ij, s_ij)` | same |
| `average_precision` | Average Precision | `average_precision_score(label_ij, s_ij)` | same |

Interpretation：

| 指标方向 | 含义 |
|---|---|
| 更高 `mean_positive_score` | 实际引用边在 SciNCL 空间中更相近 |
| 更高 `positive_negative_diff`, `cohens_d`, `roc_auc`, `average_precision` | SciNCL relevance proxy 对实际引用 vs. 比较候选的区分能力更强 |
| `NaN` | 通常表示没有负样本，不能估计区分性 |

## Ranking Metrics

按每个 `query_id` 内部用 `scincl_cosine` 从高到低排序，衡量实际引用边在候选列表中的位置。

| 指标 | 定义 | 公式 | 输出 |
|---|---|---|---|
| `n_candidates` | seed paper 的候选数 | `|C_i|` | `ranking_metrics.csv` |
| `n_positive` | seed paper 的实际引用数 | `|{j in C_i: label_ij=1}|` | same |
| `mrr` | 第一条实际引用的 reciprocal rank | `1 / rank(first positive)` | same |
| `recall@k` | top-k 覆盖的实际引用比例 | `sum_{j in top-k(C_i)} label_ij / sum_{j in C_i} label_ij` | same |
| `ndcg@k` | binary-label NDCG | `DCG@k / IDCG@k`, `DCG@k=sum_r label_r/log2(r+1)` for 1-indexed rank `r` | same |

当前默认 `k = 10, 50, 100, 300`。

## Temporal / Grouped Relevance Metrics

`run_temporal_analysis.py` 对 pair-level relevance metrics 做分组聚合：

| 输出文件 | 分组维度 | 指标集合 |
|---|---|---|
| `temporal_metrics.csv` | `query_period` | pair-level relevance metrics |
| `ai_era_metrics.csv` / `genai_metrics.csv` | `query_ai_group` | pair-level relevance metrics |
| `genai_topic_metrics.csv` | `query_genai_topic` | pair-level relevance metrics |
| `combined_group_metrics.csv` | `combined_group` | pair-level relevance metrics |

Optional DID-style regression：

```text
scincl_cosine ~ post_2020 + post_ai + post_2020:post_ai + label + same_field_int
```

其中：

| term | 含义 |
|---|---|
| `post_2020` | query in broad post-2020 period |
| `post_ai` | query in GenAI-era post window |
| `post_2020:post_ai` | broad period 与 AI-era 的交互项 |
| `label` | actual reference vs. comparison pair |
| `same_field_int` | query/candidate 是否同 field |

## Cited Distribution Metrics

这些指标只看 actual references，即 `label == 1`。当前代码按 `query_ai_group` 聚合。
如果输入来自 `run_pair_metadata.py` 而不是 `run_pair_scoring.py`，`mean_scincl` 和 `median_scincl` 会保留为 `NaN`。

| 指标 | 定义 | 公式 | 输出 |
|---|---|---|---|
| `n_reference_pairs` | 实际引用 pair 数 | `|{(i,j): label_ij=1}|` | `cited_distribution_summary.csv` |
| `n_query_papers` | 有实际引用的 seed paper 数 | `nunique(query_id)` | same |
| `mean_scincl` | 实际引用边平均 SciNCL | `mean(s_ij)` | same |
| `median_scincl` | 实际引用边 SciNCL 中位数 | `median(s_ij)` | same |
| `mean_cited_age` | 被引文献年龄均值 | `mean(y_i - y_j)` | same |
| `median_cited_age` | 被引文献年龄中位数 | `median(y_i - y_j)` | same |
| `mean_visibility_cited_by_count` | 被引文献 visibility 均值 | `mean(c_ij)` | same |
| `median_visibility_cited_by_count` | 被引文献 visibility 中位数 | `median(c_ij)` | same |
| `p25_visibility_cited_by_count` | 被引文献 visibility 25 分位 | `quantile(c_ij, .25)` | same |
| `p75_visibility_cited_by_count` | 被引文献 visibility 75 分位 | `quantile(c_ij, .75)` | same |
| `mean_log1p_visibility_cited_by_count` | log visibility 均值 | `mean(log(1 + c_ij))` | same |
| `gini_visibility_cited_by_count` | 被引文献 visibility Gini | `((2 * sum_k k*x_k) / (n*sum_k x_k)) - ((n+1)/n)`, `x` sorted ascending | same |
| `top10pct_share_of_visibility_citation_mass` | top 10% 被引文献占据的 citation mass | `sum(top ceil(.10*n) c_ij) / sum_all c_ij` | same |
| `top25pct_share_of_visibility_citation_mass` | top 25% 被引文献占据的 citation mass | `sum(top ceil(.25*n) c_ij) / sum_all c_ij` | same |
| `share_open_access` | OA 被引文献比例 | `mean(candidate_is_oa)` | same |
| `share_same_field` | 同 field 引用比例 | `mean(query_field == candidate_field)` | same |
| `share_article` | article 类型被引文献比例 | `mean(candidate_work_type == "article")` | same |
| `share_book` | book 类型被引文献比例 | `mean(candidate_work_type == "book")` | same |

Distribution tables：

| 输出文件 | 定义 |
|---|---|
| `cited_visibility_quartiles.csv` | 将 actual references 按全样本 `candidate_visibility_cited_by_count` rank 分成 Q1-Q4，再按 `query_ai_group` 统计比例 |
| `cited_work_type_distribution.csv` | 按 `query_ai_group` x `candidate_work_type` 统计 `n` 和 `share` |
| `cited_field_distribution.csv` | 按 `query_ai_group` x `candidate_field` 统计 `n` 和 `share` |
| `cited_venue_distribution.csv` | 按 `query_ai_group` x `candidate_venue` 统计 `n` 和 `share` |

## Seed-Level Democratization Metrics

这些是当前最重要的研究指标，分析单位是一篇 seed paper `i` 的 actual reference set `R_i`。默认只比较 `pre_ai` 和 `post_ai`。
当前代码把指标分为 always-on 的 `BASE_METRICS` 和条件性出现的 `RELEVANCE_METRICS`：只有输入中存在可用的 numeric `scincl_cosine` 时，`mean_scincl` 和 `share_relevant_low_citation_refs` 才会出现在 seed-level 输出、summary、comparisons 和 metric definitions 中。`median_cited_age` 可能仍出现在部分 pipeline 输出中，但不再作为当前论文叙事的核心指标。

| 指标 | 维度含义 | 定义 | 公式 |
|---|---|---|---|
| `n_reference_pairs` | coverage | seed paper `i` 可用实际引用数 | `|R_i|` |
| `share_recent_refs_3yr` | recency | 引用 3 年内文献的比例 | `mean_{j in R_i}(0 <= y_i - y_j <= 3)` |
| `median_log1p_cited_by_count` | visibility / prestige | 被引文献 citation-year visibility 的 log 中位数 | `median_{j in R_i}(log(1 + c_ij))` |
| `share_low_citation_refs` | long-tail access | 低引用文献比例 | `mean_{j in R_i}(c_ij <= T_low)` |
| `top10pct_share_of_citation_mass` | concentration | seed 的引用 visibility mass 被 top 10% references 占据的比例 | `sum_{j in Top10_i} c_ij / sum_{j in R_i} c_ij` |
| `share_cross_field_refs` | interdisciplinarity | 跨 OpenAlex field 引用比例 | `mean_{j in R_i}(field_j != field_i)` |
| `cited_field_entropy` | diversity | 被引 field 分布的 Shannon entropy | `-sum_f p_if * log(p_if)` |
| `mean_scincl` | relevance proxy | 实际引用边平均 SciNCL similarity | `mean_{j in R_i}(s_ij)` |
| `share_relevant_low_citation_refs` | relevance-adjusted long tail | 同时低引用且 SciNCL 高相关的引用比例 | `mean_{j in R_i}(c_ij <= T_low and s_ij >= T_rel)` |

其中 `Top10_i` 是 seed paper `i` 的 references 中按 `c_ij` 排名前 `ceil(.10*|R_i|)` 的 candidate paper 集合；`p_if` 是 `R_i` 中属于 field `f` 的 references 占比。

Thresholds：

| 参数 | 默认值 | 定义 |
|---|---|---|
| `T_low` | `actual["candidate_visibility_cited_by_count"].quantile(0.25)` | 当前分析样本 actual references 的全局 bottom quartile cutoff |
| `T_rel` | `0.80` | SciNCL cosine relevance threshold |

Seed-level outputs：

| 输出文件 | 内容 |
|---|---|
| `seed_democratization_metrics.csv` | 每个 `query_id` 的 seed-level metrics |
| `period_democratization_summary.csv` | 每个 `query_ai_group` x `metric` 的 `n_seed_papers`, `mean`, `median`, `std`, `sem` |
| `period_democratization_comparisons.csv` | `post_ai` vs `pre_ai` 的均值差、中位数差、Welch t-test、Mann-Whitney U test、BH q-values |
| `metric_definitions.csv` | pipeline 自动输出的短定义和本次阈值 |

Interpretation toward Matthew effect vs. democratization：

| 指标方向 | 更接近 Matthew effect | 更接近 democratization |
|---|---|---|
| `median_log1p_cited_by_count` | 上升 | 下降 |
| `share_low_citation_refs` | 下降 | 上升 |
| `top10pct_share_of_citation_mass` | 上升 | 下降 |
| `share_relevant_low_citation_refs` | 下降 | 上升 |
| `cited_field_entropy` | 下降或集中 | 上升 |
| `share_cross_field_refs` | 下降或同质化 | 上升 |
| `mean_scincl` | 需和 visibility 一起看；单独更高不等于更民主 | 若 long-tail 指标改善且 `mean_scincl` 不下降，支持“相关的长尾发现” |

## Cross-Field DID Metrics

`run_cross_field_did_analysis.py` 把多个 seed-level metric 文件合并，按 treatment dataset 和 control dataset 做 DID。
DID 会自动使用输入 seed metrics 中实际存在的 `BASE_METRICS + RELEVANCE_METRICS`，因此没有 SciNCL 的 field-scan 数据不会输出 relevance DID。

| 指标/字段 | 定义 | 公式 |
|---|---|---|
| `treatment_post_mean` | treatment dataset 在 post period 的 metric mean | `mean(metric | dataset=treat, period=post)` |
| `treatment_base_mean` | treatment dataset 在 base period 的 metric mean | `mean(metric | dataset=treat, period=base)` |
| `control_post_mean` | control dataset 在 post period 的 metric mean | `mean(metric | dataset=control, period=post)` |
| `control_base_mean` | control dataset 在 base period 的 metric mean | `mean(metric | dataset=control, period=base)` |
| `treatment_change` | treatment 前后变化 | `treatment_post_mean - treatment_base_mean` |
| `control_change` | control 前后变化 | `control_post_mean - control_base_mean` |
| `did_estimate` | difference-in-differences estimate | `(treatment_post - treatment_base) - (control_post - control_base)` |
| `interaction_coef` | OLS 交互项系数 | coefficient on `treated * post` |
| `interaction_t` | OLS 交互项 t-stat | `coef / se` |
| `interaction_p_value` | 双侧 p-value | `2 * t.sf(abs(t), df_resid)` |
| `interaction_q_value_bh` | BH 校正后的 q-value | Benjamini-Hochberg over interaction p-values |

OLS implementation：

```text
metric ~ 1 + treated + post + treated:post
```

当前 contrasts：

| post period | base period |
|---|---|
| `post_ai` | `pre_ai` |

部分历史输出可能包含 `transition` 作为 base period；当前代码只固化 `post_ai` vs `pre_ai`。

## Output Map

| 文件路径模式 | 负责 pipeline | 分析单位 | 主要内容 |
|---|---|---|---|
| `results/*_pair_scores.parquet` | `run_pair_scoring.py` | pair | SciNCL scores + query/candidate metadata |
| `results/*_pair_metadata.parquet` | `run_pair_metadata.py` | pair | query/candidate metadata with `scincl_cosine = NA` |
| `results/*_pair_scores_historical.parquet` | `run_historical_citation_counts.py` | pair | pair scores + citation-year visibility counts |
| `results/*_pair_metadata_historical.parquet` | `run_historical_citation_counts.py` | pair | pair metadata + citation-year visibility counts, without usable SciNCL scores |
| `results/*_eval/metrics_summary.csv` | `run_retrieval_eval.py` | pair aggregate | pair-level relevance metrics |
| `results/*_eval/ranking_metrics.csv` | `run_retrieval_eval.py` | query | MRR, recall@k, NDCG@k |
| `results/*_temporal/*.csv` | `run_temporal_analysis.py` | grouped pair | relevance metrics by temporal dimensions |
| `results/*_cited_distribution/*.csv` | `run_cited_distribution_analysis.py` | grouped actual references | visibility, composition, field/venue/work-type distributions |
| `results/*_democratization/*.csv` | `run_democratization_analysis.py` | seed/group | seed-level democratization metrics and post/pre comparisons |
| `results/cs_math_did_democratization/*.csv` | `run_cross_field_did_analysis.py` | dataset-period-metric | cross-field DID summaries and estimates |
| `results/field_scan_200_summary_comparisons.csv` | field-scan aggregation / plotting workflow | dataset-metric | post-AI minus pre-AI differences plus Welch and Mann-Whitney p-values |
| `results/field_scan_200_cited_distribution_summary.csv` | field-scan aggregation workflow | dataset-group | compact cited-distribution summary: refs, queries, visibility, age, same-field share |
| `results/field_scan_200_figures_python/*` | `scripts/plot_field_scan_core_metrics.py` | figure | grouped plots for visibility, recency, breadth, and all-field heatmap |

Field-scan plotting metrics:

| metric | group | displayed unit |
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
| Are GenAI-era papers opening attention to the long tail? | `share_low_citation_refs`, `share_relevant_low_citation_refs` | `mean_scincl`, `median_scincl` |
| Is attention becoming more concentrated or diverse? | `cited_field_entropy`, `share_cross_field_refs`, venue distribution | work type distribution, OA share |
| Are differences driven by recency? | `share_recent_refs_3yr` | query year/period stratification |
| Is CS changing differently from math/control fields? | DID estimates for the seed-level democratization metrics | interaction p/q-values |

## 口径注意事项

1. `SciNCL` 是 citation-informed relevance proxy，不是 citation quality judge。高分表示文本/学术相关性更强，但不能证明引用支持具体论点、不是装饰性引用、不是 hallucination。
2. `malteos/scincl` 训练基于历史语料，post-2020 和 GenAI-era 的分数需要谨慎解释。
3. Citation visibility 指标应优先使用 `candidate_visibility_cited_by_count`。如果输入 pair scores 没有 historical count 字段，代码会 fallback 到 OpenAlex 当前总引用数。
4. 历史落盘结果中可能出现 `current_cited_by_count` 列名；当前代码已迁移到 `visibility_cited_by_count` 口径。后续正式跑数建议统一使用 historical enriched pair scores。
5. 当前 expanded/math/pure_math 以及多数 200-sample OpenAlex field 文件只有 `label == 1`，因此 `roc_auc`, `average_precision`, `cohens_d`, `positive_negative_diff` 不能解释为有效区分性。
6. `T_low` 是样本内全局 bottom quartile，不是跨数据集固定阈值。跨 field DID 时，要确认各 dataset 的 threshold 是否应统一或分开估计。
7. `pre_ai` 和 `post_ai` 中间的 2022 被排除，是为了避免 ChatGPT 发布前后过渡期混入处理组或控制组。
8. 当前 OpenAlex observation end 固定为 2026-06-30；如果更新数据窗口，historical cutoff rule 和 AI-era end date 需要同步更新。
