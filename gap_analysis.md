# Gap Analysis

## 1. Overall Assessment

This research direction is still promising, but the broad version is no longer novel. Recent work already shows that LLMs can amplify high-citation bias in synthetic citation generation (P001, P002, P004, P005), that AI-like writing is now measurable in real papers (P010, P011, P012, P013), and that real-world LLM-assisted writing may be associated with citation impact, productivity, disruption, and citation diversity (P015, P016, P018).

The most defensible opportunity is narrower: estimate whether GenAI adoption changed real citation allocation and citation-network inequality across authors, institutions, fields, and venues, while distinguishing a Matthew-effect channel from a democratization channel. The empirical bar is high because AI-use detection is noisy and timing-based designs can be biased (P012, P017).

## 2. Gap-by-Gap Evaluation

### Gap A: Business Venue Gap

**Status:** Partially covered

**Evidence from literature:**

- Business/management venues have started studying GenAI and scholarly writing/reviews: AOM Proceedings uses ChatGPT release as a natural experiment for EFL-authored SSRN papers (P014), Academy of Management Annals discusses LLMs for integrative reviews (P022), and Journal of Information Technology covers GenAI literature reviews (P021).
- Economics/business-adjacent work studies hallucinated citations (P008), and SSRN-based real-world work appears in broader science-production studies (P014, P016).

**Assessment:**

The gap is not simply "business venues are missing." The more accurate gap is that business/management venues have not, in the reviewed evidence, deeply studied LLM-driven citation allocation, prestige bias, author-level visibility, or citation-network evolution.

**Potential research opportunity:**

Study whether GenAI adoption changed citation concentration and attention allocation in business/social-science preprints and journals, using SSRN plus Crossref/OpenAlex.

---

### Gap B: Author-Level Gap

**Status:** Partially covered

**Evidence from literature:**

- P003 analyzes Global East/West and country-level citation-network shifts.
- P014 compares EFL and non-EFL authors with DID.
- P015 controls for authors' prior citation records.
- P016 studies author-background heterogeneity in productivity and citation diversity, but its adoption timing is challenged by P017.

**Assessment:**

Author-level aggregation exists, but not with the full granularity proposed: large labs vs PhD students, seniority, institution prestige, industry vs university, or elite vs non-elite authors.

**Potential research opportunity:**

Estimate whether LLM-assisted writing changes citation gains differently for early-career/non-elite authors versus elite authors, with author fixed effects and pre-trend checks.

---

### Gap C: Real-World Behavior Gap

**Status:** Partially covered

**Evidence from literature:**

- Synthetic experiments dominate the seed citation-bias literature (P001, P002, P004, P005).
- Real-world AI-writing prevalence is now documented at scale (P010, P011, P012, P013).
- Real-world citation-impact and knowledge-organization studies now exist (P003, P015, P016, P018).

**Assessment:**

This gap remains valid only if framed against synthetic citation generation. It is no longer true that real-world behavior is not studied. What is still underdeveloped is the comparison between what LLMs would cite in experiments and what LLM-assisted human papers actually cite.

**Potential research opportunity:**

For the same focal paper/topic, compare synthetic LLM citation recommendations to references in AI-likely human-authored papers, then test whether overlap is biased toward high-citation or high-prestige works.

---

### Gap D: Temporal Network Evolution Gap

**Status:** Partially covered

**Evidence from literature:**

- P003 analyzes country-level pre/post ChatGPT citation-network metrics in Scopus CS papers.
- P018 links AI-assisted writing intensity to disruption and knowledge recombination from 2021-2024.
- P028 is a pre-LLM historical analog showing discovery technologies can narrow citation patterns.

**Assessment:**

Temporal network evolution is emerging but not exhausted. Existing work is either country-level and observational (P003) or structural but not focused on Matthew versus democratization mechanisms (P018).

**Potential research opportunity:**

Build paper-to-paper and author-to-author citation networks before and after November 30, 2022, with treated fields defined by measured AI-writing adoption intensity and control fields with lower adoption.

---

### Gap E: Causal Inference Gap

**Status:** Open / Partially covered

**Evidence from literature:**

- P014 uses DID around ChatGPT release for language proficiency.
- P030 is a strong causal template showing platform-mediated knowledge exposure can affect science.
- P003 makes a partial causal argument but explicitly notes weak instrument and lead-lag limitations.
- P015 uses rich controls and counterfactual ML but is not a clean exogenous design.
- P017 shows naive first-detection event studies can create spurious treatment effects.

**Assessment:**

The causal-inference gap remains strong. The literature has causal designs around writing/language and platform exposure, but not a clean causal estimate of GenAI's effect on citation inequality or citation-network evolution.

**Potential research opportunity:**

Use a DID/event-study design with field-venue cells as units, treatment intensity based on pre-registered or externally measured GenAI adoption, and explicit parallel-trends diagnostics.

---

### Gap F: Matthew Effect vs. Democratization Gap

**Status:** Partially covered

**Evidence from literature:**

- Synthetic evidence strongly supports Matthew-effect amplification: LLMs prefer or better recall highly cited papers (P001, P002, P004, P005).
- Real-world evidence is mixed: P003 suggests modest Eastern visibility gains but persistent hierarchy; P016 reports LLM adopters cite more diverse, younger, less-cited work; P018 finds disruption rises without broader cross-field sourcing.
- Foundational Matthew-effect and search-platform work gives mechanisms but predates LLMs (P026, P027, P028).

**Assessment:**

This is one of the best gaps because the evidence conflicts. Synthetic LLM behavior points toward Matthew amplification, while some real-world human-LLM behavior suggests possible democratization.

**Potential research opportunity:**

Test whether LLM adoption shifts citations toward high-prestige incumbents or toward lower-cited but semantically relevant papers, conditioning on true topical relevance.

---

### Gap G: RAG / Tool-Augmented LLM Citation Behavior Gap

**Status:** Open

**Evidence from literature:**

- The main citation-bias seeds deliberately exclude search/RAG to isolate parametric knowledge (P004, P005).
- RAG and citation-support work exists (P009, P020, P025, P031), but it mainly studies grounding, support, or system performance.
- Existing RAG studies do not appear to ask whether retrieval/search tools reduce or amplify citation prestige bias.

**Assessment:**

This gap is still open and empirically feasible. The key is not "does RAG reduce hallucination?" but "does RAG change which papers receive attention?"

**Potential research opportunity:**

Run controlled experiments comparing parametric LLM, RAG, search-augmented, and citation-recommender conditions on the same topics, measuring relevance-adjusted citation prestige and long-tail visibility.

## 3. Most Promising Gaps

1. **Matthew Effect vs. Democratization in real-world LLM-assisted citation behavior**

Why promising: synthetic and real-world evidence point in different directions (P004, P005 versus P016). Not fully answered because most synthetic studies do not observe final human citations, and real-world studies often infer AI use noisily. Data needed: full-text papers, reference lists, OpenAlex/Semantic Scholar citation histories, author/institution metadata, AI-writing intensity. Causal support: possible with DID/event-study using field-level adoption intensity. Main risks: AI-detection error, true relevance confounding, topic shocks, preprint acceptance changes.

2. **Temporal citation-network evolution after ChatGPT adoption**

Why promising: P003 and P018 start this but leave room for field/venue/author-level network designs. Not fully answered because current work is observational or not focused on inequality mechanisms. Data needed: longitudinal citation networks from 2019-2026, field/venue panels, author/institution prestige, arXiv/open-access indicators. Causal support: possible if treated fields with high AI adoption can be matched to low-adoption controls with parallel trends. Main risks: citation lags, changing publication volume, field-specific citation norms.

3. **RAG/tool augmentation and relevance-adjusted citation inequality**

Why promising: most core citation-bias papers exclude RAG (P004, P005), while RAG work emphasizes grounding rather than attention allocation (P009, P031). Not fully answered because hallucination reduction is not the same as bias reduction. Data needed: benchmark topics with gold relevance sets, citation/prestige metrics, multiple model/tool conditions. Causal support: strong in randomized synthetic experiments; weaker but possible in field deployments. Main risks: constructing a valid relevance ground truth and external validity to real researchers.

## 4. Recommended Next Step

Pursue Gap F first: **Matthew Effect vs. democratization in real-world LLM-assisted citation behavior**.

Draft research question: **Did the adoption of GenAI writing tools after November 30, 2022 shift citation attention toward already high-status papers and authors, or did it increase visibility for less-cited but semantically relevant work?**

Possible unit of analysis: citing-paper to cited-paper pair, aggregated to paper-year, author-year, and field-venue-year panels.

Possible dataset: OpenAlex or Semantic Scholar citation graph plus arXiv/SSRN/preprint metadata, full-text or abstracts for AI-writing intensity, author disambiguation, institution prestige, venue rank, open-access/arXiv indicators, publication age, and topic embeddings.

Possible identification strategy: DID/event study comparing high-GenAI-adoption fields or venues with lower-adoption controls before and after November 30, 2022; supplement with GPT-4 release on March 14, 2023 and GPT-4o release on May 13, 2024 as secondary temporal shocks. Include field-by-year and author fixed effects where feasible, and test parallel trends explicitly.

Preliminary experiment to run first: take one field with strong adoption, such as CS, and one lower-adoption comparison field; compute whether AI-likely papers cite lower- or higher-citation references than matched non-AI-likely papers after controlling for semantic relevance between citing and cited paper abstracts, cited-paper age, venue, open-access status, and author prestige.
