from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_FIELD_ID = "17"
DEFAULT_SOURCE_DATASET = "openalex_cs_200"

# The field-scan builder creates a field-scoped OpenAlex corpus with balanced
# pre/post AI-era seed papers. It preserves all usable actual references for
# those seeds; no negative pairs are sampled in this builder.

TIME_BANDS = {
    "pre_ai": ("2019-01-01", "2021-12-31"),
    "post_ai": ("2023-01-01", "2026-06-30"),
}

# These windows intentionally match the rest of the analysis pipeline and omit
# 2022 as a transition year.

QUERY_SELECT = ",".join(
    [
        "id",
        "title",
        "display_name",
        "publication_year",
        "publication_date",
        "language",
        "type",
        "primary_location",
        "open_access",
        "cited_by_count",
        "referenced_works",
        "referenced_works_count",
        "abstract_inverted_index",
        "primary_topic",
        "topics",
    ]
)

# Query works need reference lists and seed-sampling metadata. Keeping a fixed
# select list reduces payload size and makes output schemas more predictable.

REFERENCE_SELECT = ",".join(
    [
        "id",
        "title",
        "display_name",
        "publication_year",
        "publication_date",
        "language",
        "type",
        "primary_location",
        "open_access",
        "cited_by_count",
        "abstract_inverted_index",
        "primary_topic",
        "topics",
    ]
)

# Referenced works do not need `referenced_works`, so this lighter select list is
# used when hydrating candidate/cited-paper metadata.


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    """Rebuild an OpenAlex inverted-index abstract into plain text.

    Missing indexes return an empty string. Callers use that to screen out works
    that cannot be embedded by the downstream SciNCL encoder.
    """
    if not index:
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in index.items():
        positioned.extend((position, token) for position in positions)
    return " ".join(token for _, token in sorted(positioned))


def openalex_short_id(openalex_url: str) -> str:
    """Extract a compact OpenAlex work ID from a URL or already-short ID string."""
    return openalex_url.rstrip("/").split("/")[-1]


def source_name(work: dict[str, Any]) -> str:
    """Return the primary venue/source display name for an OpenAlex work."""
    source = (work.get("primary_location") or {}).get("source") or {}
    return source.get("display_name") or "unknown"


def work_field_name(work: dict[str, Any]) -> str:
    """Return the best available OpenAlex field display name for a work.

    Primary-topic field metadata is preferred because it is the canonical field
    assignment. The first topic's field is used as a fallback for older or sparse
    records; missing field data becomes `unknown`.
    """
    primary_topic = work.get("primary_topic") or {}
    primary_field = primary_topic.get("field") or {}
    if primary_field.get("display_name"):
        return primary_field["display_name"]

    topics = work.get("topics") or []
    if topics:
        field = topics[0].get("field") or {}
        if field.get("display_name"):
            return field["display_name"]
    return "unknown"


def paper_row(
    work: dict[str, Any],
    *,
    source_dataset: str,
    ai_period: str | None = None,
    is_seed: bool = False,
    seed_citation_stratum: str | None = None,
    seed_reference_count: int | None = None,
    seed_usable_reference_count: int | None = None,
) -> dict[str, Any]:
    """Flatten an OpenAlex work into the raw paper schema plus seed metadata.

    The caller marks whether the work is a sampled seed and may attach period,
    citation-stratum, and reference-count information. Optional OpenAlex fields
    are allowed to be missing; the required assumption is that `work["id"]`
    exists.
    """
    field = work_field_name(work)
    return {
        "paper_id": openalex_short_id(work["id"]),
        "title": work.get("title") or work.get("display_name") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "work_type": work.get("type"),
        "language": work.get("language"),
        "venue": source_name(work),
        "field": field,
        "is_oa": (work.get("open_access") or {}).get("is_oa"),
        "openalex_current_cited_by_count": work.get("cited_by_count"),
        "ai_period": ai_period,
        "is_seed": is_seed,
        "seed_citation_stratum": seed_citation_stratum,
        "seed_reference_count": seed_reference_count,
        "seed_usable_reference_count": seed_usable_reference_count,
        "source_dataset": source_dataset,
    }


def get_json(params: dict[str, Any], *, mailto: str | None, sleep_seconds: float) -> dict[str, Any]:
    """Call OpenAlex works with polite-pool metadata and return decoded JSON.

    `mailto` and `OPENALEX_API_KEY` are attached when available. HTTP failures
    are allowed to raise because downstream CSVs should not be written from a
    silently partial response.
    """
    if mailto:
        params = {**params, "mailto": mailto}
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params = {**params, "api_key": api_key}
    response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    response.raise_for_status()
    time.sleep(sleep_seconds)
    return response.json()


def date_filter(
    from_date: str | None,
    to_date: str | None,
    field_id: str | None,
    subfield_ids: list[str],
    language: str | None,
) -> str:
    """Build the OpenAlex filter string for seed-candidate queries.

    Either `field_id` or one or more `subfield_ids` must be supplied. Subfields
    take precedence because they are more specific. Date and language filters
    are optional, but all candidates are required to be articles with abstracts
    and references.
    """
    parts = ["has_abstract:true", "has_references:true", "type:article"]
    if subfield_ids:
        parts.append(f"primary_topic.subfield.id:{'|'.join(subfield_ids)}")
    elif field_id:
        parts.append(f"primary_topic.field.id:{field_id}")
    else:
        raise ValueError("Either field_id or subfield_ids is required.")
    if language:
        parts.append(f"language:{language}")
    if from_date:
        parts.append(f"from_publication_date:{from_date}")
    if to_date:
        parts.append(f"to_publication_date:{to_date}")
    return ",".join(parts)


def usable_query_work(work: dict[str, Any], min_references: int) -> bool:
    """Return whether an OpenAlex work can serve as a seed/query paper.

    A usable seed needs a title/display name, a reconstructable abstract, and at
    least `min_references` listed references. This check intentionally ignores
    downstream reference hydration, which may still remove some references.
    """
    if not (work.get("title") or work.get("display_name")):
        return False
    if not reconstruct_abstract(work.get("abstract_inverted_index")):
        return False
    return len(work.get("referenced_works") or []) >= min_references


def fetch_seed_candidates(
    *,
    ai_period: str,
    from_date: str | None,
    to_date: str | None,
    field_id: str | None,
    subfield_ids: list[str],
    language: str | None,
    candidate_sample_size: int,
    seed: int,
    min_references: int,
    mailto: str | None,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    """Fetch candidate seed works for one AI-era date band.

    OpenAlex random sampling is controlled by `candidate_sample_size` and
    `seed`. Results are de-duplicated and filtered locally for usable text and
    reference count. The function may return fewer candidates than requested if
    OpenAlex has sparse results for the selected field/window.
    """
    works: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_page = 200
    pages = max(1, (candidate_sample_size + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        # The same sample size is requested across pages so OpenAlex's seeded
        # sampling behavior remains deterministic for a fixed configuration.
        params = {
            "filter": date_filter(from_date, to_date, field_id, subfield_ids, language),
            "sample": candidate_sample_size,
            "seed": seed,
            "per-page": per_page,
            "page": page,
            "select": QUERY_SELECT,
        }
        results = get_json(params, mailto=mailto, sleep_seconds=sleep_seconds).get("results", [])
        if not results:
            break
        for work in results:
            work_id = openalex_short_id(work["id"])
            if work_id in seen or not usable_query_work(work, min_references):
                continue
            work["_ai_period"] = ai_period
            works.append(work)
            seen.add(work_id)
    return works


def citation_strata(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Assign seed candidates to low/medium/high citation strata.

    Strata are based on ranks of current OpenAlex cited-by counts. Ranking with
    `method="first"` handles ties deterministically by candidate order, which
    avoids qcut failures when many works share the same citation count.
    """
    rows = []
    for idx, work in enumerate(candidates):
        rows.append(
            {
                "idx": idx,
                "paper_id": openalex_short_id(work["id"]),
                "cited_by_count": int(work.get("cited_by_count") or 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df.assign(stratum=pd.Series(dtype=str))
    labels = ["low", "medium", "high"]
    df["stratum"] = pd.qcut(
        df["cited_by_count"].rank(method="first"),
        q=3,
        labels=labels,
    ).astype(str)
    return df


def stratum_quotas(total: int) -> dict[str, int]:
    """Split a target seed count as evenly as possible across three strata."""
    labels = ["low", "medium", "high"]
    base = total // len(labels)
    remainder = total % len(labels)
    return {label: base + (1 if idx < remainder else 0) for idx, label in enumerate(labels)}


def select_stratified_seeds(
    candidates: list[dict[str, Any]],
    *,
    seeds_per_period: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample a fixed number of seeds across citation strata.

    Raises when the candidate pool is too small. If a stratum cannot supply its
    full quota, remaining seeds are backfilled from other strata so the requested
    period count is still reached when enough total candidates exist.
    """
    strata = citation_strata(candidates)
    if len(strata) < seeds_per_period:
        raise ValueError(
            f"Only {len(strata)} usable seed candidates found, but {seeds_per_period} are required. "
            "Increase --candidate-sample-size or relax --min-references."
        )

    selected_indices: list[int] = []
    quotas = stratum_quotas(seeds_per_period)
    for stratum, quota in quotas.items():
        # Shuffle within each stratum using the provided RNG so sampling remains
        # reproducible across periods and Python process runs.
        stratum_indices = strata.loc[strata["stratum"] == stratum, "idx"].tolist()
        rng.shuffle(stratum_indices)
        selected_indices.extend(stratum_indices[:quota])

    if len(selected_indices) < seeds_per_period:
        # Backfill from all unselected candidates when one citation stratum is
        # undersupplied after qcut assignment.
        selected = set(selected_indices)
        remaining = [idx for idx in strata["idx"].tolist() if idx not in selected]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[: seeds_per_period - len(selected_indices)])

    selected_indices = selected_indices[:seeds_per_period]
    stratum_by_idx = dict(zip(strata["idx"], strata["stratum"], strict=True))
    selected = []
    for idx in selected_indices:
        work = candidates[idx]
        work["_seed_citation_stratum"] = stratum_by_idx[idx]
        selected.append(work)
    return selected


def fetch_works_by_ids(
    openalex_ids: list[str],
    *,
    language: str | None,
    mailto: str | None,
    sleep_seconds: float,
) -> dict[str, dict[str, Any]]:
    """Fetch referenced works by OpenAlex ID and retain embeddable records.

    IDs are normalized to short work IDs before lookup. Optional language
    filtering mirrors the seed filter; works without title or abstract are
    omitted because they cannot be encoded consistently.
    """
    out: dict[str, dict[str, Any]] = {}
    ids = [openalex_short_id(work_id) for work_id in openalex_ids]
    for start in range(0, len(ids), 50):
        # Keep ID filters in moderate chunks to stay within URL/query-size
        # limits while preserving deterministic ordering.
        chunk = ids[start : start + 50]
        params = {
            "filter": f"openalex_id:{'|'.join(chunk)}",
            "per-page": len(chunk),
            "select": REFERENCE_SELECT,
        }
        for work in get_json(params, mailto=mailto, sleep_seconds=sleep_seconds).get("results", []):
            if language and work.get("language") != language:
                continue
            if work.get("title") and reconstruct_abstract(work.get("abstract_inverted_index")):
                out[openalex_short_id(work["id"])] = work
    return out


def build_openalex_expanded(
    *,
    output_dir: Path,
    field_id: str | None,
    subfield_ids: list[str],
    language: str | None,
    source_dataset: str,
    seeds_per_period: int,
    candidate_sample_size: int,
    min_references: int,
    seed: int,
    mailto: str | None,
    sleep_seconds: float,
) -> None:
    """Build the expanded OpenAlex seed/reference dataset.

    The output directory receives raw `papers.csv`, `pairs.csv`, a seed sample,
    and summaries. Seed papers are balanced across TIME_BANDS and citation
    strata; every hydrated usable OpenAlex reference becomes a positive pair.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_seeds: list[dict[str, Any]] = []
    for offset, (ai_period, (from_date, to_date)) in enumerate(TIME_BANDS.items()):
        # Offset the OpenAlex sample seed by period so pre/post samples are
        # independently reproducible rather than identical random draws.
        candidates = fetch_seed_candidates(
            ai_period=ai_period,
            from_date=from_date,
            to_date=to_date,
            field_id=field_id,
            subfield_ids=subfield_ids,
            language=language,
            candidate_sample_size=candidate_sample_size,
            seed=seed + offset,
            min_references=min_references,
            mailto=mailto,
            sleep_seconds=sleep_seconds,
        )
        selected_seeds.extend(
            select_stratified_seeds(candidates, seeds_per_period=seeds_per_period, rng=rng)
        )

    query_to_refs: dict[str, list[str]] = {}
    ref_ids: list[str] = []
    for query in selected_seeds:
        # Collect all listed references first, then hydrate the unique set in
        # bulk. Some references will be dropped later if OpenAlex lacks text.
        query_id = openalex_short_id(query["id"])
        refs = [openalex_short_id(ref) for ref in query.get("referenced_works", [])]
        query_to_refs[query_id] = refs
        ref_ids.extend(refs)

    ref_works = fetch_works_by_ids(
        sorted(set(ref_ids)),
        language=language,
        mailto=mailto,
        sleep_seconds=sleep_seconds,
    )

    papers: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for query in selected_seeds:
        query_id = openalex_short_id(query["id"])
        ai_period = query["_ai_period"]
        stratum = query["_seed_citation_stratum"]
        refs = query_to_refs[query_id]
        usable_ref_ids = [ref_id for ref_id in refs if ref_id in ref_works]

        # Store seed metadata both in the all-paper table and in a seed-only
        # sample table so later analyses can audit sampling balance.
        papers[query_id] = paper_row(
            query,
            source_dataset=source_dataset,
            ai_period=ai_period,
            is_seed=True,
            seed_citation_stratum=stratum,
            seed_reference_count=len(refs),
            seed_usable_reference_count=len(usable_ref_ids),
        )
        seed_rows.append(
            {
                "paper_id": query_id,
                "ai_period": ai_period,
                "seed_citation_stratum": stratum,
                "year": query.get("publication_year"),
                "title": query.get("title") or query.get("display_name") or "",
                "field": work_field_name(query),
                "language": query.get("language"),
                "openalex_current_cited_by_count": query.get("cited_by_count"),
                "reference_count": len(refs),
                "usable_reference_count": len(usable_ref_ids),
            }
        )

        for ref_id in usable_ref_ids:
            # Do not overwrite a work that is itself a sampled seed; keeping
            # seed fields preserves query-level provenance for dual-role works.
            if not papers.get(ref_id, {}).get("is_seed"):
                papers[ref_id] = paper_row(ref_works[ref_id], source_dataset=source_dataset)
            pairs.append(
                {
                    "query_id": query_id,
                    "candidate_id": ref_id,
                    "label": 1,
                    "label_type": "actual_openalex_reference",
                    "split": "test",
                    "source_dataset": source_dataset,
                }
            )

    papers_df = pd.DataFrame(papers.values()).sort_values("paper_id")
    pairs_df = pd.DataFrame(
        pairs,
        columns=["query_id", "candidate_id", "label", "label_type", "split", "source_dataset"],
    )
    seed_df = pd.DataFrame(seed_rows).sort_values(["ai_period", "seed_citation_stratum", "paper_id"])

    papers_df.to_csv(output_dir / "papers.csv", index=False)
    pairs_df.to_csv(output_dir / "pairs.csv", index=False)
    seed_df.to_csv(output_dir / "seed_sample.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "papers", "value": len(papers_df)},
            {"metric": "seed_papers", "value": len(seed_df)},
            {"metric": "query_papers_with_usable_references", "value": pairs_df["query_id"].nunique()},
            {"metric": "pairs", "value": len(pairs_df)},
            {"metric": "positive_pairs", "value": int((pairs_df["label"] == 1).sum()) if not pairs_df.empty else 0},
            {"metric": "unique_reference_works", "value": pairs_df["candidate_id"].nunique()},
            {"metric": "field_id", "value": field_id or ""},
            {"metric": "subfield_ids", "value": "|".join(subfield_ids)},
            {"metric": "language", "value": language or "any"},
            {"metric": "seeds_per_period", "value": seeds_per_period},
            {"metric": "candidate_sample_size", "value": candidate_sample_size},
            {"metric": "min_references", "value": min_references},
            {"metric": "random_seed", "value": seed},
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    period_summary = (
        seed_df.groupby(["ai_period", "seed_citation_stratum"], dropna=False)
        .agg(
            seed_papers=("paper_id", "count"),
            mean_seed_cited_by_count=("openalex_current_cited_by_count", "mean"),
            median_seed_cited_by_count=("openalex_current_cited_by_count", "median"),
            mean_reference_count=("reference_count", "mean"),
            mean_usable_reference_count=("usable_reference_count", "mean"),
        )
        .reset_index()
    )
    period_summary.to_csv(output_dir / "seed_summary_by_period_stratum.csv", index=False)


def main() -> None:
    """Parse CLI arguments and build the OpenAlex field-scan raw dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/openalex_cs_200")
    parser.add_argument("--field-id", default=DEFAULT_FIELD_ID)
    parser.add_argument("--subfield-ids", default="")
    parser.add_argument("--language", default="en")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--seeds-per-period", type=int, default=200)
    parser.add_argument("--candidate-sample-size", type=int, default=3000)
    parser.add_argument("--min-references", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    args = parser.parse_args()
    subfield_ids = [part.strip() for part in args.subfield_ids.split("|") if part.strip()]

    build_openalex_expanded(
        output_dir=Path(args.output_dir),
        field_id=None if subfield_ids else args.field_id,
        subfield_ids=subfield_ids,
        language=args.language,
        source_dataset=args.source_dataset,
        seeds_per_period=args.seeds_per_period,
        candidate_sample_size=args.candidate_sample_size,
        min_references=args.min_references,
        seed=args.seed,
        mailto=args.mailto,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
