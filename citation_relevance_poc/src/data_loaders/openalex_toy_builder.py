from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in index.items():
        positioned.extend((position, token) for position in positions)
    return " ".join(token for _, token in sorted(positioned))


def openalex_short_id(openalex_url: str) -> str:
    return openalex_url.rstrip("/").split("/")[-1]


def source_name(work: dict[str, Any]) -> str:
    source = (work.get("primary_location") or {}).get("source") or {}
    return source.get("display_name") or "unknown"


def field_name(work: dict[str, Any]) -> str:
    topics = work.get("topics") or []
    if not topics:
        return "unknown"
    field = topics[0].get("field") or {}
    return field.get("display_name") or "unknown"


def paper_row(work: dict[str, Any], source_dataset: str = "openalex_toy") -> dict[str, Any]:
    return {
        "paper_id": openalex_short_id(work["id"]),
        "title": work.get("title") or work.get("display_name") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "work_type": work.get("type"),
        "venue": source_name(work),
        "field": field_name(work),
        "is_oa": (work.get("open_access") or {}).get("is_oa"),
        "openalex_current_cited_by_count": work.get("cited_by_count"),
        "source_dataset": source_dataset,
    }


def get_json(params: dict[str, Any], sleep_seconds: float = 0.15) -> dict[str, Any]:
    response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    response.raise_for_status()
    time.sleep(sleep_seconds)
    return response.json()


def fetch_query_works(
    search: str,
    from_date: str,
    to_date: str,
    n_queries: int,
    pages_to_scan: int,
) -> list[dict[str, Any]]:
    works: list[dict[str, Any]] = []
    seen: set[str] = set()
    select = ",".join(
        [
            "id",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "type",
            "primary_location",
            "open_access",
            "cited_by_count",
            "referenced_works",
            "abstract_inverted_index",
            "topics",
        ]
    )
    for page in range(1, pages_to_scan + 1):
        params = {
            "search": search,
            "filter": (
                f"from_publication_date:{from_date},"
                f"to_publication_date:{to_date},"
                "has_abstract:true,has_references:true,type:article"
            ),
            "per-page": 25,
            "page": page,
            "select": select,
            "sort": "cited_by_count:desc",
        }
        for work in get_json(params).get("results", []):
            work_id = openalex_short_id(work["id"])
            if work_id in seen:
                continue
            if len(work.get("referenced_works") or []) < 8:
                continue
            if not reconstruct_abstract(work.get("abstract_inverted_index")):
                continue
            seen.add(work_id)
            works.append(work)
            if len(works) >= n_queries:
                return works
    return works


def fetch_works_by_ids(openalex_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    select = ",".join(
        [
            "id",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "type",
            "primary_location",
            "open_access",
            "cited_by_count",
            "abstract_inverted_index",
            "topics",
        ]
    )
    ids = [openalex_short_id(work_id) for work_id in openalex_ids]
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        params = {
            "filter": f"openalex_id:{'|'.join(chunk)}",
            "per-page": len(chunk),
            "select": select,
        }
        for work in get_json(params).get("results", []):
            if work.get("title") and reconstruct_abstract(work.get("abstract_inverted_index")):
                out[openalex_short_id(work["id"])] = work
    return out


def build_openalex_toy(
    output_dir: Path,
    search: str,
    queries_per_period: int,
    positives_per_query: int,
    negatives_per_positive: int,
    seed: int,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    periods = {
        "pre_ai": ("2019-01-01", "2021-12-31"),
        "post_ai": ("2023-01-01", "2026-06-30"),
    }
    query_works: list[dict[str, Any]] = []
    for period, (from_date, to_date) in periods.items():
        works = fetch_query_works(search, from_date, to_date, queries_per_period, pages_to_scan=8)
        for work in works:
            work["_period"] = period
        query_works.extend(works)

    ref_ids: list[str] = []
    query_to_refs: dict[str, list[str]] = {}
    for query in query_works:
        query_id = openalex_short_id(query["id"])
        refs = [openalex_short_id(ref) for ref in query.get("referenced_works", [])[:60]]
        query_to_refs[query_id] = refs
        ref_ids.extend(refs)

    ref_works = fetch_works_by_ids(sorted(set(ref_ids)))

    papers: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    candidate_pool: list[str] = sorted(ref_works)

    for query in query_works:
        query_id = openalex_short_id(query["id"])
        papers[query_id] = paper_row(query)

        positive_ids = [ref_id for ref_id in query_to_refs[query_id] if ref_id in ref_works]
        positive_ids = positive_ids[:positives_per_query]
        if len(positive_ids) < positives_per_query:
            continue

        for ref_id in positive_ids:
            papers[ref_id] = paper_row(ref_works[ref_id])
            pairs.append(
                {
                    "query_id": query_id,
                    "candidate_id": ref_id,
                    "label": 1,
                    "label_type": "actual_openalex_reference",
                    "split": "test",
                    "source_dataset": "openalex_toy",
                }
            )

        excluded = set(query_to_refs[query_id]) | {query_id}
        negative_pool = [candidate_id for candidate_id in candidate_pool if candidate_id not in excluded]
        n_negatives = positives_per_query * negatives_per_positive
        for candidate_id in random.sample(negative_pool, min(n_negatives, len(negative_pool))):
            papers[candidate_id] = paper_row(ref_works[candidate_id])
            pairs.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "label": 0,
                    "label_type": "sampled_non_reference",
                    "split": "test",
                    "source_dataset": "openalex_toy",
                }
            )

    papers_df = pd.DataFrame(papers.values()).sort_values("paper_id")
    pairs_df = pd.DataFrame(pairs)
    papers_df.to_csv(output_dir / "papers.csv", index=False)
    pairs_df.to_csv(output_dir / "pairs.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "papers", "value": len(papers_df)},
            {"metric": "query_papers", "value": len(set(pairs_df["query_id"])) if not pairs_df.empty else 0},
            {"metric": "pairs", "value": len(pairs_df)},
            {"metric": "positive_pairs", "value": int((pairs_df["label"] == 1).sum()) if not pairs_df.empty else 0},
            {"metric": "negative_pairs", "value": int((pairs_df["label"] == 0).sum()) if not pairs_df.empty else 0},
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/openalex_toy")
    parser.add_argument("--search", default="machine learning")
    parser.add_argument("--queries-per-period", type=int, default=10)
    parser.add_argument("--positives-per-query", type=int, default=3)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    build_openalex_toy(
        output_dir=Path(args.output_dir),
        search=args.search,
        queries_per_period=args.queries_per_period,
        positives_per_query=args.positives_per_query,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
