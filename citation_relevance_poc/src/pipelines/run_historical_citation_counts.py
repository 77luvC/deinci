from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils.io import ensure_dir, read_table, write_table


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OBSERVATION_END_DATE = pd.Timestamp("2026-06-30")


def cutoff_date_for_query(publication_date: object, year: object) -> str | None:
    date = pd.to_datetime(publication_date, errors="coerce")
    if not pd.isna(date):
        cutoff = min(pd.Timestamp(year=int(date.year), month=12, day=31), OBSERVATION_END_DATE)
        return cutoff.strftime("%Y-%m-%d")

    numeric_year = pd.to_numeric(pd.Series([year]), errors="coerce").iloc[0]
    if pd.isna(numeric_year):
        return None
    cutoff = min(pd.Timestamp(year=int(numeric_year), month=12, day=31), OBSERVATION_END_DATE)
    return cutoff.strftime("%Y-%m-%d")


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "citation_count_cutoff_date",
                "openalex_citing_works_through_cutoff",
            ]
        )
    return read_table(path)


def load_yearly_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "openalex_current_cited_by_count",
                "counts_by_year_json",
            ]
        )
    return read_table(path)


def query_openalex_citing_count(
    candidate_id: str,
    cutoff_date: str,
    *,
    mailto: str | None,
    api_key: str | None,
    sleep_seconds: float,
    max_retries: int,
) -> int:
    params: dict[str, Any] = {
        "filter": f"cites:{candidate_id},to_publication_date:{cutoff_date}",
        "per-page": 1,
        "select": "id",
    }
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after else max(30.0, sleep_seconds * 10)
                except ValueError:
                    wait_seconds = max(30.0, sleep_seconds * 10)
                print(
                    f"Rate limited by OpenAlex for {candidate_id} through {cutoff_date}; "
                    f"waiting {wait_seconds:.1f}s before retry {attempt + 1}/{max_retries}",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            time.sleep(sleep_seconds)
            return int(response.json()["meta"]["count"])
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(sleep_seconds * (2 ** attempt + 1))
    raise RuntimeError(f"OpenAlex count query failed for {candidate_id} through {cutoff_date}") from last_error


def fetch_work_yearly_counts(
    candidate_ids: list[str],
    *,
    mailto: str | None,
    api_key: str | None,
    sleep_seconds: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "filter": f"openalex_id:{'|'.join(candidate_ids)}",
        "per-page": len(candidate_ids),
        "select": "id,cited_by_count,counts_by_year",
    }
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = float(retry_after) if retry_after else max(30.0, sleep_seconds * 10)
                except ValueError:
                    wait_seconds = max(30.0, sleep_seconds * 10)
                print(
                    f"Rate limited by OpenAlex while fetching yearly counts; "
                    f"waiting {wait_seconds:.1f}s before retry {attempt + 1}/{max_retries}",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            time.sleep(sleep_seconds)
            return response.json().get("results", [])
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(sleep_seconds * (2 ** attempt + 1))
    raise RuntimeError(f"OpenAlex yearly-count query failed for {len(candidate_ids)} works") from last_error


def count_through_year_from_yearly(current_count: object, counts_by_year_json: object, cutoff_date: object) -> float:
    current = pd.to_numeric(pd.Series([current_count]), errors="coerce").iloc[0]
    if pd.isna(current):
        return float("nan")
    cutoff = pd.to_datetime(cutoff_date, errors="coerce")
    if pd.isna(cutoff):
        return float("nan")

    try:
        counts_by_year = json.loads(counts_by_year_json) if isinstance(counts_by_year_json, str) else counts_by_year_json
    except json.JSONDecodeError:
        counts_by_year = []
    if not isinstance(counts_by_year, list):
        counts_by_year = []

    future_count = 0
    for item in counts_by_year:
        try:
            year = int(item.get("year"))
            count = int(item.get("cited_by_count") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if year > int(cutoff.year):
            future_count += count
    return float(max(current - future_count, 0))


def enrich_historical_counts(
    scores: pd.DataFrame,
    papers: pd.DataFrame,
    *,
    cache_path: Path,
    mailto: str | None,
    api_key: str | None,
    sleep_seconds: float,
    max_retries: int,
    checkpoint_every: int,
) -> pd.DataFrame:
    out = scores.copy()
    papers_by_id = papers.set_index("paper_id", drop=False)

    if "query_publication_date" not in out.columns:
        query_dates = papers_by_id["publication_date"] if "publication_date" in papers_by_id.columns else pd.Series(dtype=object)
        out["query_publication_date"] = out["query_id"].map(query_dates)
    if "query_year" not in out.columns:
        out["query_year"] = out["query_id"].map(papers_by_id["year"])

    out["citation_count_cutoff_date"] = out.apply(
        lambda row: cutoff_date_for_query(row.get("query_publication_date"), row.get("query_year")),
        axis=1,
    )

    cache = load_cache(cache_path)
    cache_key_cols = ["candidate_id", "citation_count_cutoff_date"]
    cache = cache.drop_duplicates(cache_key_cols, keep="last")
    cache_index = {
        (str(row.candidate_id), str(row.citation_count_cutoff_date)): int(row.openalex_citing_works_through_cutoff)
        for row in cache.itertuples(index=False)
        if pd.notna(row.citation_count_cutoff_date)
    }

    actual = out[out["label"].astype(int) == 1].copy()
    requests_needed = (
        actual[["candidate_id", "citation_count_cutoff_date"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["candidate_id", "citation_count_cutoff_date"])
    )

    new_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(requests_needed.itertuples(index=False), start=1):
        key = (str(row.candidate_id), str(row.citation_count_cutoff_date))
        if key in cache_index:
            continue
        count = query_openalex_citing_count(
            key[0],
            key[1],
            mailto=mailto,
            api_key=api_key,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        )
        cache_index[key] = count
        new_rows.append(
            {
                "candidate_id": key[0],
                "citation_count_cutoff_date": key[1],
                "openalex_citing_works_through_cutoff": count,
            }
        )
        if checkpoint_every and len(new_rows) % checkpoint_every == 0:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            cache = cache.drop_duplicates(cache_key_cols, keep="last")
            write_table(cache, cache_path)
            new_rows = []
            print(f"Cached {idx}/{len(requests_needed)} candidate-cutoff counts")

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates(cache_key_cols, keep="last")
        write_table(cache, cache_path)

    count_lookup = {
        (str(row.candidate_id), str(row.citation_count_cutoff_date)): int(row.openalex_citing_works_through_cutoff)
        for row in cache.itertuples(index=False)
        if pd.notna(row.citation_count_cutoff_date)
    }

    raw_counts = [
        count_lookup.get((str(row.candidate_id), str(row.citation_count_cutoff_date)), np.nan)
        for row in out.itertuples(index=False)
    ]
    out["candidate_cited_by_count_at_query_year_raw"] = raw_counts
    out["citation_count_subtracts_focal_pair"] = out["label"].astype(int) == 1
    out["candidate_cited_by_count_at_query_year"] = np.where(
        out["citation_count_subtracts_focal_pair"],
        np.maximum(pd.to_numeric(out["candidate_cited_by_count_at_query_year_raw"], errors="coerce") - 1, 0),
        out["candidate_cited_by_count_at_query_year_raw"],
    )
    out["candidate_log1p_cited_by_count_at_query_year"] = np.log1p(
        pd.to_numeric(out["candidate_cited_by_count_at_query_year"], errors="coerce")
    )
    out["citation_count_cutoff_rule"] = "min(query_publication_year_end,2026-06-30); actual references subtract focal pair"
    return out


def enrich_historical_counts_from_yearly(
    scores: pd.DataFrame,
    papers: pd.DataFrame,
    *,
    cache_path: Path,
    mailto: str | None,
    api_key: str | None,
    sleep_seconds: float,
    max_retries: int,
) -> pd.DataFrame:
    out = scores.copy()
    papers_by_id = papers.set_index("paper_id", drop=False)

    if "query_publication_date" not in out.columns:
        query_dates = papers_by_id["publication_date"] if "publication_date" in papers_by_id.columns else pd.Series(dtype=object)
        out["query_publication_date"] = out["query_id"].map(query_dates)
    if "query_year" not in out.columns:
        out["query_year"] = out["query_id"].map(papers_by_id["year"])

    out["citation_count_cutoff_date"] = out.apply(
        lambda row: cutoff_date_for_query(row.get("query_publication_date"), row.get("query_year")),
        axis=1,
    )

    cache = load_yearly_cache(cache_path).drop_duplicates(["candidate_id"], keep="last")
    cached_ids = set(cache["candidate_id"].astype(str))
    actual_ids = sorted(out.loc[out["label"].astype(int) == 1, "candidate_id"].astype(str).unique())
    missing_ids = [candidate_id for candidate_id in actual_ids if candidate_id not in cached_ids]

    new_rows: list[dict[str, Any]] = []
    for start in range(0, len(missing_ids), 50):
        chunk = missing_ids[start : start + 50]
        if not chunk:
            continue
        for work in fetch_work_yearly_counts(
            chunk,
            mailto=mailto,
            api_key=api_key,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        ):
            candidate_id = str(work["id"]).rstrip("/").split("/")[-1]
            new_rows.append(
                {
                    "candidate_id": candidate_id,
                    "openalex_current_cited_by_count": work.get("cited_by_count"),
                    "counts_by_year_json": json.dumps(work.get("counts_by_year") or []),
                }
            )
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates(["candidate_id"], keep="last")
        write_table(cache, cache_path)
        new_rows = []
        print(f"Cached yearly counts for {min(start + 50, len(missing_ids))}/{len(missing_ids)} missing works")

    yearly = cache.set_index("candidate_id", drop=False)
    current_lookup = yearly["openalex_current_cited_by_count"].to_dict()
    counts_lookup = yearly["counts_by_year_json"].to_dict()
    raw_counts = []
    for row in out.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        raw_counts.append(
            count_through_year_from_yearly(
                current_lookup.get(candidate_id),
                counts_lookup.get(candidate_id),
                row.citation_count_cutoff_date,
            )
        )

    out["candidate_cited_by_count_at_query_year_raw"] = raw_counts
    out["citation_count_subtracts_focal_pair"] = out["label"].astype(int) == 1
    out["candidate_cited_by_count_at_query_year"] = np.where(
        out["citation_count_subtracts_focal_pair"],
        np.maximum(pd.to_numeric(out["candidate_cited_by_count_at_query_year_raw"], errors="coerce") - 1, 0),
        out["candidate_cited_by_count_at_query_year_raw"],
    )
    out["candidate_log1p_cited_by_count_at_query_year"] = np.log1p(
        pd.to_numeric(out["candidate_cited_by_count_at_query_year"], errors="coerce")
    )
    out["citation_count_cutoff_rule"] = (
        "OpenAlex cited_by_count minus counts_by_year after query publication year; "
        "actual references subtract focal pair"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--method", choices=["exact", "yearly"], default="exact")
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    ensure_dir(Path(args.cache).parent)
    api_key = args.api_key or os.environ.get("OPENALEX_API_KEY")
    if args.method == "yearly":
        enriched = enrich_historical_counts_from_yearly(
            read_table(args.scores),
            read_table(args.papers),
            cache_path=Path(args.cache),
            mailto=args.mailto,
            api_key=api_key,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
    else:
        enriched = enrich_historical_counts(
            read_table(args.scores),
            read_table(args.papers),
            cache_path=Path(args.cache),
            mailto=args.mailto,
            api_key=api_key,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            checkpoint_every=args.checkpoint_every,
        )
    write_table(enriched, args.output)
    print(f"Saved historical citation counts: {args.output}")


if __name__ == "__main__":
    main()
