from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir, write_table
from src.utils.text import detect_genai

PAPER_COLUMNS = [
    "paper_id",
    "title",
    "abstract",
    "year",
    "publication_date",
    "work_type",
    "venue",
    "field",
    "is_oa",
    "openalex_current_cited_by_count",
    "is_genai",
    "source_dataset",
]

PAIR_COLUMNS = [
    "query_id",
    "candidate_id",
    "label",
    "label_type",
    "split",
    "source_dataset",
]


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_generic_pairs(papers_path: str, pairs_path: str, output_dir: str) -> None:
    papers = pd.read_csv(papers_path)
    pairs = pd.read_csv(pairs_path)

    _require_columns(papers, ["paper_id", "title", "abstract"], "papers")
    _require_columns(pairs, ["query_id", "candidate_id", "label"], "pairs")

    for col in PAPER_COLUMNS:
        if col not in papers.columns:
            papers[col] = None
    for col in PAIR_COLUMNS:
        if col not in pairs.columns:
            pairs[col] = None

    papers["paper_id"] = papers["paper_id"].astype(str)
    papers["year"] = pd.to_numeric(papers["year"], errors="coerce").astype("Int64")
    papers["source_dataset"] = papers["source_dataset"].fillna("generic")
    if papers["is_genai"].isna().any():
        papers["is_genai"] = papers.apply(
            lambda row: detect_genai(row.get("title"), row.get("abstract")),
            axis=1,
        )
    else:
        papers["is_genai"] = papers["is_genai"].astype(str).str.lower().isin(["true", "1", "yes"])

    pairs["query_id"] = pairs["query_id"].astype(str)
    pairs["candidate_id"] = pairs["candidate_id"].astype(str)
    pairs["label"] = pd.to_numeric(pairs["label"], errors="raise").astype(int)
    pairs["label_type"] = pairs["label_type"].fillna("unknown")
    pairs["split"] = pairs["split"].fillna("test")
    pairs["source_dataset"] = pairs["source_dataset"].fillna("generic")

    paper_ids = set(papers["paper_id"])
    missing_ids = sorted((set(pairs["query_id"]) | set(pairs["candidate_id"])) - paper_ids)
    if missing_ids:
        raise ValueError(f"Pairs reference paper IDs missing from papers file: {missing_ids[:20]}")

    out = ensure_dir(output_dir)
    write_table(papers[PAPER_COLUMNS], out / "papers.parquet")
    write_table(pairs[PAIR_COLUMNS], out / "pairs.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-csv", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    load_generic_pairs(args.papers_csv, args.pairs_csv, args.output_dir)


if __name__ == "__main__":
    main()
