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

# The generic loader normalizes arbitrary CSV inputs into this canonical schema.
# Missing optional paper columns are filled with nulls so later pipeline stages
# can rely on stable column names.

PAIR_COLUMNS = [
    "query_paper_id",
    "candidate_paper_id",
    "label",
    "label_type",
    "split",
    "source_dataset",
]

# Pair rows use `label=1` for actual references/relevant pairs and `label=0` for
# comparison candidates. The loader validates IDs against the paper table before
# writing processed outputs.


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise a readable error when a required input column is missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_generic_pairs(papers_path: str, pairs_path: str, output_dir: str) -> None:
    """Convert raw paper/pair CSV files into canonical Parquet inputs.

    Required paper columns are `paper_id`, `title`, and `abstract`; required raw
    pair columns are `query_id`, `candidate_id`, and `label` -- the raw dump
    keeps these names, and this loader is the one seam where they become the
    processed schema's `query_paper_id`/`candidate_paper_id`. Optional columns
    are filled with defaults or nulls. Pair labels must be numeric, and every
    pair ID must resolve to a paper row.
    """
    papers = pd.read_csv(papers_path)
    pairs = pd.read_csv(pairs_path)

    _require_columns(papers, ["paper_id", "title", "abstract"], "papers")
    _require_columns(pairs, ["query_id", "candidate_id", "label"], "pairs")
    pairs = pairs.rename(columns={"query_id": "query_paper_id", "candidate_id": "candidate_paper_id"})

    # Backfill the full canonical schema before type normalization so each
    # downstream writer can select columns in a deterministic order.
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
        # When any GenAI flags are missing, recompute the heuristic for all rows
        # so mixed manual/automatic sources do not produce partially stale flags.
        papers["is_genai"] = papers.apply(
            lambda row: detect_genai(row.get("title"), row.get("abstract")),
            axis=1,
        )
    else:
        papers["is_genai"] = papers["is_genai"].astype(str).str.lower().isin(["true", "1", "yes"])

    pairs["query_paper_id"] = pairs["query_paper_id"].astype(str)
    pairs["candidate_paper_id"] = pairs["candidate_paper_id"].astype(str)
    pairs["label"] = pd.to_numeric(pairs["label"], errors="raise").astype(int)
    pairs["label_type"] = pairs["label_type"].fillna("unknown")
    pairs["split"] = pairs["split"].fillna("test")
    pairs["source_dataset"] = pairs["source_dataset"].fillna("generic")

    paper_ids = set(papers["paper_id"])
    missing_ids = sorted((set(pairs["query_paper_id"]) | set(pairs["candidate_paper_id"])) - paper_ids)
    if missing_ids:
        raise ValueError(f"Pairs reference paper IDs missing from papers file: {missing_ids[:20]}")

    # Write processed files in the exact columns expected by embedding, scoring,
    # and analysis stages.
    out = ensure_dir(output_dir)
    write_table(papers[PAPER_COLUMNS], out / "papers.parquet")
    write_table(pairs[PAIR_COLUMNS], out / "pairs.parquet")


def main() -> None:
    """Parse CLI arguments and run the generic CSV-to-Parquet loader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-csv", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    load_generic_pairs(args.papers_csv, args.pairs_csv, args.output_dir)


if __name__ == "__main__":
    main()
