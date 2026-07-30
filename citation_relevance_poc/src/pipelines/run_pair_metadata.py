from __future__ import annotations

import argparse

import pandas as pd

from src.utils.io import read_table, write_table


def main() -> None:
    """Join pair rows with paper metadata without computing embeddings.

    This produces the same analysis-facing columns as pair scoring, but leaves
    `scincl_cosine` empty. It is useful for metadata-only analyses and assumes
    every pair ID exists in the paper table.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    papers = read_table(args.papers).set_index("paper_id", drop=False)
    pairs = read_table(args.pairs)

    rows = []
    missing = []
    for pair in pairs.to_dict("records"):
        # Keep the row shape parallel to `run_pair_scoring` so downstream
        # analyses can consume either scored or metadata-only pair files.
        query_paper_id = str(pair["query_paper_id"])
        candidate_paper_id = str(pair["candidate_paper_id"])
        if query_paper_id not in papers.index or candidate_paper_id not in papers.index:
            missing.append((query_paper_id, candidate_paper_id))
            continue
        query = papers.loc[query_paper_id]
        candidate = papers.loc[candidate_paper_id]
        rows.append(
            {
                **pair,
                "scincl_cosine": pd.NA,
                "query_year": query.get("year"),
                "query_publication_date": query.get("publication_date"),
                "candidate_year": candidate.get("year"),
                "candidate_publication_date": candidate.get("publication_date"),
                "query_field": query.get("field"),
                "candidate_field": candidate.get("field"),
                "query_is_genai": bool(query.get("is_genai")),
                "candidate_is_genai": bool(candidate.get("is_genai")),
            }
        )

    if missing:
        raise ValueError(f"Pairs reference missing paper IDs; first examples: {missing[:5]}")

    write_table(pd.DataFrame(rows), args.output)
    print(f"Saved pair metadata: {args.output}")


if __name__ == "__main__":
    main()
