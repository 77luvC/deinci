from __future__ import annotations

import argparse

import pandas as pd

from src.utils.io import read_table, write_table


def main() -> None:
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
        query_id = str(pair["query_id"])
        candidate_id = str(pair["candidate_id"])
        if query_id not in papers.index or candidate_id not in papers.index:
            missing.append((query_id, candidate_id))
            continue
        query = papers.loc[query_id]
        candidate = papers.loc[candidate_id]
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
