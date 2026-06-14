from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_table, write_table


def _embedding_paths(prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(prefix)
    if prefix.suffix == ".npy":
        npy = prefix
        meta = prefix.parent / f"{prefix.stem}_metadata.parquet"
    else:
        npy = prefix.with_suffix(".npy")
        meta = prefix.parent / f"{prefix.name}_metadata.parquet"
    return npy, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    papers = read_table(args.papers).set_index("paper_id", drop=False)
    pairs = read_table(args.pairs)
    npy_path, meta_path = _embedding_paths(args.embeddings)
    embeddings = np.load(npy_path)
    metadata = pd.read_parquet(meta_path).reset_index(drop=True)
    id_to_idx = {str(pid): idx for idx, pid in enumerate(metadata["paper_id"].astype(str))}

    rows = []
    missing = []
    for pair in pairs.to_dict("records"):
        query_id = str(pair["query_id"])
        candidate_id = str(pair["candidate_id"])
        if query_id not in id_to_idx or candidate_id not in id_to_idx:
            missing.append((query_id, candidate_id))
            continue
        query_vec = embeddings[id_to_idx[query_id]]
        candidate_vec = embeddings[id_to_idx[candidate_id]]
        score = float(np.dot(query_vec, candidate_vec))
        query = papers.loc[query_id]
        candidate = papers.loc[candidate_id]
        rows.append(
            {
                **pair,
                "scincl_cosine": score,
                "query_year": query.get("year"),
                "candidate_year": candidate.get("year"),
                "query_field": query.get("field"),
                "candidate_field": candidate.get("field"),
                "query_is_genai": bool(query.get("is_genai")),
                "candidate_is_genai": bool(candidate.get("is_genai")),
            }
        )

    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} pairs; first examples: {missing[:5]}")

    write_table(pd.DataFrame(rows), args.output)
    print(f"Saved pair scores: {args.output}")


if __name__ == "__main__":
    main()

