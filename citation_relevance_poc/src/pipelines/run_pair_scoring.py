from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.scincl_encoder import embedding_cache_paths
from src.utils.cache import CACHE_SCHEMA_VERSION, hash_dataframe, manifest_matches, read_manifest, write_manifest
from src.utils.io import read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="malteos/scincl")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute pair scores even if a matching cache already exists.",
    )
    args = parser.parse_args()

    embed_paths = embedding_cache_paths(args.embeddings, args.model)
    embed_manifest = read_manifest(embed_paths["manifest"])
    if embed_manifest is None:
        raise FileNotFoundError(
            f"No embedding cache manifest found at {embed_paths['manifest']}; run run_embed for model "
            f"'{args.model}' first."
        )

    pairs = read_table(args.pairs)
    pairs_hash = hash_dataframe(pairs, ["query_paper_id", "candidate_paper_id", "label"])

    output_path = Path(args.output)
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model_name": args.model,
        "embeddings_input_hash": embed_manifest.get("input_hash"),
        "pairs_hash": pairs_hash,
    }
    existing_manifest = read_manifest(manifest_path)
    if not args.force_recompute and manifest_matches(existing_manifest, expected, [output_path]):
        print(f"Reusing cached pair scores (unchanged pairs/embeddings/model): {output_path}")
        return

    papers = read_table(args.papers).set_index("paper_id", drop=False)
    embeddings = np.load(embed_paths["npy"])
    metadata = pd.read_parquet(embed_paths["metadata"]).reset_index(drop=True)
    id_to_idx = {str(pid): idx for idx, pid in enumerate(metadata["paper_id"].astype(str))}

    rows = []
    missing = []
    for pair in pairs.to_dict("records"):
        query_paper_id = str(pair["query_paper_id"])
        candidate_paper_id = str(pair["candidate_paper_id"])
        if query_paper_id not in id_to_idx or candidate_paper_id not in id_to_idx:
            missing.append((query_paper_id, candidate_paper_id))
            continue
        query_vec = embeddings[id_to_idx[query_paper_id]]
        candidate_vec = embeddings[id_to_idx[candidate_paper_id]]
        score = float(np.dot(query_vec, candidate_vec))
        query = papers.loc[query_paper_id]
        candidate = papers.loc[candidate_paper_id]
        rows.append(
            {
                **pair,
                "scincl_cosine": score,
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
        raise ValueError(f"Missing embeddings for {len(missing)} pairs; first examples: {missing[:5]}")

    scores = pd.DataFrame(rows)
    if scores["scincl_cosine"].isna().any():
        raise ValueError("scincl_cosine contains missing values after scoring; refusing to write a partial cache.")

    write_table(scores, args.output)
    write_manifest(manifest_path, expected)
    print(f"Saved pair scores: {args.output}")


if __name__ == "__main__":
    main()
