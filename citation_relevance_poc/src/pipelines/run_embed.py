from __future__ import annotations

import argparse

from src.models.scincl_encoder import (
    SciNCLEncoder,
    embedding_cache_paths,
    embedding_input_hash,
    save_embeddings,
)
from src.utils.cache import manifest_matches, read_manifest
from src.utils.io import read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="malteos/scincl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if a matching cache already exists.",
    )
    args = parser.parse_args()

    papers = read_table(args.papers)
    input_hash = embedding_input_hash(papers)
    paths = embedding_cache_paths(args.output, args.model)

    manifest = read_manifest(paths["manifest"])
    expected = {"model_name": args.model, "input_hash": input_hash}
    if not args.force_recompute and manifest_matches(manifest, expected, [paths["npy"], paths["metadata"]]):
        print(f"Reusing cached embeddings (unchanged inputs/model): {paths['npy']}")
        print(f"Reusing cached metadata: {paths['metadata']}")
        return

    encoder = SciNCLEncoder(model_name=args.model)
    embeddings = encoder.encode_papers(papers.to_dict("records"), batch_size=args.batch_size)
    npy_path, meta_path = save_embeddings(
        papers,
        embeddings,
        args.output,
        model_name=args.model,
        input_hash=input_hash,
    )
    print(f"Saved embeddings: {npy_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
