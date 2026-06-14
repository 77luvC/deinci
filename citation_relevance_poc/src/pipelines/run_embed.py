from __future__ import annotations

import argparse

from src.models.scincl_encoder import SciNCLEncoder, save_embeddings
from src.utils.io import read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="malteos/scincl")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    papers = read_table(args.papers)
    encoder = SciNCLEncoder(model_name=args.model)
    embeddings = encoder.encode_papers(papers.to_dict("records"), batch_size=args.batch_size)
    npy_path, meta_path = save_embeddings(papers, embeddings, args.output)
    print(f"Saved embeddings: {npy_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

