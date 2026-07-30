from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.cache import CACHE_SCHEMA_VERSION, hash_dataframe, slugify, write_manifest
from src.utils.io import ensure_dir
from src.utils.text import join_title_abstract


@dataclass
class SciNCLEncoder:
    model_name: str = "malteos/scincl"
    device: str | None = None
    max_length: int = 512
    normalize: bool = True

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_papers(self, papers: list[dict], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        texts = [join_title_abstract(p.get("title"), p.get("abstract")) for p in papers]
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding papers"):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(batch_embeddings)

        out = np.vstack(embeddings).astype("float32")
        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-12)
        return out


def embedding_cache_paths(output_prefix: str | Path, model_name: str) -> dict[str, Path]:
    """Return the stable cache file paths for one (output_prefix, model) pair.

    Embeddings live under the caller-chosen prefix directory (one per
    dataset/field), but the filenames are namespaced by model so switching
    embedding models never silently reuses another model's vectors.
    """
    prefix = Path(output_prefix)
    stem = f"{prefix.name}__{slugify(model_name)}"
    return {
        "npy": prefix.parent / f"{stem}.npy",
        "metadata": prefix.parent / f"{stem}_metadata.parquet",
        "manifest": prefix.parent / f"{stem}.manifest.json",
    }


def embedding_input_hash(papers: pd.DataFrame) -> str:
    """Hash the paper text/id fields that determine embedding content."""
    return hash_dataframe(papers, ["paper_id", "title", "abstract"])


def save_embeddings(
    papers: pd.DataFrame,
    embeddings: np.ndarray,
    output_prefix: str | Path,
    *,
    model_name: str,
    input_hash: str,
) -> tuple[Path, Path]:
    paths = embedding_cache_paths(output_prefix, model_name)
    ensure_dir(paths["npy"].parent)
    np.save(paths["npy"], embeddings)
    papers.to_parquet(paths["metadata"], index=False)
    write_manifest(
        paths["manifest"],
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "model_name": model_name,
            "input_hash": input_hash,
            "n_papers": int(len(papers)),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return paths["npy"], paths["metadata"]

