from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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


def save_embeddings(
    papers: pd.DataFrame,
    embeddings: np.ndarray,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    ensure_dir(prefix.parent)
    npy_path = prefix.with_suffix(".npy")
    meta_path = prefix.parent / f"{prefix.name}_metadata.parquet"
    np.save(npy_path, embeddings)
    papers.to_parquet(meta_path, index=False)
    return npy_path, meta_path

