"""ColPali visual late-interaction retrieval over guideline page images.

ColPali embeds each page *image* as a set of patch-token vectors and scores a text query
against it with MaxSim late interaction (ColBERT-style) — so layout, tables, and figures
are searchable, not just the text layer. The MaxSim math here is plain numpy (and unit
tested); the ColPali encoder itself lives behind the optional `colpali` (torch) extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    """Late-interaction score: sum over query tokens of the max similarity to any doc
    token. query:[m, dim], doc:[n, dim] (assumed L2-normalised). Returns a scalar."""
    if query.size == 0 or doc.size == 0:
        return 0.0
    sims = query @ doc.T  # [m, n]
    return float(sims.max(axis=1).sum())


class MaxSimIndex:
    """In-memory multi-vector index with MaxSim scoring, persisted as .npz + json.

    Pages are few (guideline PDFs), so brute-force MaxSim is fine and exact — no ANN
    structure needed at this scale."""

    def __init__(self) -> None:
        self._meta: list[dict] = []
        self._embs: list[np.ndarray] = []

    def add(self, meta: dict, emb: np.ndarray) -> None:
        self._meta.append(meta)
        self._embs.append(emb.astype("float32"))

    def __len__(self) -> int:
        return len(self._meta)

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[dict, float]]:
        scored = [(m, maxsim(query_emb, e)) for m, e in zip(self._meta, self._embs, strict=True)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ragged arrays: store each page's matrix under an indexed key + the meta json.
        arrays = {str(i): e for i, e in enumerate(self._embs)}
        np.savez_compressed(path, **arrays)
        path.with_suffix(".meta.json").write_text(
            json.dumps(self._meta), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> MaxSimIndex:
        path = Path(path)
        idx = cls()
        idx._meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        with np.load(path) as data:
            idx._embs = [data[str(i)] for i in range(len(idx._meta))]
        return idx


class ColPaliPageEmbedder:
    """ColPali encoder (vidore/colpali-*) producing multi-vector page/query embeddings."""

    name = "colpali"

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = ColPali.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device).eval()
        self._processor = ColPaliProcessor.from_pretrained(model_name)

    def _to_numpy(self, batch_out) -> list[np.ndarray]:
        return [e.cpu().float().numpy() for e in batch_out]

    def embed_images(self, images_png: list[bytes]) -> list[np.ndarray]:
        from io import BytesIO

        from PIL import Image

        pil = [Image.open(BytesIO(b)).convert("RGB") for b in images_png]
        batch = self._processor.process_images(pil).to(self.device)
        with self._torch.no_grad():
            out = self._model(**batch)
        return self._to_numpy(out)

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        batch = self._processor.process_queries(texts).to(self.device)
        with self._torch.no_grad():
            out = self._model(**batch)
        return self._to_numpy(out)
