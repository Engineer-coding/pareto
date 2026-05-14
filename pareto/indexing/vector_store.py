"""
FAISS-backed vector store with HNSW graph index.

Why HNSW?
    * Approximate nearest neighbor with sub-linear query time.
    * Naive search on 1M vectors takes ~800ms. HNSW does it in ~12ms with
      >95% recall when tuned.
    * Foundation of every production vector DB (Pinecone, Qdrant, Weaviate
      all wrap HNSW).

Three tunable parameters control the recall/speed trade-off:

    M               — number of bi-directional links per node.
                       Higher → better recall, more memory.   (default 32)
    efConstruction  — quality of the build. Higher → slower build,
                       better index quality.                  (default 200)
    efSearch        — quality of the query. Higher → slower query,
                       better recall.                         (default 50)

We expose all three; Week 5 will tune them with a benchmark.

Why inner product (METRIC_INNER_PRODUCT)?
    Our embedder L2-normalizes vectors, so dot product == cosine similarity.
    Inner product is the most efficient metric in FAISS HNSW.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pareto.indexing.models import SearchResult, VectorRecord

if TYPE_CHECKING:
    import numpy as np


@dataclass
class VectorStoreConfig:
    """Construction parameters for the index. Kept separate for benchmarks."""

    embedding_dim: int
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 50
    embedder_name: str = ""

    def to_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "embedder_name": self.embedder_name,
        }


class VectorStore:
    """In-memory FAISS HNSW vector store with parallel metadata records."""

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, config: VectorStoreConfig):
        import faiss  # heavy import deferred

        self._config = config
        self._index = faiss.IndexHNSWFlat(
            config.embedding_dim,
            config.M,
            faiss.METRIC_INNER_PRODUCT,
        )
        self._index.hnsw.efConstruction = config.ef_construction
        self._index.hnsw.efSearch = config.ef_search

        # Parallel arrays: FAISS internal id i ↔ self._records[i]
        self._records: list[VectorRecord] = []
        self._chunk_id_to_internal: dict[str, int] = {}

    # ── basic stats ───────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def config(self) -> VectorStoreConfig:
        return self._config

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"VectorStore(size={self.size}, dim={self._config.embedding_dim}, "
            f"M={self._config.M}, ef_search={self._config.ef_search})"
        )

    # ── add ───────────────────────────────────────────────────────────────
    def add(
        self,
        records: list[VectorRecord],
        vectors: "np.ndarray",
    ) -> None:
        """
        Append `len(records)` new entries to the index.

        Args:
            records: parallel metadata; each row corresponds to vectors[i].
            vectors: (N, dim) float32 array, ideally L2-normalized.

        Duplicate chunk_ids raise ValueError to keep the chunk_id ↔ internal_id
        mapping invariant.
        """
        import numpy as np

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if vectors.shape[0] != len(records):
            raise ValueError(
                f"vectors ({vectors.shape[0]}) and records ({len(records)}) length mismatch"
            )
        if vectors.shape[1] != self._config.embedding_dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != store dim {self._config.embedding_dim}"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Duplicate check
        for rec in records:
            if rec.chunk_id in self._chunk_id_to_internal:
                raise ValueError(f"Duplicate chunk_id: {rec.chunk_id}")

        # Append. FAISS internal ids are assigned sequentially from current size.
        start_id = self.size
        self._index.add(vectors)
        for i, rec in enumerate(records):
            self._records.append(rec)
            self._chunk_id_to_internal[rec.chunk_id] = start_id + i

    # ── search ────────────────────────────────────────────────────────────
    def search(
        self,
        query_vector: "np.ndarray",
        k: int = 10,
        filter_fn=None,
        oversample: int = 3,
    ) -> list[SearchResult]:
        """
        Return the top-k most similar records.

        Args:
            query_vector: 1-D float32 array of shape (dim,).
            k: number of final results.
            filter_fn: optional predicate `(VectorRecord) -> bool`. Results not
                matching are dropped. We over-fetch by `oversample * k` to make
                room for filtering, then truncate to k.
            oversample: multiplier for how many candidates to fetch before
                filtering. Ignored when filter_fn is None.
        """
        import numpy as np

        if self.size == 0:
            return []

        if query_vector.ndim != 1:
            raise ValueError(f"query_vector must be 1-D, got shape {query_vector.shape}")
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        fetch_k = k * oversample if filter_fn is not None else k
        fetch_k = min(fetch_k, self.size)

        scores, indices = self._index.search(
            query_vector.reshape(1, -1), fetch_k
        )

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer hits than fetch_k
                continue
            rec = self._records[idx]
            if filter_fn is not None and not filter_fn(rec):
                continue
            results.append(SearchResult(record=rec, score=float(score)))
            if len(results) >= k:
                break

        return results

    # ── retrieval by id ───────────────────────────────────────────────────
    def get(self, chunk_id: str) -> VectorRecord | None:
        internal = self._chunk_id_to_internal.get(chunk_id)
        if internal is None:
            return None
        return self._records[internal]

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_id_to_internal

    # ── ef_search runtime tuning ──────────────────────────────────────────
    def set_ef_search(self, ef_search: int) -> None:
        """
        Change query-time recall/speed trade-off without rebuilding the index.
        Larger → slower queries, better recall.
        """
        self._index.hnsw.efSearch = ef_search
        self._config.ef_search = ef_search

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, directory: str | Path) -> Path:
        """
        Persist the index + metadata to a directory.

        Files:
            <dir>/index.faiss        — the binary FAISS index
            <dir>/metadata.json      — VectorStoreConfig + all VectorRecord rows
        """
        import faiss

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dir_path / self.INDEX_FILENAME))

        payload = {
            "config": self._config.to_dict(),
            "records": [r.model_dump() for r in self._records],
        }
        (dir_path / self.METADATA_FILENAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return dir_path

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Inverse of save(). Restores a VectorStore from disk."""
        import faiss

        dir_path = Path(directory)
        meta_path = dir_path / cls.METADATA_FILENAME
        idx_path = dir_path / cls.INDEX_FILENAME
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"Missing {cls.INDEX_FILENAME} or {cls.METADATA_FILENAME} in {dir_path}"
            )

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        config = VectorStoreConfig(**payload["config"])
        store = cls(config)

        # Replace the freshly-built (empty) index with the persisted one.
        store._index = faiss.read_index(str(idx_path))
        # Restore parallel arrays
        records = [VectorRecord(**r) for r in payload["records"]]
        store._records = records
        store._chunk_id_to_internal = {r.chunk_id: i for i, r in enumerate(records)}

        # Make sure efSearch reflects the saved config (read_index restores
        # the index, but we keep our own setting authoritative)
        store._index.hnsw.efSearch = config.ef_search

        return store

    # ── iteration helpers ─────────────────────────────────────────────────
    def iter_records(self) -> Iterable[VectorRecord]:
        return iter(self._records)