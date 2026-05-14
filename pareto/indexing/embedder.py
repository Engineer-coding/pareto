"""
Embedding layer — turn text into dense vectors.

Design:
    * BaseEmbedder is the abstract contract.
    * SentenceTransformerEmbedder is the local default (E5-multilingual-small).
    * Later additions (OpenAI, Cohere, Voyage) just implement BaseEmbedder.

E5-family models REQUIRE prefix tokens to work properly:
    "query: ..."   for retrieval queries
    "passage: ..." for indexed documents

Without these prefixes the model performs roughly 15-20% worse on retrieval
benchmarks. We encapsulate the prefix logic so callers never see it.

All embeddings are L2-normalized so cosine similarity == dot product.
This matches FAISS HNSW IP (Inner Product) index expectations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ── prefix table for known model families ─────────────────────────────────
# Maps a prefix of the model name to (query_prefix, passage_prefix).
_PREFIX_TABLE: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "intfloat/e5":              ("query: ", "passage: "),
    "BAAI/bge":                 ("Represent this sentence for searching relevant passages: ", ""),
}


def _prefixes_for(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) for the given model, or ('', '')."""
    for key, prefixes in _PREFIX_TABLE.items():
        if model_name.startswith(key):
            return prefixes
    return "", ""


class BaseEmbedder(ABC):
    """Abstract embedder. Concrete subclasses produce L2-normalized vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model — used for caching keys."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of vectors produced by this embedder."""

    @abstractmethod
    def encode_query(self, query: str) -> "np.ndarray":
        """Encode a single query string into a 1-D numpy array."""

    @abstractmethod
    def encode_passages(
        self,
        passages: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> "np.ndarray":
        """Encode N passages into an (N, dim) numpy array."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedder backed by sentence-transformers.

    Default model: intfloat/multilingual-e5-small (384-dim, 100+ languages
    including Turkish, MIT license). Override via the `model_name` argument.
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-small"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ):
        # Heavy imports deferred to runtime
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._query_prefix, self._passage_prefix = _prefixes_for(model_name)

    # ── BaseEmbedder API ──────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_query(self, query: str) -> "np.ndarray":
        prefixed = f"{self._query_prefix}{query}" if self._query_prefix else query
        vec = self._model.encode(
            prefixed,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        # ST returns either a tensor or numpy; coerce to numpy.
        import numpy as np
        return np.asarray(vec, dtype=np.float32)

    def encode_passages(
        self,
        passages: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> "np.ndarray":
        if self._passage_prefix:
            inputs = [f"{self._passage_prefix}{p}" for p in passages]
        else:
            inputs = passages
        vecs = self._model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_progress,
        )
        import numpy as np
        return np.asarray(vecs, dtype=np.float32)

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder(model={self._model_name!r}, dim={self._dim})"