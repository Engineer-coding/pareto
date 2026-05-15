"""
Test set loader.

The benchmark YAML format is intentionally hand-editable: every query is a
top-level list item, keys map directly to BenchmarkQuery fields. Loading is
a thin Pydantic-validated read.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pareto.benchmark.models import BenchmarkQuery


class TestSet:
    """A collection of BenchmarkQuery items, addressable by id or domain."""

    def __init__(self, name: str, queries: list[BenchmarkQuery]):
        self.name = name
        self.queries = queries

    # ── factories ─────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path, name: str | None = None) -> "TestSet":
        """Load a test set from a single YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Test set YAML not found: {p}")

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"Test set must be a YAML list at root, got {type(raw).__name__}")

        queries = [BenchmarkQuery(**item) for item in raw]
        cls._validate_unique_ids(queries)

        return cls(name=name or p.stem, queries=queries)

    # ── access ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def by_domain(self, domain: str) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.domain == domain]

    def domains(self) -> list[str]:
        seen: list[str] = []
        for q in self.queries:
            if q.domain not in seen:
                seen.append(q.domain)
        return seen

    def summary(self) -> dict[str, int]:
        """Counts per domain and per type — useful as a sanity print."""
        by_domain: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        for q in self.queries:
            by_domain[q.domain] = by_domain.get(q.domain, 0) + 1
            by_type[q.type.value] = by_type.get(q.type.value, 0) + 1
            by_difficulty[q.difficulty.value] = by_difficulty.get(q.difficulty.value, 0) + 1
        return {
            "total": len(self.queries),
            "by_domain": by_domain,
            "by_type": by_type,
            "by_difficulty": by_difficulty,
        }

    # ── internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate_unique_ids(queries: list[BenchmarkQuery]) -> None:
        seen: set[str] = set()
        for q in queries:
            if q.id in seen:
                raise ValueError(f"Duplicate query id in test set: {q.id}")
            seen.add(q.id)