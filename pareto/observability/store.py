"""
QueryLogStore — SQLite-backed log of every RAG query.

Each call to NaiveRAG.query() can be persisted as a row. The schema is
hot-field-column / cold-field-JSON: timestamps, tokens, costs, latencies,
and cache info are first-class columns for fast aggregation; citations
and free-form extras live in JSON BLOBs so we don't need a migration
each time a retriever adds a new field.

Schema evolves via idempotent migrations: `_ensure_schema()` checks
PRAGMA table_info and adds missing columns. Older DBs upgrade in place.

This is the foundation of Pareto's "built-in cost observability" claim.
Week 4+ (adaptive routing) and Week 7+ (knowledge graph) will all log
their layer-specific metadata to the same table via the `extra` slot.

Logging is opt-in and isolation-safe: a failure inside log() must NEVER
break the calling RAG query.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pareto.rag.models import RAGResponse


# ── base schema (v1) ─────────────────────────────────────────────────────

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS queries (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    question                TEXT    NOT NULL,
    retriever               TEXT,
    top_k                   INTEGER,
    model                   TEXT,
    answer                  TEXT,
    citations_json          TEXT,
    prompt_tokens           INTEGER DEFAULT 0,
    completion_tokens       INTEGER DEFAULT 0,
    total_tokens            INTEGER DEFAULT 0,
    cost_usd                REAL    DEFAULT 0.0,
    retrieval_latency_ms    INTEGER DEFAULT 0,
    generation_latency_ms   INTEGER DEFAULT 0,
    total_latency_ms        INTEGER DEFAULT 0,
    extra_json              TEXT
);

CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_model     ON queries(model);
CREATE INDEX IF NOT EXISTS idx_queries_retriever ON queries(retriever);
"""

# ── v2 additions (Week 3: cache) ─────────────────────────────────────────

_V2_COLUMNS = [
    ("cache_hit",        "INTEGER DEFAULT 0"),
    ("cache_similarity", "REAL"),
]

_V2_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_queries_cache_hit ON queries(cache_hit)",
]


# ── config ───────────────────────────────────────────────────────────────

@dataclass
class QueryLogConfig:
    db_path: Path = Path("benchmarks/results/pareto.db")
    enabled: bool = True

    def __post_init__(self):
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)


# ── store ────────────────────────────────────────────────────────────────

class QueryLogStore:
    """SQLite-backed log of RAG queries. Thread-safe via per-call connections."""

    def __init__(self, config: QueryLogConfig | None = None):
        self.config = config or QueryLogConfig()
        if self.config.enabled:
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Idempotent schema setup + migrations."""
        with self._connection() as conn:
            # v1 base
            conn.executescript(_SCHEMA_V1)
            # v2 cache columns (additive, safe to run repeatedly)
            existing = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(queries)").fetchall()
            }
            for col_name, col_def in _V2_COLUMNS:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE queries ADD COLUMN {col_name} {col_def}")
            for idx_sql in _V2_INDEXES:
                conn.execute(idx_sql)

    # ── write ─────────────────────────────────────────────────────────────
    def log(
        self,
        response: "RAGResponse",
        retriever: str = "unknown",
        top_k: int | None = None,
    ) -> int | None:
        """
        Persist one RAGResponse. Returns the new row id, or None if disabled
        or on failure. Cache info is extracted from response.extra.
        """
        if not self.config.enabled:
            return None

        try:
            citations = response.citations()
            cache_hit = bool(response.extra.get("cache_hit", False))
            cache_similarity = response.extra.get("cache_similarity")

            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO queries (
                        timestamp, question, retriever, top_k, model, answer,
                        citations_json, prompt_tokens, completion_tokens,
                        total_tokens, cost_usd, retrieval_latency_ms,
                        generation_latency_ms, total_latency_ms, extra_json,
                        cache_hit, cache_similarity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        response.question,
                        retriever,
                        top_k,
                        response.model,
                        response.answer,
                        json.dumps(citations, ensure_ascii=False),
                        response.prompt_tokens,
                        response.completion_tokens,
                        response.total_tokens,
                        response.cost_usd,
                        response.retrieval_latency_ms,
                        response.generation_latency_ms,
                        response.total_latency_ms,
                        json.dumps(response.extra, ensure_ascii=False)
                            if response.extra else None,
                        1 if cache_hit else 0,
                        float(cache_similarity) if cache_similarity is not None else None,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[pareto-observability] log() failed: {e}", file=sys.stderr)
            return None

    # ── read ──────────────────────────────────────────────────────────────
    def last_n(self, n: int = 10) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def total_count(self) -> int:
        if not self.config.enabled:
            return 0
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM queries").fetchone()
        return row["c"] if row else 0

    def query_range(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        retriever: str | None = None,
        model: str | None = None,
        cache_hit: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())
        if retriever is not None:
            clauses.append("retriever = ?")
            params.append(retriever)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if cache_hit is not None:
            clauses.append("cache_hit = ?")
            params.append(1 if cache_hit else 0)

        sql = "SELECT * FROM queries"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── aggregate ─────────────────────────────────────────────────────────
    def aggregate(
        self,
        since: datetime | None = None,
        group_by: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Aggregate stats grouped by a column (e.g. 'model', 'retriever').
        Cache metrics included for every group.
        """
        if not self.config.enabled:
            return {}

        # Build metric expressions. Cache metrics:
        #   - cache_hits: count where cache_hit = 1
        #   - cache_hit_rate: hits / total
        #   - avg_cache_similarity: only over hits
        #   - total_saved_cost_usd: sum of original_cost_usd in extra for hits
        #     (we approximate by summing the saved_cost expression below)
        metric_cols = """
            COUNT(*) AS count,
            SUM(prompt_tokens) AS total_prompt_tokens,
            SUM(completion_tokens) AS total_completion_tokens,
            SUM(total_tokens) AS total_tokens,
            SUM(cost_usd) AS total_cost_usd,
            AVG(cost_usd) AS avg_cost_usd,
            AVG(retrieval_latency_ms) AS avg_retrieval_latency_ms,
            AVG(generation_latency_ms) AS avg_generation_latency_ms,
            AVG(total_latency_ms) AS avg_total_latency_ms,
            MAX(total_latency_ms) AS max_total_latency_ms,
            SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits,
            AVG(CASE WHEN cache_hit = 1 THEN cache_similarity ELSE NULL END)
                AS avg_cache_similarity
        """

        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""

        if group_by is None:
            sql = f"SELECT {metric_cols} FROM queries{where}"
            with self._connection() as conn:
                row = conn.execute(sql, params).fetchone()
            stats = dict(row) if row else {}
            stats["cache_hit_rate"] = self._compute_hit_rate(stats)
            return {"overall": stats}

        if group_by not in {"model", "retriever", "cache_hit"}:
            raise ValueError(
                f"group_by must be 'model', 'retriever', or 'cache_hit', got {group_by!r}"
            )

        sql = f"SELECT {group_by}, {metric_cols} FROM queries{where} GROUP BY {group_by}"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        result = {}
        for row in rows:
            key = row[group_by]
            if group_by == "cache_hit":
                key = "hit" if key == 1 else "miss"
            elif key is None:
                key = "<null>"
            stats = dict(row)
            stats["cache_hit_rate"] = self._compute_hit_rate(stats)
            result[str(key)] = stats
        return result

    @staticmethod
    def _compute_hit_rate(stats: dict[str, Any]) -> float:
        """Helper: cache_hits / count."""
        count = stats.get("count") or 0
        hits = stats.get("cache_hits") or 0
        return (hits / count) if count > 0 else 0.0

    # ── slow queries (cache-aware) ────────────────────────────────────────
    def slow_queries(
        self,
        threshold_ms: int = 30_000,
        limit: int = 10,
        exclude_cache_hits: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Find slowest queries above threshold.
        By default, cache hits are excluded — they're always fast and
        cluttering them in slow analysis defeats the purpose.
        """
        if not self.config.enabled:
            return []

        clauses = ["total_latency_ms >= ?"]
        params: list[Any] = [threshold_ms]
        if exclude_cache_hits:
            clauses.append("(cache_hit IS NULL OR cache_hit = 0)")

        sql = (
            f"SELECT * FROM queries WHERE {' AND '.join(clauses)} "
            f"ORDER BY total_latency_ms DESC LIMIT ?"
        )
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── savings projection ────────────────────────────────────────────────
    def total_savings(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Estimate cost + latency saved by cache hits.
        Reads `original_cost_usd` and `original_generation_latency_ms`
        from the extra_json blob on cache-hit rows.
        """
        if not self.config.enabled:
            return {"hits": 0, "saved_cost_usd": 0.0, "saved_latency_ms": 0}

        clauses = ["cache_hit = 1"]
        params: list[Any] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())

        sql = f"SELECT extra_json FROM queries WHERE {' AND '.join(clauses)}"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        saved_cost = 0.0
        saved_latency = 0
        for row in rows:
            try:
                extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
            except (json.JSONDecodeError, TypeError):
                continue
            saved_cost += float(extra.get("original_cost_usd") or 0)
            saved_latency += int(extra.get("original_generation_latency_ms") or 0)

        return {
            "hits": len(rows),
            "saved_cost_usd": saved_cost,
            "saved_latency_ms": saved_latency,
        }

    def __repr__(self) -> str:
        return (
            f"QueryLogStore(db={self.config.db_path}, "
            f"enabled={self.config.enabled}, "
            f"count={self.total_count()})"
        )