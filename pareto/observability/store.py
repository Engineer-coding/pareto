"""
QueryLogStore — SQLite-backed log of every RAG query.

Each call to NaiveRAG.query() can be persisted as a row. The schema is
hot-field-column / cold-field-JSON: timestamps, tokens, costs, and
latencies are first-class columns for fast aggregation; citations and
free-form extras live in JSON BLOBs so we don't need a migration each
time a retriever adds a new field.

This is the foundation of Pareto's "built-in cost observability" claim.
Week 4+ (adaptive routing) and Week 7+ (knowledge graph) will all log
their layer-specific metadata to the same table via the `extra` slot.

Logging is opt-in and isolation-safe: a failure inside log() must NEVER
break the calling RAG query. The try/except in NaiveRAG ensures that.
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


# ── schema ───────────────────────────────────────────────────────────────

_SCHEMA = """
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


# ── config ───────────────────────────────────────────────────────────────

@dataclass
class QueryLogConfig:
    db_path: Path = Path("benchmarks/results/pareto.db")
    enabled: bool = True

    def __post_init__(self):
        # Allow string paths from CLI / env
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
        """Per-call connection. SQLite handles concurrency at the file level."""
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row  # dict-like access on rows
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(_SCHEMA)

    # ── write ─────────────────────────────────────────────────────────────
    def log(
        self,
        response: "RAGResponse",
        retriever: str = "unknown",
        top_k: int | None = None,
    ) -> int | None:
        """
        Persist one RAGResponse. Returns the new row id, or None if disabled.

        This must never raise — callers wrap in try/except just in case,
        but we also defend internally. A logging failure is not a feature
        failure.
        """
        if not self.config.enabled:
            return None

        try:
            citations = response.citations()
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO queries (
                        timestamp, question, retriever, top_k, model, answer,
                        citations_json, prompt_tokens, completion_tokens,
                        total_tokens, cost_usd, retrieval_latency_ms,
                        generation_latency_ms, total_latency_ms, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )
                return cursor.lastrowid
        except Exception as e:  # noqa: BLE001
            # Logging must not break the calling query. Print for visibility,
            # but don't re-raise.
            import sys
            print(f"[pareto-observability] log() failed: {e}", file=sys.stderr)
            return None

    # ── read ──────────────────────────────────────────────────────────────
    def last_n(self, n: int = 10) -> list[dict[str, Any]]:
        """Most recent N queries, newest first."""
        if not self.config.enabled:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?",
                (n,),
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
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Filtered read. Filters compose with AND."""
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
        Returns {group_value: {count, total_cost_usd, avg_total_latency_ms, ...}}.

        If group_by is None, returns {"overall": {...}}.
        """
        if not self.config.enabled:
            return {}

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
            MAX(total_latency_ms) AS max_total_latency_ms
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
            return {"overall": dict(row) if row else {}}

        if group_by not in {"model", "retriever"}:
            raise ValueError(f"group_by must be 'model' or 'retriever', got {group_by!r}")

        sql = f"SELECT {group_by}, {metric_cols} FROM queries{where} GROUP BY {group_by}"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {row[group_by] or "<null>": dict(row) for row in rows}

    # ── slow queries ──────────────────────────────────────────────────────
    def slow_queries(
        self,
        threshold_ms: int = 30_000,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find slowest queries above threshold. Useful for debug."""
        if not self.config.enabled:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM queries
                WHERE total_latency_ms >= ?
                ORDER BY total_latency_ms DESC
                LIMIT ?
                """,
                (threshold_ms, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def __repr__(self) -> str:
        return (
            f"QueryLogStore(db={self.config.db_path}, "
            f"enabled={self.config.enabled}, "
            f"count={self.total_count()})"
        )