"""SQLite audit log for all paid API calls."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from clients.base import CallResult
from utils import LOGS


RAW_RESPONSE_CHAR_CAP = 10_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    experiment TEXT,
    model TEXT,
    provider TEXT,
    endpoint_family TEXT,
    domain TEXT,
    demographic TEXT,
    seed INTEGER,
    output_mode TEXT,
    prompt_variant TEXT,
    control_family TEXT,
    source_cell TEXT,
    prompt_text TEXT,
    prompt_hash TEXT,
    system_prompt TEXT,
    parity_temperature REAL,
    parity_max_output_tokens INTEGER,
    parity_reasoning_effort TEXT,
    request_json TEXT,
    response_text TEXT,
    response_json TEXT,
    raw_response TEXT,
    thinking TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cost_usd REAL,
    latency_ms INTEGER,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    schema_valid INTEGER,
    record_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_calls_experiment ON api_calls(experiment);
CREATE INDEX IF NOT EXISTS idx_api_calls_provider ON api_calls(provider);
CREATE INDEX IF NOT EXISTS idx_api_calls_model ON api_calls(model);
CREATE INDEX IF NOT EXISTS idx_api_calls_prompt_hash ON api_calls(prompt_hash);
DROP INDEX IF EXISTS idx_api_calls_unique_record;
"""


class AuditLogger:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else LOGS / "api_calls.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass
        return conn

    def log_call(
        self,
        *,
        result: CallResult,
        context: dict[str, Any],
        cost_usd: float,
        record_path: str,
    ) -> int:
        row = (
            context.get("timestamp"),
            context.get("experiment"),
            context.get("model"),
            context.get("provider"),
            context.get("endpoint_family"),
            context.get("domain"),
            context.get("demographic"),
            context.get("seed"),
            context.get("output_mode"),
            context.get("prompt_variant"),
            context.get("control_family"),
            context.get("source_cell"),
            context.get("prompt_text"),
            context.get("prompt_hash"),
            context.get("system_prompt"),
            context.get("parity_temperature"),
            context.get("parity_max_output_tokens"),
            context.get("parity_reasoning_effort"),
            json.dumps(result.request or {}, ensure_ascii=False, default=str, sort_keys=True),
            result.text,
            json.dumps(result.parsed, ensure_ascii=False, default=str, sort_keys=True) if result.parsed is not None else None,
            json.dumps(result.raw or {}, ensure_ascii=False, default=str)[:RAW_RESPONSE_CHAR_CAP],
            result.thinking,
            int(result.input_tokens),
            int(result.output_tokens),
            int(context.get("cache_read_tokens", 0) or 0),
            int(context.get("cache_creation_tokens", 0) or 0),
            float(cost_usd),
            int(result.latency_ms),
            result.error,
            int(result.retry_count or context.get("retry_count", 0) or 0),
            1 if result.schema_valid else 0,
            record_path,
        )
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with self._lock, self._connect() as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO api_calls (
                            timestamp, experiment, model, provider, endpoint_family,
                            domain, demographic, seed, output_mode, prompt_variant,
                            control_family, source_cell, prompt_text, prompt_hash,
                            system_prompt, parity_temperature, parity_max_output_tokens,
                            parity_reasoning_effort, request_json, response_text,
                            response_json, raw_response, thinking, input_tokens,
                            output_tokens, cache_read_tokens, cache_creation_tokens,
                            cost_usd, latency_ms, error, retry_count, schema_valid,
                            record_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                    conn.commit()
                    return int(cur.lastrowid or 0)
            except sqlite3.OperationalError as e:
                last_err = e
                time.sleep(0.25 * (2 ** attempt))
        raise RuntimeError(f"audit log write failed: {last_err}")


def audit_db_path() -> Path:
    return LOGS / "api_calls.sqlite"
