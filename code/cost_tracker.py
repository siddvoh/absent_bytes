"""Per-call cost logging and real-time budget watchdog."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from utils import CallRecord, cost_log_path, append_jsonl


def log_cost(record: CallRecord) -> None:
    append_jsonl(cost_log_path(), {
        "timestamp": record.timestamp,
        "provider": record.provider,
        "model": record.model,
        "experiment": record.experiment,
        "domain": record.domain,
        "demographic": record.demographic,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "cost_usd": record.cost_usd,
    })


def current_spend() -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    path = cost_log_path()
    if not path.exists():
        return dict(totals)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            totals[r["provider"]] += float(r.get("cost_usd", 0.0))
            totals["_total"] += float(r.get("cost_usd", 0.0))
    return dict(totals)


def budget_status(budgets: dict[str, float] | None = None) -> dict:
    budgets = budgets or {
        "anthropic": 50.0,
        "openai": 25.0,
        "vertex": 25.0,
        "_total": 100.0,
    }
    spend = current_spend()
    return {
        provider: {
            "spend": round(spend.get(provider, 0.0), 4),
            "budget": budget,
            "remaining": round(budget - spend.get(provider, 0.0), 4),
            "pct_used": round(100 * spend.get(provider, 0.0) / budget, 1) if budget else 0.0,
        }
        for provider, budget in budgets.items()
    }


def estimate_cost(input_tokens: int, output_tokens: int, cfg: dict) -> float:
    return (
        input_tokens * cfg["input_price_per_mtok"] / 1_000_000
        + output_tokens * cfg["output_price_per_mtok"] / 1_000_000
    )
