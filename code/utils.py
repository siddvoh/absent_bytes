"""Shared utilities."""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from dataclasses import asdict, dataclass, field

import yaml
from dotenv import load_dotenv

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_CAT = REPO_ROOT / "data" / "categorized"
DATA_AGG = REPO_ROOT / "data" / "aggregated"
LOGS = REPO_ROOT / "logs"

def load_env() -> None:
    for _p in (DATA_RAW, DATA_CAT, DATA_AGG, LOGS, LOGS / "run_summaries"):
        _p.mkdir(parents=True, exist_ok=True)
    load_dotenv(REPO_ROOT / ".env")


def load_yaml(path: pathlib.Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_models_config() -> dict:
    return load_yaml(CONFIGS / "models.yaml")


def load_prompts_config() -> dict:
    return load_yaml(CONFIGS / "prompts.yaml")


def load_judge_config() -> dict:
    return load_yaml(CONFIGS / "judge_prompt.yaml")


def load_taxonomy(domain: str) -> dict:
    return load_yaml(CONFIGS / "taxonomies" / f"{domain}.yaml")


def load_seeds() -> list[int]:
    return load_yaml(CONFIGS / "seeds.yaml")["seeds"]


@dataclass
class CallRecord:
    """One VLM call. Persisted as a single JSONL line."""
    timestamp: str
    model: str
    provider: str
    domain: str
    demographic: str
    seed: int
    experiment: str
    prompt: str
    response_text: str           # raw serialized JSON from provider
    response_json: dict | None   # parsed structured output (schema-conformant)
    schema_valid: bool           # passed src.schema.is_valid_payload
    thinking: str | None         # full CoT / reasoning summary where available
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    provider_meta: dict | None = None   # stop_reason, thoughts_tokens, etc.
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: pathlib.Path | str, record: CallRecord | dict) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.to_json() if isinstance(record, CallRecord) else json.dumps(record, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")


def cost_log_path() -> pathlib.Path:
    return LOGS / "cost_log.jsonl"


def error_log_path() -> pathlib.Path:
    return LOGS / "error_log.jsonl"


def raw_path(model_slug: str, domain: str, demographic: str, experiment: str = "E1") -> pathlib.Path:
    return DATA_RAW / f"{experiment}_{model_slug}_{domain}_{demographic}.jsonl"


MODEL_SLUGS = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-5.4",
    "gemini": "gemini-3.1-pro-preview",
}
