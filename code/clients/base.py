"""Abstract base client interface for VLM providers (v2, JSON-schema protocol)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CallResult:
    text: str = ""                     # raw serialized JSON from provider
    parsed: dict | None = None         # parsed JSON payload (schema-conformant)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    raw: dict | None = None            # provider metadata (stop_reason, model id, etc.)
    error: str | None = None
    thinking: str | None = None        # full CoT / reasoning summary where exposed
    schema_valid: bool = False         # passed src.schema.is_valid_payload


class BaseClient(ABC):
    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]

    @abstractmethod
    def call(self, prompt: str, seed: int | None = None) -> CallResult:
        """Synchronous single-call. Returns CallResult; never raises on API errors."""

    def estimate_cost(self, in_tok: int, out_tok: int) -> float:
        return (
            in_tok * self.cfg["input_price_per_mtok"] / 1_000_000
            + out_tok * self.cfg["output_price_per_mtok"] / 1_000_000
        )
