"""Anthropic Claude Opus 4.7 client (v2, JSON-schema via forced tool_use).

Key API constraint confirmed by probe: thinking cannot coexist with
tool_choice that forces a tool. So we run WITHOUT any thinking config and
accept that Claude will not emit thinking blocks here. Asadi §7.1 likewise
accessed Opus 4.5 on Vertex with default parameters (no thinking knob).
The alignment is complete but coincidental.
"""
from __future__ import annotations

import json
import os
import time

from anthropic import Anthropic, APIError

from .base import BaseClient, CallResult
from schema import DIAGNOSIS_SCHEMA, is_valid_payload


TOOL_DESCRIPTION = "Record the diagnostic response."


class AnthropicClient(BaseClient):
    name = "anthropic"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tool_name = cfg.get("tool_name", "record_diagnosis")
        self.tool = {
            "name": self.tool_name,
            "description": TOOL_DESCRIPTION,
            "input_schema": DIAGNOSIS_SCHEMA,
        }

    def call(self, prompt: str, seed: int | None = None) -> CallResult:
        kwargs = {
            "model": self.model,
            "max_tokens": self.cfg.get("max_tokens", 4000),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.get("temperature", 1.0),
            "tools": [self.tool],
            "tool_choice": {"type": "tool", "name": self.tool_name},
        }
        # top_p is deprecated on Opus 4.7 and returns 400.

        start = time.perf_counter()
        try:
            resp = self.client.messages.create(**kwargs)
        except APIError as e:
            return CallResult(
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        parsed: dict | None = None
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == self.tool_name:
                payload = block.input
                parsed = dict(payload) if isinstance(payload, dict) else None
                text = json.dumps(parsed, ensure_ascii=False) if parsed is not None else ""
                break

        return CallResult(
            text=text,
            parsed=parsed,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=elapsed_ms,
            raw={"stop_reason": resp.stop_reason, "model": resp.model},
            thinking=None,  # always None under forced tool_use
            schema_valid=is_valid_payload(parsed),
        )
