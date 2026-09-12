"""OpenAI GPT-5.4 client (v2, Responses API + strict json_schema).

Asadi §7.1 parameters verbatim for mirage-mode: reasoning_effort="medium",
temperature=1.0 (temp=0 is unsupported on thinking models). CoT captured
via reasoning.summary="auto".
"""
from __future__ import annotations

import json
import os
import time

from openai import OpenAI, APIError

from .base import BaseClient, CallResult
from schema import DIAGNOSIS_SCHEMA, SCHEMA_NAME, is_valid_payload


class OpenAIClient(BaseClient):
    name = "openai"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.text_format = {
            "format": {
                "type": "json_schema",
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": DIAGNOSIS_SCHEMA,
            }
        }

    def call(self, prompt: str, seed: int | None = None) -> CallResult:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "reasoning": {
                "effort": self.cfg.get("reasoning_effort", "medium"),
                "summary": self.cfg.get("reasoning_summary", "auto"),
            },
            "max_output_tokens": self.cfg.get("max_output_tokens", 4000),
            "temperature": self.cfg.get("temperature", 1.0),
            "text": self.text_format,
        }

        start = time.perf_counter()
        try:
            resp = self.client.responses.create(**kwargs)
        except APIError as e:
            return CallResult(
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        text = resp.output_text or ""
        parsed: dict | None = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

        reasoning_parts: list[str] = []
        for item in (resp.output or []):
            if getattr(item, "type", None) == "reasoning":
                for s in (getattr(item, "summary", None) or []):
                    t = getattr(s, "text", None)
                    if t:
                        reasoning_parts.append(t)
        thinking = "\n".join(reasoning_parts) if reasoning_parts else None

        usage = resp.usage
        return CallResult(
            text=text,
            parsed=parsed,
            input_tokens=(usage.input_tokens if usage else 0),
            output_tokens=(usage.output_tokens if usage else 0),
            latency_ms=elapsed_ms,
            raw={"model": getattr(resp, "model", self.model)},
            thinking=thinking,
            schema_valid=is_valid_payload(parsed),
        )
