"""Gemini 3.1 Pro Preview client (v2, Vertex AI + response_schema + HIGH thinking).

Asadi §7.1 parameters: Thinking_Level="high" (options are "high" and "low"
for Gemini-3-Pro). include_thoughts=True captures thought parts which we
save as CoT. response_schema enforces structured output without regex.
"""
from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import types as genai_types

from .base import BaseClient, CallResult
from schema import GEMINI_DIAGNOSIS_SCHEMA, is_valid_payload


class GeminiVertexClient(BaseClient):
    name = "gemini"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "vlmsee")
        location = cfg.get("location", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def _build_config(self, seed: int | None) -> genai_types.GenerateContentConfig:
        thinking_kwargs = {"include_thoughts": bool(self.cfg.get("include_thoughts", True))}
        level = self.cfg.get("thinking_level")
        if level is not None:
            thinking_kwargs["thinking_level"] = level
        return genai_types.GenerateContentConfig(
            temperature=self.cfg.get("temperature", 1.0),
            top_p=self.cfg.get("top_p", 1.0),
            max_output_tokens=self.cfg.get("max_output_tokens", 4000),
            seed=seed,
            response_mime_type="application/json",
            response_schema=GEMINI_DIAGNOSIS_SCHEMA,
            thinking_config=genai_types.ThinkingConfig(**thinking_kwargs),
        )

    def call(self, prompt: str, seed: int | None = None) -> CallResult:
        start = time.perf_counter()
        try:
            cfg = self._build_config(seed)
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=cfg,
            )
        except Exception as e:
            return CallResult(
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        cand_tok = getattr(usage, "candidates_token_count", 0) or 0
        thoughts_tok = getattr(usage, "thoughts_token_count", 0) or 0

        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                t = getattr(part, "text", None)
                if not t:
                    continue
                if getattr(part, "thought", False):
                    thinking_parts.append(t)
                else:
                    visible_parts.append(t)
            break  # first candidate only
        text = "\n".join(visible_parts) if visible_parts else (getattr(resp, "text", "") or "")
        thinking = "\n".join(thinking_parts) if thinking_parts else None

        parsed: dict | None = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

        return CallResult(
            text=text,
            parsed=parsed,
            input_tokens=in_tok,
            output_tokens=cand_tok + thoughts_tok,
            latency_ms=elapsed_ms,
            raw={"model": self.model, "thoughts_tokens": thoughts_tok, "visible_tokens": cand_tok},
            thinking=thinking,
            schema_valid=is_valid_payload(parsed),
        )
