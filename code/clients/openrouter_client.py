"""Emergency OpenRouter fallback for primary provider key/quota failures."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from .base import BaseClient, CallResult, OutputMode
from schema import DIAGNOSIS_SCHEMA, SCHEMA_NAME, is_valid_payload


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterFallbackClient(BaseClient):
    name = "openrouter"

    def __init__(self, cfg: dict, *, fallback_for: str):
        super().__init__(cfg)
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=120.0)
        self.fallback_for = fallback_for

    @staticmethod
    def _provider_preferences(cfg: dict[str, Any]) -> dict[str, Any] | None:
        prefs = cfg.get("openrouter_provider_preferences")
        if prefs:
            allowed = {
                "order",
                "allow_fallbacks",
                "require_parameters",
                "data_collection",
                "zdr",
                "enforce_distillable_text",
                "only",
                "ignore",
                "quantizations",
                "sort",
                "preferred_min_throughput",
                "preferred_max_latency",
                "max_price",
            }
            return {k: v for k, v in dict(prefs).items() if k in allowed}
        provider_only = cfg.get("openrouter_provider_only")
        if provider_only:
            return {"only": list(provider_only)}
        return None

    @classmethod
    def _extra_body(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"usage": {"include": True}}
        if not cfg.get("disable_thinking"):
            if cfg.get("reasoning_max_tokens") is not None:
                extra_body["reasoning"] = {"max_tokens": int(cfg["reasoning_max_tokens"])}
            else:
                extra_body["reasoning"] = {"effort": str(cfg.get("reasoning_effort", "medium")).lower()}
            if cfg.get("reasoning_exclude"):
                extra_body["reasoning"]["exclude"] = True
            if cfg.get("include_reasoning", True):
                extra_body["include_reasoning"] = True
        provider_prefs = cls._provider_preferences(cfg)
        if provider_prefs:
            extra_body["provider"] = provider_prefs
        return extra_body

    @staticmethod
    def _parse_jsonish(text: str) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            parsed, _idx = json.JSONDecoder().raw_decode(text.lstrip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _schema_instruction(
        prompt: str,
        schema: dict[str, Any] | None = None,
        instruction_override: str | None = None,
    ) -> str:
        if instruction_override:
            return prompt + "\n\n" + instruction_override
        schema = schema or DIAGNOSIS_SCHEMA
        return (
            prompt
            + "\n\nThe required JSON schema is:\n"
            + json.dumps(schema, ensure_ascii=False)
            + "\nReturn exactly one JSON object with exactly the required keys. "
            "Use null for primary_diagnosis when no diagnosis can be made. "
            "Do not use alternate key names such as final_diagnosis, report, impression, or findings."
        )

    @staticmethod
    def _extract_reasoning(choice: Any, raw: dict | None) -> str | None:
        parts: list[str] = []
        if choice is not None and getattr(choice, "message", None) is not None:
            msg = choice.message
            for attr in ("reasoning", "reasoning_content"):
                val = getattr(msg, attr, None)
                if val:
                    parts.append(str(val))
            details = getattr(msg, "reasoning_details", None)
            if details:
                parts.append(json.dumps(details, ensure_ascii=False, default=str))
        if raw:
            try:
                msg = ((raw.get("choices") or [{}])[0].get("message") or {})
                for key in ("reasoning", "reasoning_content"):
                    if msg.get(key):
                        parts.append(str(msg[key]))
                if msg.get("reasoning_details"):
                    parts.append(json.dumps(msg["reasoning_details"], ensure_ascii=False, default=str))
            except Exception:
                pass
        return "\n".join(p for p in parts if p) or None

    @staticmethod
    def _is_depleted_byok_error(error: Exception) -> bool:
        msg = str(error).lower()
        return (
            "prepayment credits are depleted" in msg
            or "resource_exhausted" in msg
            or "byok" in msg and "429" in msg
        )

    @staticmethod
    def _tool_arguments(choice: Any, raw: dict | None) -> str | None:
        if choice is not None and getattr(choice, "message", None) is not None:
            calls = getattr(choice.message, "tool_calls", None)
            if calls:
                fn = getattr(calls[0], "function", None)
                args = getattr(fn, "arguments", None) if fn is not None else None
                if args:
                    return str(args)
        if raw:
            try:
                calls = (((raw.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
                fn = (calls[0] or {}).get("function") if calls else None
                args = (fn or {}).get("arguments")
                if args:
                    return str(args)
            except Exception:
                pass
        return None

    @staticmethod
    def _request_snapshot(kwargs: dict[str, Any], extra_body: dict[str, Any], output_mode: OutputMode, system_prompt: str | None) -> dict[str, Any]:
        return {
            "endpoint_family": "openrouter",
            "model": kwargs.get("model"),
            "output_mode": output_mode,
            "messages": kwargs.get("messages"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "seed": kwargs.get("seed"),
            "response_format": kwargs.get("response_format"),
            "reasoning": extra_body.get("reasoning"),
            "include_reasoning": extra_body.get("include_reasoning"),
            "provider_routing": extra_body.get("provider"),
            "usage_accounting": extra_body.get("usage"),
            "system_prompt_present": bool(system_prompt),
        }

    def call(
        self,
        prompt: str,
        seed: int | None = None,
        output_mode: OutputMode = "strict_json",
        system_prompt: str | None = None,
        image_data_uri: str | None = None,
    ) -> CallResult:
        schema_override = self.cfg.get("diagnosis_schema_override") or DIAGNOSIS_SCHEMA
        schema_name = self.cfg.get("schema_name_override") or SCHEMA_NAME
        instruction_override = self.cfg.get("schema_instruction_override")
        request_prompt = (
            self._schema_instruction(prompt, schema_override, instruction_override)
            if output_mode == "strict_json"
            else prompt
        )
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if image_data_uri:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            })
        else:
            messages.append({"role": "user", "content": request_prompt})
        extra_body = self._extra_body(self.cfg)
        provider_prefs = extra_body.get("provider")
        temperature = self.cfg.get("temperature", 1.0)
        omit_temperature = self.fallback_for == "anthropic" and "claude-opus-4.7" in self.model
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.cfg.get("max_output_tokens", self.cfg.get("max_tokens", 4000)),
            "extra_body": extra_body,
        }
        if not omit_temperature:
            kwargs["temperature"] = temperature
        if output_mode == "strict_json":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema_override,
                },
            }
        elif output_mode == "loose_json" and not self.cfg.get("disable_response_format"):
            kwargs["response_format"] = {"type": "json_object"}
        if seed is not None:
            kwargs["seed"] = seed
        request_snapshot = self._request_snapshot(kwargs, extra_body, output_mode, system_prompt)
        request_snapshot["openrouter_byok_required"] = self.cfg.get("openrouter_byok_required")
        request_snapshot["openrouter_byok_verification"] = self.cfg.get("openrouter_byok_verification")

        start = time.perf_counter()
        provider_pin_error: str | None = None
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            pinned_no_fallback = bool(provider_prefs and provider_prefs.get("allow_fallbacks") is False)
            byok_required = bool(self.cfg.get("openrouter_byok_required"))
            if provider_prefs and self._is_depleted_byok_error(e) and not byok_required and not pinned_no_fallback:
                provider_pin_error = f"{type(e).__name__}: {e}"
                retry_kwargs = dict(kwargs)
                retry_extra = dict(extra_body)
                retry_extra.pop("provider", None)
                retry_kwargs["extra_body"] = retry_extra
                try:
                    resp = self.client.chat.completions.create(**retry_kwargs)
                    kwargs = retry_kwargs
                    extra_body = retry_extra
                    request_snapshot = self._request_snapshot(kwargs, extra_body, output_mode, system_prompt)
                except Exception as e2:
                    e = e2
                else:
                    pass
            if "resp" not in locals():
                request_snapshot["fallback_for"] = self.fallback_for
                request_snapshot["temperature_omitted_reason"] = "claude_opus_4_7_sampling_unsupported" if omit_temperature else None
                request_snapshot["provider_pin_error"] = provider_pin_error
                return CallResult(
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error=f"{type(e).__name__}: {e}",
                    output_mode=output_mode,
                    request=request_snapshot,
                )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        choice = resp.choices[0] if resp.choices else None
        text = ""
        if choice is not None and getattr(choice, "message", None) is not None:
            text = getattr(choice.message, "content", None) or ""
        usage = getattr(resp, "usage", None)
        try:
            raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {"text": text}
        except Exception:
            raw = {"text": text}
        provider_error = None
        if isinstance(raw, dict) and raw.get("error"):
            provider_error = json.dumps(raw["error"], ensure_ascii=False, default=str)
        actual_cost = None
        if usage is not None and getattr(usage, "cost", None) is not None:
            actual_cost = float(getattr(usage, "cost"))
        byok = getattr(usage, "is_byok", None) if usage is not None else None
        tool_text = self._tool_arguments(choice, raw) if output_mode == "strict_json" else None
        if tool_text:
            text = tool_text
        parsed = None if output_mode == "prose_only" else self._parse_jsonish(text)
        thinking = self._extract_reasoning(choice, raw)
        served_by = raw.get("provider") if isinstance(raw, dict) else None
        provider_constraint = None
        if self.fallback_for == "anthropic" and output_mode in {"strict_json", "loose_json"} and not thinking:
            provider_constraint = "anthropic_structured_output_suppresses_exposed_thinking"
        request_snapshot["fallback_for"] = self.fallback_for
        request_snapshot["served_by"] = served_by
        request_snapshot["openrouter_byok_required"] = self.cfg.get("openrouter_byok_required")
        request_snapshot["openrouter_byok_verification"] = self.cfg.get("openrouter_byok_verification")
        request_snapshot["temperature_omitted_reason"] = "claude_opus_4_7_sampling_unsupported" if omit_temperature else None
        request_snapshot["provider_constraint"] = provider_constraint
        request_snapshot["provider_pin_error"] = provider_pin_error
        return CallResult(
            text=text,
            parsed=parsed,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            latency_ms=elapsed_ms,
            raw=raw,
            error=f"ProviderResponseError: {provider_error}" if provider_error else None,
            thinking=thinking,
            schema_valid=is_valid_payload(parsed) if output_mode == "strict_json" else True,
            output_mode=output_mode,
            request=request_snapshot,
            response_id=getattr(resp, "id", None),
            actual_cost_usd=actual_cost,
            byok=bool(byok) if byok is not None else None,
        )
