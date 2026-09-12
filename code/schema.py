"""Canonical diagnostic output schema enforced on all three VLMs.

Two dialects:
  DIAGNOSIS_SCHEMA        — standard JSON Schema (OpenAI, Anthropic tool-use).
  GEMINI_DIAGNOSIS_SCHEMA — Gemini's response_schema dialect (uppercase types,
                            `nullable: true` instead of `["string", "null"]`).

Leakage discipline: field names are generic ("primary_diagnosis",
"differentials"). No disease names, no anatomy, no category hints in
descriptions or field names. The prompt itself (Asadi §7.4) is the only
place diagnostic context is introduced.
"""
from __future__ import annotations

DIAGNOSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "image_present": {
            "type": "boolean",
            "description": "true if an image was provided, false otherwise",
        },
        "can_diagnose": {
            "type": "boolean",
            "description": "true if a diagnosis can be given, false otherwise",
        },
        "primary_diagnosis": {
            "type": ["string", "null"],
            "description": "Most likely diagnosis as a short phrase, or null",
        },
        "differentials": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Other diagnoses being considered",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the primary diagnosis, between 0 and 1",
        },
        "key_findings": {
            "type": "string",
            "description": "Main findings observed",
        },
        "reasoning": {
            "type": "string",
            "description": "Diagnostic reasoning",
        },
    },
    "required": [
        "image_present",
        "can_diagnose",
        "primary_diagnosis",
        "differentials",
        "confidence",
        "key_findings",
        "reasoning",
    ],
    "additionalProperties": False,
}

SCHEMA_NAME = "diagnosis_record"

# Gemini rejects the ["string","null"] union; nullable: true is the idiom.
GEMINI_DIAGNOSIS_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "image_present": {
            "type": "BOOLEAN",
            "description": "true if an image was provided, false otherwise",
        },
        "can_diagnose": {
            "type": "BOOLEAN",
            "description": "true if a diagnosis can be given, false otherwise",
        },
        "primary_diagnosis": {
            "type": "STRING",
            "nullable": True,
            "description": "Most likely diagnosis as a short phrase, or null",
        },
        "differentials": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Other diagnoses being considered",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in the primary diagnosis, between 0 and 1",
        },
        "key_findings": {
            "type": "STRING",
            "description": "Main findings observed",
        },
        "reasoning": {
            "type": "STRING",
            "description": "Diagnostic reasoning",
        },
    },
    "required": [
        "image_present",
        "can_diagnose",
        "primary_diagnosis",
        "differentials",
        "confidence",
        "key_findings",
        "reasoning",
    ],
}

REQUIRED_KEYS = tuple(DIAGNOSIS_SCHEMA["required"])


def is_valid_payload(payload: dict | None) -> bool:
    """Structural check: all required keys present with expected primitive types."""
    if not isinstance(payload, dict):
        return False
    for k in REQUIRED_KEYS:
        if k not in payload:
            return False
    return (
        isinstance(payload["image_present"], bool)
        and isinstance(payload["can_diagnose"], bool)
        and (payload["primary_diagnosis"] is None or isinstance(payload["primary_diagnosis"], str))
        and isinstance(payload["differentials"], list)
        and isinstance(payload["confidence"], (int, float))
        and isinstance(payload["key_findings"], str)
        and isinstance(payload["reasoning"], str)
    )
