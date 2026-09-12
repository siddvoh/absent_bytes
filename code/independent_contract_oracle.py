from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_COMMITMENT_KEYS = frozenset(
    {
        "active_field",
        "contract_version",
        "ordered_image_mime_types",
        "ordered_image_sha256",
        "task_id",
        "task_text_sha256",
    }
)
_BOUND_SCHEMA_FIELDS = (
    "task_id",
    "task_text_sha256",
    "contract_sha256",
    "contract_version",
    "active_field",
    "correlation_id",
)
_SEMANTIC_RESPONSE_FIELDS = (
    "primary_diagnosis",
    "differentials",
    "clinical_assessment",
)
_SEMANTIC_RESPONSE_PROPERTIES = {
    "primary_diagnosis": {"type": ["string", "null"]},
    "differentials": {"type": "array", "items": {"type": "string"}},
    "clinical_assessment": {"type": "string"},
}
_MIMES = frozenset({"image/png", "image/jpeg", "image/webp"})
_DATA_URI = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]*={0,2})$")


@dataclass(frozen=True)
class OracleDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class _Expected:
    task_id: str
    task_text_sha256: str
    active_field: str
    contract_version: int
    mime_types: tuple[str, ...]
    image_sha256: tuple[str, ...]
    contract_sha256: str


def assess_serialized_request(
    body_bytes: bytes,
    sdk_family: str,
    expected_commitment: Mapping[str, Any],
    expected_task_text: str,
    expected_correlation_id: str,
    client_response_binding: Mapping[str, Any] | None = None,
) -> OracleDecision:
    expected = _expected(expected_commitment, expected_task_text, expected_correlation_id)
    if expected is None:
        return _deny("invalid_expected_commitment")
    if sdk_family not in {"openai", "google_genai"}:
        return _deny("unsupported_sdk_family")
    if not isinstance(body_bytes, bytes):
        return _deny("invalid_body")
    try:
        body = json.loads(body_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _deny("malformed_json")
    if not isinstance(body, Mapping):
        return _deny("request_is_not_an_object")
    turns = _turns(body, sdk_family)
    if turns is None:
        return _deny("malformed_turns")
    user_indexes = [index for index, turn in enumerate(turns) if turn.get("role") == "user"]
    if not user_indexes:
        return _deny("missing_last_user_turn")
    active_index = user_indexes[-1]
    parsed = [_parse_turn(turn, sdk_family) for turn in turns]
    if any(item is None for item in parsed):
        return _deny("malformed_turn")
    for index, item in enumerate(parsed):
        if index != active_index and item[1]:
            return _deny("image_outside_last_user_turn")
    texts, images = parsed[active_index]
    if len(texts) != 1:
        return _deny("ambiguous_task_text")
    if not _matches_task_text(texts[0], expected.task_id, expected_task_text):
        return _deny("task_text_or_id_mismatch")
    decoded: list[tuple[str, bytes]] = []
    for image in images:
        value = _decode_image(image, sdk_family)
        if value is None:
            return _deny("malformed_image")
        decoded.append(value)
    if len(decoded) > len(expected.image_sha256):
        return _deny("unexpected_extra_image")
    if len(decoded) < len(expected.image_sha256):
        return _deny("missing_image")
    observed_mimes = tuple(item[0] for item in decoded)
    for mime_type, data in decoded:
        if _signature_mime(data) != mime_type:
            return _deny("image_mime_mismatch")
    if observed_mimes != expected.mime_types:
        return _deny("image_mime_mismatch")
    observed_hashes = tuple(hashlib.sha256(item[1]).hexdigest() for item in decoded)
    if observed_hashes != expected.image_sha256:
        if sorted(observed_hashes) == sorted(expected.image_sha256):
            return _deny("image_order_mismatch")
        return _deny("image_digest_mismatch")
    schema = _response_schema(body, sdk_family)
    if schema is None:
        return _deny("missing_response_schema")
    if not _is_semantic_response_schema(schema):
        return _deny("invalid_semantic_response_schema")
    binding_status = _binding_status(
        client_response_binding, expected, expected_correlation_id
    )
    if binding_status is not None:
        return _deny(binding_status)
    return OracleDecision(True, "allowed")


def _deny(reason: str) -> OracleDecision:
    return OracleDecision(False, reason)


def _expected(
    commitment: Mapping[str, Any], task_text: str, correlation_id: str
) -> _Expected | None:
    if (
        not isinstance(commitment, Mapping)
        or set(commitment) != _COMMITMENT_KEYS
        or not isinstance(task_text, str)
        or not task_text
        or not isinstance(correlation_id, str)
        or not correlation_id
    ):
        return None
    task_id = commitment.get("task_id")
    task_digest = commitment.get("task_text_sha256")
    active_field = commitment.get("active_field")
    version = commitment.get("contract_version")
    mimes = commitment.get("ordered_image_mime_types")
    hashes = commitment.get("ordered_image_sha256")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(task_digest, str)
        or not _is_digest(task_digest)
        or task_digest != hashlib.sha256(task_text.encode("utf-8")).hexdigest()
        or not isinstance(active_field, str)
        or not active_field
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
        or not isinstance(mimes, (list, tuple))
        or not isinstance(hashes, (list, tuple))
        or len(mimes) != len(hashes)
        or any(not isinstance(item, str) or item not in _MIMES for item in mimes)
        or any(not isinstance(item, str) or not _is_digest(item) for item in hashes)
    ):
        return None
    canonical = {
        "active_field": active_field,
        "contract_version": version,
        "ordered_image_mime_types": list(mimes),
        "ordered_image_sha256": list(hashes),
        "task_id": task_id,
        "task_text_sha256": task_digest,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return _Expected(task_id, task_digest, active_field, version, tuple(mimes), tuple(hashes), digest)


def _is_digest(value: str) -> bool:
    return len(value) == 64 and not (set(value) - set("0123456789abcdef"))


def _turns(body: Mapping[str, Any], sdk_family: str) -> list[Mapping[str, Any]] | None:
    raw = body.get("messages") if sdk_family == "openai" else body.get("contents")
    if sdk_family == "google_genai" and isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        return None
    return list(raw)


def _parse_turn(
    turn: Mapping[str, Any], sdk_family: str
) -> tuple[list[str], list[Mapping[str, Any]]] | None:
    value = turn.get("content") if sdk_family == "openai" else turn.get("parts")
    if isinstance(value, str) and sdk_family == "openai":
        return [value], []
    if not isinstance(value, list):
        return None
    texts: list[str] = []
    images: list[Mapping[str, Any]] = []
    for part in value:
        if not isinstance(part, Mapping):
            return None
        if sdk_family == "openai":
            if part.get("type") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    return None
                texts.append(text)
            if part.get("type") == "image_url" or "image_url" in part:
                images.append(part)
        else:
            if "text" in part:
                text = part.get("text")
                if not isinstance(text, str):
                    return None
                texts.append(text)
            if "inlineData" in part or "inline_data" in part:
                images.append(part)
            if "fileData" in part or "file_data" in part:
                images.append(part)
    return texts, images


def _matches_task_text(text: str, task_id: str, expected_text: str) -> bool:
    lines = text.splitlines()
    if not lines or lines[0] != f"evidence_task_id: {task_id}":
        return False
    remainder = lines[1:]
    if remainder and remainder[0].startswith("evidence_case_id: "):
        if not remainder[0][len("evidence_case_id: ") :]:
            return False
        remainder = remainder[1:]
    return "\n".join(remainder) == expected_text


def _decode_image(part: Mapping[str, Any], sdk_family: str) -> tuple[str, bytes] | None:
    if sdk_family == "openai":
        image_url = part.get("image_url")
        if not isinstance(image_url, Mapping) or not isinstance(image_url.get("url"), str):
            return None
        match = _DATA_URI.fullmatch(image_url["url"])
        if match is None:
            return None
        mime_type, encoded = match.groups()
    else:
        keys = [key for key in ("inlineData", "inline_data") if key in part]
        if len(keys) != 1 or not isinstance(part[keys[0]], Mapping):
            return None
        inline = part[keys[0]]
        mime_key = "mimeType" if keys[0] == "inlineData" else "mime_type"
        mime_type, encoded = inline.get(mime_key), inline.get("data")
        if not isinstance(mime_type, str) or not isinstance(encoded, str):
            return None
    if mime_type not in _MIMES:
        return None
    try:
        altchars = b"-_" if sdk_family == "google_genai" else None
        return mime_type, base64.b64decode(
            encoded, altchars=altchars, validate=True
        )
    except (ValueError, binascii.Error):
        return None


def _signature_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _response_schema(body: Mapping[str, Any], sdk_family: str) -> Mapping[str, Any] | None:
    if sdk_family == "openai":
        response_format = body.get("response_format")
        if not isinstance(response_format, Mapping):
            return None
        json_schema = response_format.get("json_schema")
        return json_schema.get("schema") if isinstance(json_schema, Mapping) else None
    generation = body.get("generationConfig")
    if not isinstance(generation, Mapping):
        return None
    schemas = [generation[key] for key in ("responseSchema", "responseJsonSchema") if key in generation]
    return schemas[0] if len(schemas) == 1 and isinstance(schemas[0], Mapping) else None


def _is_semantic_response_schema(schema: Mapping[str, Any]) -> bool:
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return False
    return bool(
        schema.get("type") == "object"
        and dict(properties) == _SEMANTIC_RESPONSE_PROPERTIES
        and set(required) == set(_SEMANTIC_RESPONSE_FIELDS)
        and len(required) == len(_SEMANTIC_RESPONSE_FIELDS)
        and schema.get("additionalProperties") is False
    )


def _binding_status(
    binding: Mapping[str, Any] | None, expected: _Expected, correlation_id: str
) -> str | None:
    if not isinstance(binding, Mapping) or set(binding) != set(_BOUND_SCHEMA_FIELDS):
        return "invalid_client_response_binding"
    values: dict[str, Any] = {
        "task_id": expected.task_id,
        "task_text_sha256": expected.task_text_sha256,
        "contract_sha256": expected.contract_sha256,
        "contract_version": expected.contract_version,
        "active_field": expected.active_field,
        "correlation_id": correlation_id,
    }
    if (
        not isinstance(binding["task_id"], str)
        or not binding["task_id"]
        or not isinstance(binding["task_text_sha256"], str)
        or not _is_digest(binding["task_text_sha256"])
        or not isinstance(binding["contract_sha256"], str)
        or not _is_digest(binding["contract_sha256"])
        or not isinstance(binding["contract_version"], int)
        or isinstance(binding["contract_version"], bool)
        or binding["contract_version"] < 1
        or not isinstance(binding["active_field"], str)
        or not binding["active_field"]
        or not isinstance(binding["correlation_id"], str)
        or not binding["correlation_id"]
    ):
        return "invalid_client_response_binding"
    if dict(binding) != values:
        return "client_response_binding_mismatch"
    return None
