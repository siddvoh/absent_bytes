from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import re
from functools import lru_cache
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from PIL import Image, UnidentifiedImageError


RESPONSE_FIELDS = (
    "primary_diagnosis",
    "differentials",
    "clinical_assessment",
)
BINDING_FIELDS = (
    "task_id",
    "task_text_sha256",
    "contract_sha256",
    "contract_version",
    "active_field",
    "correlation_id",
)
MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
TASK_ID_PATTERN = re.compile(r"(?m)^evidence_task_id: ([a-z0-9-]+)$")
TASK_METADATA_LINE_PATTERN = re.compile(
    r"^evidence_(?:task_id|case_id): [a-z0-9-]+$"
)
HEX_SHA256 = "^[0-9a-f]{64}$"


class EnvelopeMaterializationError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_base64(value: str) -> tuple[bytes, bool]:
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, altchars=b"-_", validate=True), True
    except (ValueError, binascii.Error):
        return b"", False


@lru_cache(maxsize=None)
def _decoded_mime_type(data: bytes) -> str | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return MIME_BY_FORMAT.get(str(image.format or "").upper())
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def _evidence_rows(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    references: list[str] = []

    def add(encoded: str, declared_mime: str) -> None:
        data, base64_valid = _decode_base64(encoded)
        rows.append(
            {
                "base64_valid": base64_valid,
                "declared_mime": declared_mime.lower(),
                "decoded_mime": _decoded_mime_type(data) or "",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, Mapping):
            return
        consumed: set[str] = set()
        image_url = item.get("image_url")
        if isinstance(image_url, Mapping):
            consumed.add("image_url")
            url = str(image_url.get("url") or "")
            if url.startswith("data:") and ";base64," in url:
                header, encoded = url.split(",", 1)
                add(encoded, header[5:].split(";", 1)[0])
            elif url:
                references.append(url)
        for key in ("inlineData", "inline_data"):
            inline = item.get(key)
            if not isinstance(inline, Mapping):
                continue
            consumed.add(key)
            add(
                str(inline.get("data") or ""),
                str(inline.get("mimeType") or inline.get("mime_type") or ""),
            )
        for key in ("fileData", "file_data"):
            file_data = item.get(key)
            if not isinstance(file_data, Mapping):
                continue
            consumed.add(key)
            uri = str(file_data.get("fileUri") or file_data.get("file_uri") or "")
            if uri:
                references.append(uri)
        for key, child in item.items():
            if key not in consumed:
                walk(child)

    walk(value)
    return rows, references


def _text_fields(value: Any) -> list[str]:
    values: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, Mapping):
            return
        if isinstance(item.get("text"), str):
            values.append(item["text"])
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            return
        for key, child in item.items():
            if key != "text":
                walk(child)

    if isinstance(value, str):
        values.append(value)
    else:
        walk(value)
    return values


def _task_id(value: Any) -> str:
    matches = TASK_ID_PATTERN.findall("\n".join(_text_fields(value)))
    return matches[0] if len(matches) == 1 else ""


def _task_text(value: Any) -> str:
    lines: list[str] = []
    for text in _text_fields(value):
        lines.extend(
            line.rstrip()
            for line in text.splitlines()
            if not TASK_METADATA_LINE_PATTERN.fullmatch(line.strip())
        )
    return "\n".join(lines).strip()


def _response_schema(body: Mapping[str, Any], sdk_family: str) -> Mapping[str, Any] | None:
    if sdk_family == "openai":
        response_format = body.get("response_format")
        if not isinstance(response_format, Mapping):
            return None
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, Mapping):
            return None
        schema = json_schema.get("schema")
    else:
        generation = body.get("generationConfig")
        if not isinstance(generation, Mapping):
            return None
        candidates = [
            generation[key]
            for key in ("responseSchema", "responseJsonSchema")
            if key in generation
        ]
        schema = candidates[0] if len(candidates) == 1 else None
    return schema if isinstance(schema, Mapping) else None


def _response_projection(
    schema: Mapping[str, Any] | None, sdk_family: str
) -> dict[str, Any]:
    path = (
        "$.response_format.json_schema.schema"
        if sdk_family == "openai"
        else "$.generationConfig.responseJsonSchema"
    )
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    required = schema.get("required") if isinstance(schema, Mapping) else None
    forbidden = (
        sorted(set(properties).intersection(BINDING_FIELDS))
        if isinstance(properties, Mapping)
        else []
    )
    expected_properties = {
        "primary_diagnosis": {"type": ["string", "null"]},
        "differentials": {"type": "array", "items": {"type": "string"}},
        "clinical_assessment": {"type": "string"},
    }
    return {
        "additional_properties": (
            schema.get("additionalProperties") if isinstance(schema, Mapping) else None
        ),
        "forbidden_binding_fields": forbidden,
        "path": path,
        "present": schema is not None,
        "properties": sorted(properties) if isinstance(properties, Mapping) else [],
        "required": sorted(required) if isinstance(required, list) else [],
        "valid": bool(
            isinstance(schema, Mapping)
            and schema.get("type") == "object"
            and dict(properties) == expected_properties
            and isinstance(required, list)
            and set(required) == set(RESPONSE_FIELDS)
            and len(required) == len(RESPONSE_FIELDS)
            and schema.get("additionalProperties") is False
            and not forbidden
        ),
    }


def _response_binding(expected: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = expected.get("client_response_binding")
    if not isinstance(binding, Mapping) or set(binding) != set(BINDING_FIELDS):
        raise ValueError("client response binding has an invalid shape")
    if (
        not isinstance(binding["task_id"], str)
        or not binding["task_id"]
        or not isinstance(binding["task_text_sha256"], str)
        or not re.fullmatch(HEX_SHA256, binding["task_text_sha256"])
        or not isinstance(binding["contract_sha256"], str)
        or not re.fullmatch(HEX_SHA256, binding["contract_sha256"])
        or not isinstance(binding["contract_version"], int)
        or isinstance(binding["contract_version"], bool)
        or binding["contract_version"] < 1
        or binding["active_field"] not in RESPONSE_FIELDS
        or not isinstance(binding["correlation_id"], str)
        or not binding["correlation_id"]
    ):
        raise ValueError("client response binding has invalid values")
    expected_binding = expected.get("expected_response_binding")
    if not isinstance(expected_binding, Mapping) or dict(binding) != dict(expected_binding):
        raise ValueError("client response binding does not match its expectation")
    return dict(binding)


def materialize_request_envelope(
    body_bytes: bytes,
    sdk_family: str,
    *,
    parser_active_field: str,
) -> dict[str, Any]:
    try:
        body = json.loads(body_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvelopeMaterializationError("request body is not valid JSON") from exc
    if not isinstance(body, Mapping):
        raise EnvelopeMaterializationError("request body is not an object")
    if sdk_family == "openai":
        turns = body.get("messages")
        content_key = "content"
    elif sdk_family == "google_genai":
        turns = body.get("contents")
        content_key = "parts"
    else:
        raise EnvelopeMaterializationError(f"unsupported SDK family: {sdk_family}")
    if not isinstance(turns, list):
        raise EnvelopeMaterializationError("request turn collection is not a list")
    user_turns = [
        turn for turn in turns if isinstance(turn, Mapping) and turn.get("role") == "user"
    ]
    if not user_turns:
        raise EnvelopeMaterializationError("request has no user turn")
    active = user_turns[-1].get(content_key)
    active_images, active_references = _evidence_rows(active)
    global_images, global_references = _evidence_rows(body)
    task_text = _task_text(active)
    return {
        "active_images": active_images,
        "active_reference_count": len(active_references),
        "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "global_reference_count": len(global_references),
        "out_of_scope_image_count": len(global_images) - len(active_images),
        "parser_active_field": parser_active_field,
        "response_contract": _response_projection(
            _response_schema(body, sdk_family), sdk_family
        ),
        "sdk_family": sdk_family,
        "task_id": _task_id(active),
        "task_text_sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
    }


def request_envelope_schema(expected: Mapping[str, Any]) -> dict[str, Any]:
    binding = _response_binding(expected)
    expected_images = expected.get("images")
    if not isinstance(expected_images, list) or not expected_images:
        raise ValueError("expected ordered images are required")
    image_schemas = []
    for image in expected_images:
        if not isinstance(image, Mapping):
            raise ValueError("expected image is not an object")
        image_schemas.append(
            {
                "additionalProperties": False,
                "properties": {
                    "base64_valid": {"const": True},
                    "declared_mime": {"const": image["mime_type"]},
                    "decoded_mime": {"const": image["mime_type"]},
                    "sha256": {"const": image["sha256"]},
                },
                "required": [
                    "base64_valid",
                    "declared_mime",
                    "decoded_mime",
                    "sha256",
                ],
                "type": "object",
            }
        )
    response_schema: dict[str, Any]
    if expected.get("response_schema_required"):
        serialized_response_schema = expected.get("serialized_response_schema")
        if not isinstance(serialized_response_schema, Mapping):
            raise ValueError("serialized response schema is required")
        if serialized_response_schema.get("valid") is not True:
            raise ValueError("provider response schema is invalid")
        response_schema = {
            "const": dict(serialized_response_schema),
        }
    else:
        response_schema = {"type": "object"}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "active_images": {
                "items": False,
                "maxItems": len(image_schemas),
                "minItems": len(image_schemas),
                "prefixItems": image_schemas,
                "type": "array",
            },
            "active_reference_count": {"const": 0},
            "body_sha256": {"pattern": HEX_SHA256, "type": "string"},
            "global_reference_count": {"const": 0},
            "out_of_scope_image_count": {"const": 0},
            "parser_active_field": {"const": binding["active_field"]},
            "response_contract": response_schema,
            "sdk_family": {"enum": ["openai", "google_genai"]},
            "task_id": {"const": binding["task_id"]},
            "task_text_sha256": {"const": binding["task_text_sha256"]},
        },
        "required": [
            "active_images",
            "active_reference_count",
            "body_sha256",
            "global_reference_count",
            "out_of_scope_image_count",
            "parser_active_field",
            "response_contract",
            "sdk_family",
            "task_id",
            "task_text_sha256",
        ],
        "type": "object",
    }
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=None)
def _compiled_validator(expected_json: bytes) -> Draft202012Validator:
    expected = json.loads(expected_json)
    return Draft202012Validator(request_envelope_schema(expected))


def evaluate_request(
    body_bytes: bytes,
    sdk_family: str,
    expected: Mapping[str, Any],
    *,
    parser_active_field: str | None = None,
) -> tuple[bool, str]:
    try:
        binding = _response_binding(expected)
        envelope = materialize_request_envelope(
            body_bytes,
            sdk_family,
            parser_active_field=parser_active_field or str(binding["active_field"]),
        )
        validator = _compiled_validator(canonical_json_bytes(expected))
    except (EnvelopeMaterializationError, KeyError, TypeError, ValueError) as exc:
        return False, f"integrated_envelope_materialization:{exc}"
    errors = sorted(
        validator.iter_errors(envelope),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if not errors:
        return True, "draft202012_integrated_envelope_satisfied"
    first = errors[0]
    path = "/".join(str(part) for part in first.absolute_path) or "$"
    return False, f"draft202012:{path}:{first.validator}"


def schema_size_bytes(expected: Mapping[str, Any]) -> int:
    return len(canonical_json_bytes(request_envelope_schema(expected)))
