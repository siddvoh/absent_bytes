#!/usr/bin/env python3
"""Exercise evidence-to-task bindings through current SDK request serialization.

This is a deterministic, no-network compatibility matrix. It inspects the exact
JSON bytes given to httpx.MockTransport by the current OpenAI and Google Gen AI
SDKs; it does not make claims about historical requests or production behavior.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier
from typing import Any, Callable, Mapping

import httpx
import openai
import PIL
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import types
from openai import OpenAI
from PIL import Image, UnidentifiedImageError


OPENAI_ROUTES = {
    "openrouter_anthropic_sdk": {
        "model": "anthropic/claude-opus-4.7",
        "provider": {
            "order": ["anthropic"],
            "only": ["anthropic"],
            "allow_fallbacks": False,
        },
    },
    "openrouter_openai_sdk": {
        "model": "openai/gpt-5.4",
        "provider": {"only": ["openai"], "allow_fallbacks": False},
    },
    "openrouter_qwen_sdk": {
        "model": "qwen/qwen3-vl-32b-instruct",
        "provider": {"only": ["alibaba"], "allow_fallbacks": False},
    },
    "openrouter_llama_sdk": {
        "model": "meta-llama/llama-4-maverick",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
    },
}
VERTEX_ROUTE = "google_vertex_sdk"
VERTEX_MODEL = "gemini-3.1-pro-preview"
DATA_URI_PREFIX = "data:"
TASK_ID_PATTERN = re.compile(r"(?m)^evidence_task_id: ([a-z0-9-]+)$")
CASE_ID_PATTERN = re.compile(r"(?m)^evidence_case_id: ([a-z0-9-]+)$")
ACTIVE_FIELD_PATTERN = re.compile(r"(?m)^evidence_active_field: ([a-z0-9_]+)$")
CONTRACT_VERSION_PATTERN = re.compile(r"(?m)^evidence_contract_version: ([0-9]+)$")
TASK_METADATA_LINE_PATTERN = re.compile(
    r"^evidence_(?:task_id|case_id): [a-z0-9-]+$"
)
MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
REMOTE_FIXTURE_URI = "https://evidence.example.test/case-remote.png"
UNRESOLVED_URI = "gs://unresolved-evidence/missing-image.png"
DEFAULT_TASK_TEXT = "Describe the bound image evidence."


@dataclass(frozen=True)
class EvidencePart:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    images: tuple[EvidencePart, ...]
    task_text: str = DEFAULT_TASK_TEXT
    active_field: str = "primary_diagnosis"
    contract_version: int = 1


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    serialized_task_id: str
    images: tuple[EvidencePart, ...]
    expected_decision: str
    expected_reason: str
    input_flow: str
    materialized_source_uri: str | None = None
    unresolved_uri: str | None = None
    expected_task_id: str | None = None
    message_layout: str = "active_user_turn"
    serialized_task_text: str = DEFAULT_TASK_TEXT

    @property
    def contract_task_id(self) -> str:
        return self.expected_task_id or self.serialized_task_id


@dataclass(frozen=True)
class DispatchExpectation:
    case_id: str
    contract: TaskContract
    correlation_id: str | None = None
    client_response_binding: Mapping[str, Any] | None = None


PNG_IMAGE = EvidencePart(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    ),
    "image/png",
)
JPEG_IMAGE = EvidencePart(
    base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgIC"
        "AgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBD"
        "AQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoK"
        "CgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAACAAIDAREAAhEBAxEB/8QAHwAAAQUB"
        "AQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQID"
        "AAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0"
        "NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKT"
        "lJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl"
        "5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL"
        "/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHB"
        "CSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpj"
        "ZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3"
        "uLm6wsLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oADAMB"
        "AAIRAxEAPwD8e6+sPlz/2Q=="
    ),
    "image/jpeg",
)
WEBP_IMAGE = EvidencePart(
    base64.b64decode("UklGRh4AAABXRUJQVlA4TBEAAAAvAUAAAAfQqhLVrP+BiOh/AAA="),
    "image/webp",
)
MULTI_PNG_A = EvidencePart(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGMs75jJ"
        "wMDAxMDAwMDAAAATKAGccHUYYAAAAABJRU5ErkJggg=="
    ),
    "image/png",
)
MULTI_PNG_B = EvidencePart(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGNctfsM"
        "AwMDEwMDAwMDAwAaVAI1BBOjVgAAAABJRU5ErkJggg=="
    ),
    "image/png",
)
REMOTE_PNG = EvidencePart(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGNUdEpm"
        "YGBgYmBgYGBgAAAJMADKduWbEAAAAABJRU5ErkJggg=="
    ),
    "image/png",
)
UNBOUND_PNG = EvidencePart(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGO8l2/O"
        "wMDAxMDAwMDAAAATAQGICQrpyQAAAABJRU5ErkJggg=="
    ),
    "image/png",
)
SOURCE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGM8I6j0"
    "n4GBgYGJAQoAHsICAs0lRdgAAAAASUVORK5CYII="
)
EXPECTED_TRANSFORMED_JPEG_SHA256 = (
    "6b5c42d42845a9e58026b2d3c947b74a38548576f58b95038961cb3a614f94b2"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def transform_for_submission(source: bytes) -> bytes:
    """Normalize a fixture into the exact JPEG bytes bound by task-transformed."""
    with Image.open(io.BytesIO(source)) as image:
        output = io.BytesIO()
        image.convert("RGB").save(
            output,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    return output.getvalue()


TRANSFORMED_JPEG = EvidencePart(transform_for_submission(SOURCE_PNG), "image/jpeg")
if sha256_bytes(TRANSFORMED_JPEG.data) != EXPECTED_TRANSFORMED_JPEG_SHA256:
    raise RuntimeError("the deterministic transformed fixture digest changed")

TASKS = {
    "task-png": TaskContract("task-png", (PNG_IMAGE,)),
    "task-jpeg": TaskContract("task-jpeg", (JPEG_IMAGE,)),
    "task-webp": TaskContract("task-webp", (WEBP_IMAGE,)),
    "task-multi": TaskContract("task-multi", (MULTI_PNG_A, MULTI_PNG_B)),
    "task-remote": TaskContract("task-remote", (REMOTE_PNG,)),
    "task-transformed": TaskContract("task-transformed", (TRANSFORMED_JPEG,)),
}
OFFLINE_REMOTE_FIXTURES = {REMOTE_FIXTURE_URI: REMOTE_PNG}


def materialize_remote_reference(uri: str) -> EvidencePart:
    try:
        return OFFLINE_REMOTE_FIXTURES[uri]
    except KeyError as error:
        raise ValueError(f"no offline fixture for remote reference: {uri}") from error


def matrix_cases() -> tuple[MatrixCase, ...]:
    transformed = transform_for_submission(SOURCE_PNG)
    if transformed != TRANSFORMED_JPEG.data:
        raise ValueError("the deterministic transformed bytes no longer match the bound fixture")
    return (
        MatrixCase(
            "png-inline",
            "task-png",
            (PNG_IMAGE,),
            "forwarded",
            "evidence_binding_satisfied",
            "inline_bytes",
        ),
        MatrixCase(
            "jpeg-inline",
            "task-jpeg",
            (JPEG_IMAGE,),
            "forwarded",
            "evidence_binding_satisfied",
            "inline_bytes",
        ),
        MatrixCase(
            "webp-inline",
            "task-webp",
            (WEBP_IMAGE,),
            "forwarded",
            "evidence_binding_satisfied",
            "inline_bytes",
        ),
        MatrixCase(
            "multi-image-ordered",
            "task-multi",
            (MULTI_PNG_A, MULTI_PNG_B),
            "forwarded",
            "evidence_binding_satisfied",
            "ordered_multi_image",
        ),
        MatrixCase(
            "materialized-remote-reference",
            "task-remote",
            (materialize_remote_reference(REMOTE_FIXTURE_URI),),
            "forwarded",
            "evidence_binding_satisfied",
            "materialized_remote_reference",
            materialized_source_uri=REMOTE_FIXTURE_URI,
        ),
        MatrixCase(
            "expected-transformed-bytes",
            "task-transformed",
            (EvidencePart(transformed, "image/jpeg"),),
            "forwarded",
            "evidence_binding_satisfied",
            "deterministic_transformation",
        ),
        MatrixCase(
            "missing-image",
            "task-png",
            (),
            "blocked",
            "missing_image",
            "no_image",
        ),
        MatrixCase(
            "corrupt-image",
            "task-png",
            (EvidencePart(b"corrupt-image-bytes", "image/png"),),
            "blocked",
            "corrupt_image",
            "corrupt_inline_bytes",
        ),
        MatrixCase(
            "mime-mismatch",
            "task-png",
            (EvidencePart(PNG_IMAGE.data, "image/jpeg"),),
            "blocked",
            "mime_mismatch",
            "inline_bytes",
        ),
        MatrixCase(
            "digest-mismatch",
            "task-png",
            (UNBOUND_PNG,),
            "blocked",
            "digest_mismatch",
            "inline_bytes",
        ),
        MatrixCase(
            "wrong-task-binding",
            "task-png",
            (REMOTE_PNG,),
            "blocked",
            "wrong_task_image_binding",
            "inline_bytes",
        ),
        MatrixCase(
            "unresolved-uri",
            "task-png",
            (),
            "blocked",
            "unresolved_uri",
            "unresolved_reference",
            unresolved_uri=UNRESOLVED_URI,
        ),
        MatrixCase(
            "unexpected-extra-image",
            "task-png",
            (PNG_IMAGE, JPEG_IMAGE),
            "blocked",
            "unexpected_extra_image",
            "inline_bytes",
        ),
        MatrixCase(
            "wrong-image-order",
            "task-multi",
            (MULTI_PNG_B, MULTI_PNG_A),
            "blocked",
            "wrong_image_order",
            "ordered_multi_image",
        ),
        MatrixCase(
            "wrong-role-task-binding",
            "task-png",
            (PNG_IMAGE,),
            "blocked",
            "task_scope_mismatch",
            "task_marker_outside_active_user_turn",
            message_layout="task_marker_in_system_or_model_role",
        ),
        MatrixCase(
            "wrong-turn-image-binding",
            "task-png",
            (PNG_IMAGE,),
            "blocked",
            "evidence_scope_mismatch",
            "image_outside_active_user_turn",
            message_layout="image_in_prior_user_turn",
        ),
        MatrixCase(
            "complete-task-body-substitution",
            "task-remote",
            (REMOTE_PNG,),
            "blocked",
            "task_id_mismatch",
            "complete_task_body_substitution",
            expected_task_id="task-png",
        ),
    )


def decode_base64(value: str) -> tuple[bytes, bool]:
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, altchars=b"-_", validate=True), True
    except (ValueError, base64.binascii.Error):
        return b"", False


def inspect_inline_evidence(
    value: Any, root_path: str = "$"
) -> tuple[list[dict[str, Any]], list[str]]:
    images: list[dict[str, Any]] = []
    references: list[str] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if not isinstance(item, dict):
            return

        consumed: set[str] = set()
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            consumed.add("image_url")
            url = str(image_url.get("url") or "")
            if url.startswith(DATA_URI_PREFIX) and ";base64," in url:
                header, encoded = url.split(",", 1)
                data, valid = decode_base64(encoded)
                images.append(
                    {
                        "base64_valid": valid,
                        "data": data,
                        "declared_mime_type": header[5:].split(";", 1)[0].lower(),
                        "source": f"{path}.image_url.url",
                    }
                )
            elif url:
                references.append(url)

        for key in ("inlineData", "inline_data"):
            inline_data = item.get(key)
            if not isinstance(inline_data, dict):
                continue
            consumed.add(key)
            data, valid = decode_base64(str(inline_data.get("data") or ""))
            images.append(
                {
                    "base64_valid": valid,
                    "data": data,
                    "declared_mime_type": str(
                        inline_data.get("mimeType") or inline_data.get("mime_type") or ""
                    ).lower(),
                    "source": f"{path}.{key}.data",
                }
            )

        for key in ("fileData", "file_data"):
            file_data = item.get(key)
            if not isinstance(file_data, dict):
                continue
            consumed.add(key)
            uri = str(file_data.get("fileUri") or file_data.get("file_uri") or "")
            if uri:
                references.append(uri)

        for key, child in item.items():
            if key not in consumed:
                walk(child, f"{path}.{key}")

    walk(value, root_path)
    return images, references


def decoded_mime_type(data: bytes) -> str | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return MIME_BY_FORMAT.get(str(image.format or "").upper())
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def text_fields(value: Any) -> list[str]:
    texts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        for key, child in item.items():
            if key == "text" and isinstance(child, str):
                texts.append(child)
            elif key == "content" and isinstance(child, str):
                texts.append(child)
            else:
                walk(child)

    walk(value)
    return texts


def request_metadata(value: Any) -> tuple[str, str]:
    text = "\n".join(text_fields(value))
    task_ids = TASK_ID_PATTERN.findall(text)
    case_ids = CASE_ID_PATTERN.findall(text)
    if len(task_ids) != 1 or len(case_ids) != 1:
        return "", ""
    return task_ids[0], case_ids[0]


def canonical_task_text(value: Any) -> str:
    lines: list[str] = []
    for text in text_fields(value):
        lines.extend(
            line.rstrip()
            for line in text.splitlines()
            if not TASK_METADATA_LINE_PATTERN.fullmatch(line.strip())
        )
    return "\n".join(lines).strip()


def request_contract_metadata(value: Any) -> tuple[str | None, int | None, bool]:
    text = "\n".join(text_fields(value))
    active_fields = ACTIVE_FIELD_PATTERN.findall(text)
    versions = CONTRACT_VERSION_PATTERN.findall(text)
    if not active_fields and not versions:
        return None, None, True
    if len(active_fields) != 1 or len(versions) != 1:
        return None, None, False
    return active_fields[0], int(versions[0]), True


def active_user_scope(body: dict[str, Any], sdk_family: str) -> dict[str, Any]:
    if sdk_family == "openai":
        collection_key = "messages"
        content_key = "content"
    elif sdk_family == "google_genai":
        collection_key = "contents"
        content_key = "parts"
    else:
        raise ValueError(f"unsupported SDK family: {sdk_family}")

    messages = body.get(collection_key)
    if not isinstance(messages, list):
        messages = []
    user_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_indexes:
        return {
            "active_message_index": None,
            "active_message_role": None,
            "active_scope_path": f"$.{collection_key}",
            "images": [],
            "references": [],
            "scoped_case_id": "",
            "scoped_task_id": "",
            "scoped_task_text": "",
        }

    index = user_indexes[-1]
    message = messages[index]
    scope = message.get(content_key)
    path = f"$.{collection_key}[{index}].{content_key}"
    images, references = inspect_inline_evidence(scope, path)
    scoped_task_id, scoped_case_id = request_metadata(scope)
    return {
        "active_message_index": index,
        "active_message_role": "user",
        "active_scope_path": path,
        "images": images,
        "references": references,
        "scoped_case_id": scoped_case_id,
        "scoped_task_id": scoped_task_id,
        "scoped_task_text": canonical_task_text(scope),
    }


def task_digest_sequences(
    task_registry: Mapping[str, TaskContract] | None = None,
) -> dict[tuple[str, ...], set[str]]:
    registry = TASKS if task_registry is None else task_registry
    sequences: dict[tuple[str, ...], set[str]] = {}
    for task in registry.values():
        digests = tuple(sha256_bytes(image.data) for image in task.images)
        sequences.setdefault(digests, set()).add(task.task_id)
    return sequences


def evidence_decision(
    expectation: DispatchExpectation,
    scoped_task_id: str,
    scoped_task_text: str,
    global_task_id: str,
    images: list[dict[str, Any]],
    references: list[str],
    out_of_scope_image_count: int,
    task_registry: Mapping[str, TaskContract] | None = None,
) -> tuple[str, str]:
    registry = TASKS if task_registry is None else task_registry
    if scoped_task_id != expectation.contract.task_id:
        if global_task_id == expectation.contract.task_id:
            return "blocked", "task_scope_mismatch"
        return "blocked", "task_id_mismatch"
    if scoped_task_text != expectation.contract.task_text:
        return "blocked", "task_text_mismatch"
    if references:
        return "blocked", "unresolved_uri"
    contract = expectation.contract
    if out_of_scope_image_count:
        return "blocked", "evidence_scope_mismatch"
    if not images:
        return "blocked", "missing_image"
    if len(images) > len(contract.images):
        return "blocked", "unexpected_extra_image"
    if len(images) < len(contract.images):
        return "blocked", "missing_image"

    for image in images:
        image["decoded_mime_type"] = decoded_mime_type(image["data"])
        if not image["base64_valid"] or image["decoded_mime_type"] is None:
            return "blocked", "corrupt_image"
        if image["declared_mime_type"] != image["decoded_mime_type"]:
            return "blocked", "mime_mismatch"

    observed_digests = tuple(sha256_bytes(image["data"]) for image in images)
    expected_digests = tuple(sha256_bytes(image.data) for image in contract.images)
    if (
        observed_digests != expected_digests
        and len(observed_digests) > 1
        and sorted(observed_digests) == sorted(expected_digests)
    ):
        return "blocked", "wrong_image_order"
    observed_mime_types = tuple(image["decoded_mime_type"] for image in images)
    expected_mime_types = tuple(image.mime_type for image in contract.images)
    if observed_mime_types != expected_mime_types:
        return "blocked", "mime_mismatch"
    if observed_digests == expected_digests:
        return "forwarded", "evidence_binding_satisfied"
    owners = task_digest_sequences(registry).get(observed_digests, set())
    if owners and contract.task_id not in owners:
        return "blocked", "wrong_task_image_binding"
    return "blocked", "digest_mismatch"


def mime_valid(image: dict[str, Any]) -> bool:
    decoded = image.get("decoded_mime_type") or decoded_mime_type(image["data"])
    image["decoded_mime_type"] = decoded
    return bool(
        image["base64_valid"]
        and decoded is not None
        and image["declared_mime_type"] == decoded
    )


def mime_aware_presence_decision(
    images: list[dict[str, Any]], references: list[str]
) -> tuple[str, str]:
    if references or not images:
        return "blocked", "no_materialized_image"
    if not all(mime_valid(image) for image in images):
        return "blocked", "nondecodable_or_mime_mismatch"
    return "forwarded", "decoded_presence_satisfied"


def body_only_consistency_decision(
    task_id: str,
    images: list[dict[str, Any]],
    references: list[str],
    task_registry: Mapping[str, TaskContract] | None = None,
) -> tuple[str, str]:
    registry = TASKS if task_registry is None else task_registry
    if references:
        return "blocked", "unresolved_reference"
    contract = registry.get(task_id)
    if contract is None:
        return "blocked", "body_task_missing"
    if len(images) != len(contract.images):
        return "blocked", "body_image_count_mismatch"
    if not all(mime_valid(image) for image in images):
        return "blocked", "nondecodable_or_mime_mismatch"
    observed = tuple(
        (image["declared_mime_type"], sha256_bytes(image["data"]))
        for image in images
    )
    expected = tuple(
        (image.mime_type, sha256_bytes(image.data)) for image in contract.images
    )
    if observed != expected:
        return "blocked", "body_task_content_mismatch"
    return "forwarded", "body_self_consistency_satisfied"


def openai_response(content: dict[str, Any], body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "offline-evidence-binding-response",
            "object": "chat.completion",
            "created": 0,
            "model": str(body.get("model") or "unknown"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, sort_keys=True),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )


def vertex_response(content: dict[str, Any], _body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": json.dumps(content, sort_keys=True)}],
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    )


@dataclass
class DownstreamSpy:
    route: str
    response_formatter: Callable[[dict[str, Any], dict[str, Any]], httpx.Response]
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        request: httpx.Request,
        content: dict[str, Any],
        body: dict[str, Any],
        case_id: str,
    ) -> httpx.Response:
        self.invocations.append(
            {
                "body_sha256": sha256_bytes(request.content),
                "case_id": case_id,
                "route": self.route,
            }
        )
        return self.response_formatter(content, body)


@dataclass
class GuardedTransport:
    route: str
    sdk_family: str
    expectation: DispatchExpectation
    blocked_response_formatter: Callable[
        [dict[str, Any], dict[str, Any]], httpx.Response
    ]
    downstream: DownstreamSpy
    task_registry: Mapping[str, TaskContract] = field(default_factory=lambda: TASKS)
    response_schema_inspector: Callable[
        [Mapping[str, Any], str], dict[str, Any]
    ] | None = None
    captured: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.captured:
            raise RuntimeError("one-shot guarded transport received more than one request")
        expectation = self.expectation
        body = json.loads(request.content)
        global_images, global_references = inspect_inline_evidence(body)
        global_task_id, global_case_id = request_metadata(body)
        global_task_text = canonical_task_text(body)
        scope = active_user_scope(body, self.sdk_family)
        images = scope["images"]
        references = scope["references"]
        scoped_task_id = scope["scoped_task_id"]
        scoped_task_text = scope["scoped_task_text"]
        scoped_case_id = scope["scoped_case_id"]
        out_of_scope_image_count = len(global_images) - len(images)
        decision, reason = evidence_decision(
            expectation,
            scoped_task_id,
            scoped_task_text,
            global_task_id,
            images,
            references,
            out_of_scope_image_count,
            self.task_registry,
        )
        mime_presence_decision, mime_presence_reason = mime_aware_presence_decision(
            global_images,
            global_references,
        )
        body_consistency_decision, body_consistency_reason = (
            body_only_consistency_decision(
                global_task_id,
                global_images,
                global_references,
                self.task_registry,
            )
        )
        contract = expectation.contract
        serialized_response_schema = None
        client_response_binding = None
        if self.response_schema_inspector is not None:
            serialized_response_schema = self.response_schema_inspector(
                body, self.sdk_family
            )
            client_response_binding = expectation.client_response_binding
            if not serialized_response_schema["present"]:
                decision, reason = "blocked", "response_schema_missing"
            elif not serialized_response_schema["valid"]:
                decision, reason = "blocked", "response_schema_invalid"
            elif client_response_binding is None:
                decision, reason = "blocked", "client_response_binding_missing"
            elif not isinstance(client_response_binding, Mapping):
                decision, reason = "blocked", "client_response_binding_invalid"
            else:
                expected_binding = {
                    "task_id": contract.task_id,
                    "task_text_sha256": sha256_bytes(
                        contract.task_text.encode("utf-8")
                    ),
                    "contract_sha256": getattr(contract, "sha256", None),
                    "contract_version": contract.contract_version,
                    "active_field": contract.active_field,
                    "correlation_id": expectation.correlation_id,
                }
                client_response_binding = dict(client_response_binding)
                if set(client_response_binding) != set(expected_binding):
                    decision, reason = "blocked", "client_response_binding_invalid"
                elif client_response_binding != expected_binding:
                    decision, reason = "blocked", "client_response_binding_mismatch"
        else:
            active_field, contract_version, contract_metadata_valid = (
                request_contract_metadata(body)
            )
            if not contract_metadata_valid:
                decision, reason = "blocked", "contract_metadata_invalid"
            elif active_field is not None and active_field != contract.active_field:
                decision, reason = "blocked", "active_field_mismatch"
            elif (
                contract_version is not None
                and contract_version != contract.contract_version
            ):
                decision, reason = "blocked", "contract_version_mismatch"
        response_content = {"decision": decision, "reason": reason}
        invocation_count_before = len(self.downstream.invocations)
        if decision == "forwarded":
            response = self.downstream(
                request,
                response_content,
                body,
                expectation.case_id,
            )
        else:
            response = self.blocked_response_formatter(response_content, body)
        downstream_invocations = len(self.downstream.invocations) - invocation_count_before
        expected_invocations = 1 if decision == "forwarded" else 0
        if downstream_invocations != expected_invocations:
            raise RuntimeError(
                f"{self.route}/{expectation.case_id} produced {downstream_invocations} "
                f"downstream invocations"
            )
        captured = {
                "body_sha256": sha256_bytes(request.content),
                "case_id": expectation.case_id,
                "comparative_gates": {
                    "body_only_task_content_consistency": {
                        "decision": body_consistency_decision,
                        "reason": body_consistency_reason,
                    },
                    "mime_aware_decoded_presence": {
                        "decision": mime_presence_decision,
                        "reason": mime_presence_reason,
                    },
                },
                "decision": decision,
                "downstream_invocations": downstream_invocations,
                "active_message_index": scope["active_message_index"],
                "active_message_role": scope["active_message_role"],
                "active_scope_path": scope["active_scope_path"],
                "expected_case_id": expectation.case_id,
                "expected_image_sha256": [
                    sha256_bytes(image.data) for image in contract.images
                ],
                "expected_task_id": contract.task_id,
                "expected_task_text_sha256": sha256_bytes(
                    contract.task_text.encode("utf-8")
                ),
                "image_base64_valid": [image["base64_valid"] for image in images],
                "image_byte_count": sum(len(image["data"]) for image in images),
                "image_decoded_mime_types": [
                    image.get("decoded_mime_type") or decoded_mime_type(image["data"])
                    for image in images
                ],
                "image_declared_mime_types": [
                    image["declared_mime_type"] for image in images
                ],
                "image_sha256": [sha256_bytes(image["data"]) for image in images],
                "reason": reason,
                "reference_count": len(references),
                "global_case_id": global_case_id,
                "global_image_count": len(global_images),
                "global_reference_count": len(global_references),
                "global_task_id": global_task_id,
                "global_task_text_sha256": sha256_bytes(
                    global_task_text.encode("utf-8")
                ),
                "out_of_scope_image_count": out_of_scope_image_count,
                "route": self.route,
                "scoped_case_id": scoped_case_id,
                "scoped_image_count": len(images),
                "scoped_task_id": scoped_task_id,
                "scoped_task_text_sha256": sha256_bytes(
                    scoped_task_text.encode("utf-8")
                ),
                "serialized_task_id": global_task_id,
                "url": str(request.url),
            }
        if serialized_response_schema is not None:
            captured["serialized_response_schema"] = serialized_response_schema
            captured["client_response_binding"] = client_response_binding
        self.captured.append(captured)
        return response


def prompt_for(case: MatrixCase) -> str:
    return "\n".join(
        (
            f"evidence_task_id: {case.serialized_task_id}",
            f"evidence_case_id: {case.case_id}",
            case.serialized_task_text,
        )
    )


def openai_image_content(case: MatrixCase) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in case.images:
        data_uri = "data:" + image.mime_type + ";base64," + base64.b64encode(
            image.data
        ).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    if case.unresolved_uri:
        content.append(
            {"type": "image_url", "image_url": {"url": case.unresolved_uri}}
        )
    return content


def openai_messages(case: MatrixCase) -> list[dict[str, Any]]:
    text = {"type": "text", "text": prompt_for(case)}
    images = openai_image_content(case)
    if case.message_layout == "task_marker_in_system_or_model_role":
        return [
            {"role": "system", "content": prompt_for(case)},
            {"role": "user", "content": images},
        ]
    if case.message_layout == "image_in_prior_user_turn":
        return [
            {"role": "user", "content": images},
            {"role": "user", "content": [text]},
        ]
    return [{"role": "user", "content": [text, *images]}]


def vertex_image_parts(case: MatrixCase) -> list[types.Part]:
    content = [
        types.Part.from_bytes(data=image.data, mime_type=image.mime_type)
        for image in case.images
    ]
    if case.unresolved_uri:
        content.append(
            types.Part.from_uri(file_uri=case.unresolved_uri, mime_type="image/png")
        )
    return content


def vertex_contents(case: MatrixCase) -> list[types.Content]:
    text = types.Part.from_text(text=prompt_for(case))
    images = vertex_image_parts(case)
    if case.message_layout == "task_marker_in_system_or_model_role":
        return [
            types.Content(role="model", parts=[text]),
            types.Content(role="user", parts=images),
        ]
    if case.message_layout == "image_in_prior_user_turn":
        return [
            types.Content(role="user", parts=images),
            types.Content(role="user", parts=[text]),
        ]
    return [types.Content(role="user", parts=[text, *images])]


def verify_response(
    transport: GuardedTransport, case: MatrixCase, response: dict[str, Any]
) -> None:
    record = transport.captured[-1]
    expected = {"decision": case.expected_decision, "reason": case.expected_reason}
    if (
        record["case_id"] != case.case_id
        or record["expected_case_id"] != case.case_id
        or record["serialized_task_id"] != case.serialized_task_id
        or record["expected_task_id"] != case.contract_task_id
    ):
        raise ValueError(f"{transport.route}/{case.case_id} lost request binding metadata")
    if response != expected or {
        "decision": record["decision"],
        "reason": record["reason"],
    } != expected:
        raise ValueError(f"{transport.route}/{case.case_id} produced {response}")
    expected_invocations = 1 if case.expected_decision == "forwarded" else 0
    if record["downstream_invocations"] != expected_invocations:
        raise ValueError(
            f"{transport.route}/{case.case_id} reached the downstream spy "
            f"{record['downstream_invocations']} times"
        )


def summarize_gate(
    decisions: list[dict[str, Any]], gate_name: str
) -> dict[str, Any]:
    def decision(row: dict[str, Any]) -> str:
        if gate_name == "caller_owned_evidence_contract":
            return str(row["decision"])
        return str(row["comparative_gates"][gate_name]["decision"])

    blocked = [row["case_id"] for row in decisions if decision(row) == "blocked"]
    forwarded = [
        row["case_id"] for row in decisions if decision(row) == "forwarded"
    ]
    return {
        "blocked": len(blocked),
        "blocked_case_ids": blocked,
        "forwarded": len(forwarded),
        "forwarded_case_ids": forwarded,
    }


def summarize_route(
    runs: list[dict[str, Any]], model: str, sdk_family: str
) -> dict[str, Any]:
    decisions = [run["record"] for run in runs]
    downstream_invocations = [
        invocation
        for run in runs
        for invocation in run["downstream_invocations"]
    ]
    return {
        "blocked": sum(row["decision"] == "blocked" for row in decisions),
        "blocked_reasons": dict(
            sorted(
                Counter(
                    row["reason"] for row in decisions if row["decision"] == "blocked"
                ).items()
            )
        ),
        "captured_request_bodies": len(decisions),
        "comparative_gates": {
            name: summarize_gate(decisions, name)
            for name in (
                "mime_aware_decoded_presence",
                "body_only_task_content_consistency",
                "caller_owned_evidence_contract",
            )
        },
        "decision_audit": decisions,
        "downstream_invocation_audit": downstream_invocations,
        "downstream_invocations": len(downstream_invocations),
        "forwarded": sum(row["decision"] == "forwarded" for row in decisions),
        "model": model,
        "request_body_sha256": [row["body_sha256"] for row in decisions],
        "request_urls": sorted({row["url"] for row in decisions}),
        "sdk_family": sdk_family,
    }


def expectation_for(case: MatrixCase) -> DispatchExpectation:
    return DispatchExpectation(
        case_id=case.case_id,
        contract=TASKS[case.contract_task_id],
    )


def run_openai_case(
    route: str,
    config: dict[str, Any],
    case: MatrixCase,
    barrier: Barrier,
) -> dict[str, Any]:
    transport = GuardedTransport(
        route=route,
        sdk_family="openai",
        expectation=expectation_for(case),
        blocked_response_formatter=openai_response,
        downstream=DownstreamSpy(route=route, response_formatter=openai_response),
    )
    client = OpenAI(
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
        max_retries=0,
    )
    try:
        barrier.wait()
        completion = client.chat.completions.create(
            model=config["model"],
            messages=openai_messages(case),
            extra_body={"provider": config["provider"]},
        )
        verify_response(
            transport,
            case,
            json.loads(completion.choices[0].message.content or "{}"),
        )
    finally:
        client.close()
    return {
        "record": transport.captured[0],
        "downstream_invocations": transport.downstream.invocations,
    }


def run_openai_route(
    route: str, config: dict[str, Any], cases: tuple[MatrixCase, ...]
) -> dict[str, Any]:
    barrier = Barrier(len(cases))
    with ThreadPoolExecutor(
        max_workers=len(cases),
        thread_name_prefix=f"{route}-dispatch",
    ) as executor:
        runs = list(
            executor.map(
                lambda case: run_openai_case(route, config, case, barrier),
                cases,
            )
        )
    return summarize_route(runs, config["model"], "openai")


def run_vertex_case(case: MatrixCase, barrier: Barrier) -> dict[str, Any]:
    transport = GuardedTransport(
        route=VERTEX_ROUTE,
        sdk_family="google_genai",
        expectation=expectation_for(case),
        blocked_response_formatter=vertex_response,
        downstream=DownstreamSpy(
            route=VERTEX_ROUTE,
            response_formatter=vertex_response,
        ),
    )
    credentials = AnonymousCredentials()
    credentials.token = "test"
    client = genai.Client(
        vertexai=True,
        project="offline-project",
        location="global",
        credentials=credentials,
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(transport))
        ),
    )
    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=4000,
        response_mime_type="application/json",
    )
    try:
        barrier.wait()
        result = client.models.generate_content(
            model=VERTEX_MODEL,
            contents=vertex_contents(case),
            config=config,
        )
        verify_response(transport, case, json.loads(result.text or "{}"))
    finally:
        client.close()
    return {
        "record": transport.captured[0],
        "downstream_invocations": transport.downstream.invocations,
    }


def run_vertex_route(cases: tuple[MatrixCase, ...]) -> dict[str, Any]:
    barrier = Barrier(len(cases))
    with ThreadPoolExecutor(
        max_workers=len(cases),
        thread_name_prefix=f"{VERTEX_ROUTE}-dispatch",
    ) as executor:
        runs = list(executor.map(lambda case: run_vertex_case(case, barrier), cases))
    return summarize_route(runs, VERTEX_MODEL, "google_genai")


def case_definition(case: MatrixCase) -> dict[str, Any]:
    expected_task = TASKS[case.contract_task_id]
    return {
        "case_id": case.case_id,
        "expected_decision": case.expected_decision,
        "expected_reason": case.expected_reason,
        "expected_task_id": case.contract_task_id,
        "expected_task_text_sha256": sha256_bytes(
            expected_task.task_text.encode("utf-8")
        ),
        "input_flow": case.input_flow,
        "message_layout": case.message_layout,
        "materialized_source_uri": case.materialized_source_uri,
        "submitted_image_mime_types": [image.mime_type for image in case.images],
        "submitted_image_sha256": [sha256_bytes(image.data) for image in case.images],
        "serialized_task_id": case.serialized_task_id,
        "serialized_task_text_sha256": sha256_bytes(
            case.serialized_task_text.encode("utf-8")
        ),
        "unresolved_uri": case.unresolved_uri,
    }


def run() -> dict[str, Any]:
    cases = matrix_cases()
    transports = {
        route: run_openai_route(route, config, cases)
        for route, config in OPENAI_ROUTES.items()
    }
    transports[VERTEX_ROUTE] = run_vertex_route(cases)

    total_forwarded = sum(route["forwarded"] for route in transports.values())
    total_blocked = sum(route["blocked"] for route in transports.values())
    total_captured = sum(
        route["captured_request_bodies"] for route in transports.values()
    )
    downstream_invocations = sum(
        route["downstream_invocations"] for route in transports.values()
    )
    blocked_downstream_invocations = sum(
        row["downstream_invocations"]
        for route in transports.values()
        for row in route["decision_audit"]
        if row["decision"] == "blocked"
    )
    route_count = len(transports)
    forwarded_cases = [case.case_id for case in cases if case.expected_decision == "forwarded"]
    blocked_cases = [case.case_id for case in cases if case.expected_decision == "blocked"]
    if (
        route_count,
        total_forwarded,
        total_blocked,
        total_captured,
        downstream_invocations,
        blocked_downstream_invocations,
    ) != (5, 30, 55, 85, 30, 0):
        raise ValueError("combined matrix totals failed")
    for route, audit in transports.items():
        if (
            audit["captured_request_bodies"],
            audit["forwarded"],
            audit["blocked"],
            audit["downstream_invocations"],
        ) != (17, 6, 11, 6):
            raise ValueError(f"{route} matrix totals failed")

    control_ids = {
        case.case_id for case in cases if case.expected_decision == "forwarded"
    }
    violation_ids = {
        case.case_id for case in cases if case.expected_decision == "blocked"
    }
    gate_comparison: dict[str, dict[str, Any]] = {}
    gate_descriptions = {
        "mime_aware_decoded_presence": (
            "whole-body image presence with successful decoding and MIME agreement"
        ),
        "body_only_task_content_consistency": (
            "whole-body task marker, count, order, MIME, and digest agreement without "
            "caller-owned state"
        ),
        "caller_owned_evidence_contract": (
            "last-user-turn evidence matched to an immutable caller-owned task contract"
        ),
    }
    for gate_name, description in gate_descriptions.items():
        blocked_controls = 0
        forwarded_controls = 0
        blocked_violations = 0
        forwarded_violations = 0
        for audit in transports.values():
            gate = audit["comparative_gates"][gate_name]
            blocked = set(gate["blocked_case_ids"])
            forwarded = set(gate["forwarded_case_ids"])
            blocked_controls += len(blocked & control_ids)
            forwarded_controls += len(forwarded & control_ids)
            blocked_violations += len(blocked & violation_ids)
            forwarded_violations += len(forwarded & violation_ids)
        gate_comparison[gate_name] = {
            "blocked_controls": blocked_controls,
            "blocked_violations": blocked_violations,
            "description": description,
            "forwarded_controls": forwarded_controls,
            "forwarded_violations": forwarded_violations,
        }
    expected_gate_comparison = {
        "mime_aware_decoded_presence": (30, 0, 20, 35),
        "body_only_task_content_consistency": (30, 0, 40, 15),
        "caller_owned_evidence_contract": (30, 0, 55, 0),
    }
    for gate_name, expected in expected_gate_comparison.items():
        gate = gate_comparison[gate_name]
        observed = (
            gate["forwarded_controls"],
            gate["blocked_controls"],
            gate["blocked_violations"],
            gate["forwarded_violations"],
        )
        if observed != expected:
            raise ValueError(f"{gate_name} comparison totals failed: {observed}")

    return {
        "audit_name": "concurrent offline multi-SDK evidence-binding matrix",
        "audit_schema_version": 6,
        "byte_transformation": {
            "operation": "Pillow RGB conversion to JPEG quality=95 subsampling=0",
            "source_sha256": sha256_bytes(SOURCE_PNG),
            "transformed_sha256": sha256_bytes(TRANSFORMED_JPEG.data),
        },
        "case_definitions": [case_definition(case) for case in cases],
        "concurrent_dispatch": {
            "binding": "immutable one-shot transport expectation per SDK call",
            "calls_released_per_route": len(cases),
            "sdk_client_per_dispatch": True,
            "shared_mutable_pending_expectation": False,
            "thread_barrier": True,
            "workers_per_route": len(cases),
        },
        "dependencies": {
            "google_genai": genai.__version__,
            "httpx": httpx.__version__,
            "openai": openai.__version__,
            "pillow": PIL.__version__,
        },
        "evidence_contract": {
            "accepted_mime_types": ["image/jpeg", "image/png", "image/webp"],
            "checks": [
                "task_marker_is_in_last_user_turn",
                "canonical_task_text_matches_caller_commitment",
                "image_evidence_is_in_same_last_user_turn",
                "serialized_task_id_matches_out_of_band_expected_contract",
                "reference_is_materialized_before_serialization",
                "inline_image_decodes_to_declared_mime_type",
                "decoded_image_mime_sequence_matches_caller_commitment",
                "task_bound_image_digest_matches",
                "task_bound_image_order_matches",
                "image_count_matches_task_contract",
            ],
            "source": (
                "immutable caller-owned DispatchExpectation retained by a one-shot "
                "transport for each SDK call; serialized evidence is read only from "
                "the last user turn"
            ),
        },
        "gate_comparison": gate_comparison,
        "matrix": {
            "blocked_cases_per_route": len(blocked_cases),
            "cases_per_route": len(cases),
            "forwarded_cases_per_route": len(forwarded_cases),
            "forwarded_control_case_ids": forwarded_cases,
            "negative_case_ids": blocked_cases,
            "route_count": route_count,
            "total_cases": total_captured,
        },
        "downstream_spy": {
            "blocked_case_invocations": blocked_downstream_invocations,
            "forwarded_case_invocations": downstream_invocations,
            "total_invocations": downstream_invocations,
        },
        "network_calls": 0,
        "offline_materialization": {
            "network_calls": 0,
            "resolver": "in_memory_fixture",
            "source_sha256": sha256_bytes(REMOTE_PNG.data),
            "source_uri": REMOTE_FIXTURE_URI,
        },
        "scope": (
            "Current SDK serialization, simultaneous per-route dispatch, comparative "
            "integrity gates, last-user-turn path binding, and post-guard suppression "
            "are exercised through one-shot mock transports and a local downstream "
            "spy; this audit does not establish production effectiveness or historical "
            "behavior."
        ),
        "sdk_families": 2,
        "transport": {
            "blocked": total_blocked,
            "captured_request_bodies": total_captured,
            "downstream_invocations": downstream_invocations,
            "forwarded": total_forwarded,
        },
        "transports": transports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        parser.error(f"output directory does not exist: {args.out.parent}")
    result = run()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "blocked": result["transport"]["blocked"],
                "forwarded": result["transport"]["forwarded"],
                "model_routes": result["matrix"]["route_count"],
                "network_calls": result["network_calls"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
