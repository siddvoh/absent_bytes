#!/usr/bin/env python3
"""Audit benign and adversarial compositions of exact current SDK bodies."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import httpx
import openai
import PIL
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import types
from openai import OpenAI


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evidence_contract import (
    ClientOwnedResponseBinding,
    EvidenceFieldContract,
    inspect_serialized_response_schema,
    mock_structured_response,
    response_json_schema,
)
from independent_contract_oracle import assess_serialized_request

from audit_evidence_binding_matrix import (
    DispatchExpectation,
    DownstreamSpy,
    GuardedTransport,
    VERTEX_MODEL,
    openai_response,
    sha256_bytes,
    vertex_response,
)
from audit_evidence_mutation_properties import (
    OPENAI_CONFIG,
    GeneratedDispatch,
    build_dispatches,
    generate_contracts,
    load_contract_spec,
    openai_messages,
    replacement_image,
    vertex_contents,
)


SEED = 20260716
WORKERS = 16
ORACLE_PATH = CODE_ROOT / "independent_contract_oracle.py"


@dataclass(frozen=True)
class CapturedBody:
    case_id: str
    contract_index: int
    contract: EvidenceFieldContract
    correlation_id: str
    method: str
    raw_body: bytes
    response_binding: ClientOwnedResponseBinding
    sdk_family: str
    url: str


BENIGN_TRANSFORMS = (
    "identity_replay",
    "recursive_keys_reversed",
    "recursive_keys_sorted",
    "pretty_json_one_space",
    "pretty_json_two_spaces",
    "pretty_json_four_spaces",
    "schema_properties_reversed",
    "schema_required_reversed",
    "schema_required_rotated",
    "active_images_before_text",
    "active_text_between_images",
    "system_history",
    "prior_user_history",
    "assistant_history",
    "three_turn_history",
    "unicode_history",
)


ADVERSARIAL_OPERATORS = (
    "append_extra_image",
    "replace_first_image",
    "corrupt_first_image",
    "declared_mime_drift",
    "reverse_image_order",
    "add_out_of_scope_image",
    "append_unresolved_reference",
    "task_id_drift",
    "task_text_drift",
    "duplicate_task_marker",
    "schema_missing_property",
    "schema_extra_property",
    "schema_required_drift",
    "schema_allows_additional_properties",
)


def _control_dispatches(
    sdk_family: str,
    contracts: tuple[EvidenceFieldContract, ...],
    spec: dict[str, Any],
) -> tuple[GeneratedDispatch, ...]:
    return tuple(
        row
        for row in build_dispatches(sdk_family, contracts, spec)
        if row.is_control
    )


def _capture_openai(dispatch: GeneratedDispatch, index: int) -> CapturedBody:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return openai_response(
            mock_structured_response(dispatch.contract),
            json.loads(request.content),
        )

    client = OpenAI(
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    try:
        client.chat.completions.create(
            model=OPENAI_CONFIG["model"],
            messages=openai_messages(dispatch),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "compositional_evidence_response",
                    "strict": True,
                    "schema": response_json_schema(),
                },
            },
            extra_body={"provider": OPENAI_CONFIG["provider"]},
            extra_headers={"x-evidence-correlation-id": dispatch.correlation_id},
        )
    finally:
        client.close()
    if len(captured) != 1:
        raise ValueError(f"OpenAI capture count failed for {dispatch.correlation_id}")
    request = captured[0]
    return CapturedBody(
        case_id=dispatch.case.case_id,
        contract_index=index,
        contract=dispatch.contract,
        correlation_id=dispatch.correlation_id,
        method=request.method,
        raw_body=bytes(request.content),
        response_binding=dispatch.response_binding,
        sdk_family="openai",
        url=str(request.url),
    )


def _capture_google(dispatch: GeneratedDispatch, index: int) -> CapturedBody:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return vertex_response(
            mock_structured_response(dispatch.contract),
            json.loads(request.content),
        )

    credentials = AnonymousCredentials()
    credentials.token = "test"
    client = genai.Client(
        vertexai=True,
        project="offline-project",
        location="global",
        credentials=credentials,
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=4000,
        response_mime_type="application/json",
        response_json_schema=response_json_schema(),
    )
    try:
        client.models.generate_content(
            model=VERTEX_MODEL,
            contents=vertex_contents(dispatch),
            config=config,
        )
    finally:
        client.close()
    if len(captured) != 1:
        raise ValueError(f"Google capture count failed for {dispatch.correlation_id}")
    request = captured[0]
    return CapturedBody(
        case_id=dispatch.case.case_id,
        contract_index=index,
        contract=dispatch.contract,
        correlation_id=dispatch.correlation_id,
        method=request.method,
        raw_body=bytes(request.content),
        response_binding=dispatch.response_binding,
        sdk_family="google_genai",
        url=str(request.url),
    )


def capture_current_sdk_bodies(
    contracts: tuple[EvidenceFieldContract, ...], spec: dict[str, Any]
) -> tuple[CapturedBody, ...]:
    openai_rows = _control_dispatches("openai", contracts, spec)
    google_rows = _control_dispatches("google_genai", contracts, spec)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        openai_captures = list(
            executor.map(
                lambda item: _capture_openai(item[1], item[0]),
                enumerate(openai_rows),
            )
        )
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        google_captures = list(
            executor.map(
                lambda item: _capture_google(item[1], item[0]),
                enumerate(google_rows),
            )
        )
    captures = tuple(
        sorted(
            [*openai_captures, *google_captures],
            key=lambda row: (row.sdk_family, row.contract_index),
        )
    )
    if len(captures) != 64 or len({row.raw_body for row in captures}) != 64:
        raise ValueError("exact SDK control capture cardinality failed")
    return captures


def _recursive_order(value: Any, reverse: bool) -> Any:
    if isinstance(value, dict):
        keys = sorted(value, reverse=reverse)
        return {key: _recursive_order(value[key], reverse) for key in keys}
    if isinstance(value, list):
        return [_recursive_order(item, reverse) for item in value]
    return value


def _turns(body: dict[str, Any], sdk_family: str) -> list[dict[str, Any]]:
    key = "messages" if sdk_family == "openai" else "contents"
    turns = body.get(key)
    if not isinstance(turns, list) or not all(isinstance(row, dict) for row in turns):
        raise ValueError(f"malformed {sdk_family} turns")
    return turns


def _active_parts(body: dict[str, Any], sdk_family: str) -> list[dict[str, Any]]:
    turns = _turns(body, sdk_family)
    indexes = [index for index, row in enumerate(turns) if row.get("role") == "user"]
    if not indexes:
        raise ValueError("captured body has no user turn")
    key = "content" if sdk_family == "openai" else "parts"
    parts = turns[indexes[-1]].get(key)
    if not isinstance(parts, list) or not all(isinstance(row, dict) for row in parts):
        raise ValueError("captured active turn is not multipart")
    return parts


def _is_image(part: Mapping[str, Any], sdk_family: str) -> bool:
    if sdk_family == "openai":
        return part.get("type") == "image_url" or "image_url" in part
    return "inlineData" in part or "inline_data" in part


def _is_text(part: Mapping[str, Any], sdk_family: str) -> bool:
    if sdk_family == "openai":
        return part.get("type") == "text" and isinstance(part.get("text"), str)
    return isinstance(part.get("text"), str)


def _schema(body: dict[str, Any], sdk_family: str) -> dict[str, Any]:
    if sdk_family == "openai":
        value = body["response_format"]["json_schema"]["schema"]
    else:
        value = body["generationConfig"]["responseJsonSchema"]
    if not isinstance(value, dict):
        raise ValueError("captured response schema is absent")
    return value


def _history_turn(sdk_family: str, role: str, text: str) -> dict[str, Any]:
    if sdk_family == "openai":
        mapped_role = "assistant" if role == "model" else role
        return {"role": mapped_role, "content": text}
    mapped_role = "model" if role in {"system", "assistant"} else role
    return {"role": mapped_role, "parts": [{"text": text}]}


def benign_body(captured: CapturedBody, transform: str) -> bytes:
    if transform == "identity_replay":
        return captured.raw_body
    body = json.loads(captured.raw_body)
    if transform == "recursive_keys_reversed":
        body = _recursive_order(body, True)
        return json.dumps(body, separators=(",", ":")).encode("utf-8")
    if transform == "recursive_keys_sorted":
        body = _recursive_order(body, False)
        return json.dumps(body, separators=(",", ":")).encode("utf-8")
    if transform == "pretty_json_one_space":
        return json.dumps(body, indent=1).encode("utf-8")
    if transform == "pretty_json_two_spaces":
        return json.dumps(body, indent=2).encode("utf-8")
    if transform == "pretty_json_four_spaces":
        return json.dumps(body, indent=4).encode("utf-8")
    if transform == "schema_properties_reversed":
        schema = _schema(body, captured.sdk_family)
        schema["properties"] = dict(reversed(list(schema["properties"].items())))
    elif transform == "schema_required_reversed":
        schema = _schema(body, captured.sdk_family)
        schema["required"] = list(reversed(schema["required"]))
    elif transform == "schema_required_rotated":
        schema = _schema(body, captured.sdk_family)
        schema["required"] = [*schema["required"][1:], schema["required"][0]]
    elif transform in {"active_images_before_text", "active_text_between_images"}:
        parts = _active_parts(body, captured.sdk_family)
        images = [row for row in parts if _is_image(row, captured.sdk_family)]
        texts = [row for row in parts if _is_text(row, captured.sdk_family)]
        others = [
            row
            for row in parts
            if not _is_image(row, captured.sdk_family)
            and not _is_text(row, captured.sdk_family)
        ]
        if len(images) != 2 or len(texts) != 1 or others:
            raise ValueError("unexpected captured active content")
        if transform == "active_images_before_text":
            parts[:] = [*images, *texts]
        else:
            parts[:] = [images[0], texts[0], images[1]]
    elif transform in {
        "system_history",
        "prior_user_history",
        "assistant_history",
        "three_turn_history",
        "unicode_history",
    }:
        turns = _turns(body, captured.sdk_family)
        if transform == "system_history":
            prefix = [_history_turn(captured.sdk_family, "system", "Retained context.")]
        elif transform == "prior_user_history":
            prefix = [_history_turn(captured.sdk_family, "user", "Earlier text-only request.")]
        elif transform == "assistant_history":
            prefix = [_history_turn(captured.sdk_family, "assistant", "Earlier text-only reply.")]
        elif transform == "three_turn_history":
            prefix = [
                _history_turn(captured.sdk_family, "system", "Retained context."),
                _history_turn(captured.sdk_family, "user", "Earlier text-only request."),
                _history_turn(captured.sdk_family, "assistant", "Earlier text-only reply."),
            ]
        else:
            prefix = [_history_turn(captured.sdk_family, "assistant", "Prior note: café.")]
        turns[:0] = prefix
    else:
        raise ValueError(f"unknown benign transform: {transform}")
    return json.dumps(
        body,
        ensure_ascii=transform != "unicode_history",
        separators=(",", ":"),
    ).encode("utf-8")


def _image_parts(body: dict[str, Any], sdk_family: str) -> list[dict[str, Any]]:
    parts = _active_parts(body, sdk_family)
    images = [row for row in parts if _is_image(row, sdk_family)]
    if len(images) < 2:
        raise ValueError("captured body does not contain two inline images")
    return images


def _image_payload(part: dict[str, Any], sdk_family: str) -> tuple[str, bytes]:
    if sdk_family == "openai":
        url = part["image_url"]["url"]
        header, encoded = url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0]
    else:
        key = "inlineData" if "inlineData" in part else "inline_data"
        inline = part[key]
        mime_key = "mimeType" if key == "inlineData" else "mime_type"
        mime_type = inline[mime_key]
        encoded = inline["data"]
    return mime_type, base64.b64decode(encoded, altchars=b"-_", validate=True)


def _set_image_payload(
    part: dict[str, Any], sdk_family: str, data: bytes, mime_type: str
) -> None:
    encoded = base64.b64encode(data).decode("ascii")
    if sdk_family == "openai":
        part["image_url"]["url"] = f"data:{mime_type};base64,{encoded}"
    else:
        key = "inlineData" if "inlineData" in part else "inline_data"
        inline = part[key]
        mime_key = "mimeType" if key == "inlineData" else "mime_type"
        inline[mime_key] = mime_type
        inline["data"] = encoded


def _active_text_part(body: dict[str, Any], sdk_family: str) -> dict[str, Any]:
    parts = _active_parts(body, sdk_family)
    texts = [row for row in parts if _is_text(row, sdk_family)]
    if len(texts) != 1:
        raise ValueError("captured active task text is ambiguous")
    return texts[0]


def apply_adversarial_operator(
    body: dict[str, Any], captured: CapturedBody, operator: str
) -> None:
    sdk_family = captured.sdk_family
    if operator == "append_extra_image":
        _active_parts(body, sdk_family).append(copy.deepcopy(_image_parts(body, sdk_family)[0]))
    elif operator == "replace_first_image":
        image = _image_parts(body, sdk_family)[0]
        mime_type, original = _image_payload(image, sdk_family)
        replacement = replacement_image(captured.contract_index, mime_type)
        if replacement.data == original:
            raise ValueError("replacement image did not change bytes")
        _set_image_payload(image, sdk_family, replacement.data, mime_type)
    elif operator == "corrupt_first_image":
        image = _image_parts(body, sdk_family)[0]
        mime_type, _ = _image_payload(image, sdk_family)
        _set_image_payload(image, sdk_family, b"compositional-corrupt-image", mime_type)
    elif operator == "declared_mime_drift":
        image = _image_parts(body, sdk_family)[0]
        mime_type, data = _image_payload(image, sdk_family)
        wrong_mime = "image/jpeg" if mime_type != "image/jpeg" else "image/png"
        _set_image_payload(image, sdk_family, data, wrong_mime)
    elif operator == "reverse_image_order":
        parts = _active_parts(body, sdk_family)
        indexes = [index for index, row in enumerate(parts) if _is_image(row, sdk_family)]
        values = [parts[index] for index in indexes]
        for index, value in zip(indexes, reversed(values), strict=True):
            parts[index] = value
    elif operator == "add_out_of_scope_image":
        image = copy.deepcopy(_image_parts(body, sdk_family)[0])
        turns = _turns(body, sdk_family)
        if sdk_family == "openai":
            turns.insert(0, {"role": "user", "content": [image]})
        else:
            turns.insert(0, {"role": "user", "parts": [image]})
    elif operator == "append_unresolved_reference":
        parts = _active_parts(body, sdk_family)
        if sdk_family == "openai":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "https://evidence.example.test/unresolved.png"},
                }
            )
        else:
            parts.append(
                {
                    "fileData": {
                        "fileUri": "gs://offline-unresolved/compositional.png",
                        "mimeType": "image/png",
                    }
                }
            )
    elif operator in {"task_id_drift", "task_text_drift", "duplicate_task_marker"}:
        part = _active_text_part(body, sdk_family)
        text = part["text"]
        lines = text.splitlines()
        if operator == "task_id_drift":
            lines[0] = "evidence_task_id: adversarial-task"
        elif operator == "task_text_drift":
            lines[2:] = ["Summarize only the transport metadata."]
        else:
            lines.insert(1, lines[0])
        part["text"] = "\n".join(lines)
    elif operator == "schema_missing_property":
        _schema(body, sdk_family)["properties"].pop("clinical_assessment")
    elif operator == "schema_extra_property":
        _schema(body, sdk_family)["properties"]["provider_note"] = {
            "type": "string"
        }
    elif operator == "schema_required_drift":
        _schema(body, sdk_family)["required"].remove("differentials")
    elif operator == "schema_allows_additional_properties":
        _schema(body, sdk_family)["additionalProperties"] = True
    else:
        raise ValueError(f"unknown adversarial operator: {operator}")


def adversarial_chains() -> tuple[tuple[str, ...], ...]:
    rng = random.Random(SEED)
    singles = [(name,) for name in ADVERSARIAL_OPERATORS]
    pairs = rng.sample(list(itertools.combinations(ADVERSARIAL_OPERATORS, 2)), 24)
    triples = rng.sample(list(itertools.combinations(ADVERSARIAL_OPERATORS, 3)), 18)
    quadruples = rng.sample(list(itertools.combinations(ADVERSARIAL_OPERATORS, 4)), 4)
    quintuples = rng.sample(list(itertools.combinations(ADVERSARIAL_OPERATORS, 5)), 4)
    chains = tuple([*singles, *pairs, *triples, *quadruples, *quintuples])
    if len(chains) != 64 or len(set(chains)) != 64:
        raise ValueError("adversarial composition cardinality failed")
    return chains


def adversarial_body(captured: CapturedBody, chain: tuple[str, ...]) -> bytes:
    body = json.loads(captured.raw_body)
    for operator in chain:
        apply_adversarial_operator(body, captured, operator)
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def execute_body(
    captured: CapturedBody,
    body_bytes: bytes,
    *,
    audit_case_id: str,
    case_kind: str,
    operator_chain: tuple[str, ...],
    expected_forward: bool,
    task_registry: Mapping[str, EvidenceFieldContract],
) -> dict[str, Any]:
    formatter = openai_response if captured.sdk_family == "openai" else vertex_response
    downstream = DownstreamSpy(captured.sdk_family, formatter)
    guard = GuardedTransport(
        route=f"compositional-{captured.sdk_family}",
        sdk_family=captured.sdk_family,
        expectation=DispatchExpectation(
            captured.case_id,
            captured.contract,
            captured.correlation_id,
            captured.response_binding.commitment(),
        ),
        blocked_response_formatter=formatter,
        downstream=downstream,
        task_registry=task_registry,
        response_schema_inspector=inspect_serialized_response_schema,
    )
    request = httpx.Request(
        captured.method,
        captured.url,
        headers={"content-type": "application/json"},
        content=body_bytes,
    )
    guard(request)
    record = guard.captured[0]
    oracle = assess_serialized_request(
        body_bytes,
        captured.sdk_family,
        captured.contract.commitment(),
        captured.contract.task_text,
        captured.correlation_id,
        captured.response_binding.commitment(),
    )
    return {
        "audit_case_id": audit_case_id,
        "body_sha256": sha256_bytes(body_bytes),
        "case_kind": case_kind,
        "contract_sha256": captured.contract.sha256,
        "correlation_id": captured.correlation_id,
        "decision_agreement": oracle.allowed == (record["decision"] == "forwarded"),
        "downstream_invocations": len(downstream.invocations),
        "expected_forward": expected_forward,
        "guard_decision": record["decision"],
        "guard_reason": record["reason"],
        "operator_chain": list(operator_chain),
        "operator_depth": len(operator_chain),
        "oracle_allowed": oracle.allowed,
        "oracle_reason": oracle.reason,
        "sdk_family": captured.sdk_family,
        "source_body_sha256": sha256_bytes(captured.raw_body),
        "source_case_id": captured.case_id,
        "task_id": captured.contract.task_id,
    }


def audit_captured_body(
    captured: CapturedBody,
    task_registry: Mapping[str, EvidenceFieldContract],
    chains: tuple[tuple[str, ...], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, transform in enumerate(BENIGN_TRANSFORMS):
        rows.append(
            execute_body(
                captured,
                benign_body(captured, transform),
                audit_case_id=f"{captured.correlation_id}-benign-{index:02d}",
                case_kind="benign",
                operator_chain=(transform,),
                expected_forward=True,
                task_registry=task_registry,
            )
        )
    for index, chain in enumerate(chains):
        rows.append(
            execute_body(
                captured,
                adversarial_body(captured, chain),
                audit_case_id=f"{captured.correlation_id}-adversarial-{index:02d}",
                case_kind="adversarial",
                operator_chain=chain,
                expected_forward=False,
                task_registry=task_registry,
            )
        )
    if len(rows) != 80:
        raise ValueError(f"per-body audit cardinality failed for {captured.correlation_id}")
    return rows


def summarize(
    captures: tuple[CapturedBody, ...], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    benign = [row for row in rows if row["case_kind"] == "benign"]
    adversarial = [row for row in rows if row["case_kind"] == "adversarial"]
    totals = {
        "adversarial_blocked": sum(row["guard_decision"] == "blocked" for row in adversarial),
        "adversarial_cases": len(adversarial),
        "adversarial_downstream_invocations": sum(row["downstream_invocations"] for row in adversarial),
        "benign_cases": len(benign),
        "benign_forwarded": sum(row["guard_decision"] == "forwarded" for row in benign),
        "benign_oracle_allowed": sum(row["oracle_allowed"] for row in benign),
        "decision_agreements": sum(row["decision_agreement"] for row in rows),
        "exact_sdk_source_bodies": len(captures),
        "false_blocks": sum(row["guard_decision"] != "forwarded" for row in benign),
        "false_forwards": sum(row["guard_decision"] != "blocked" for row in adversarial),
        "logical_cases": len(rows),
        "network_calls": 0,
        "oracle_blocked_adversarial": sum(not row["oracle_allowed"] for row in adversarial),
        "unique_body_sha256": len({row["body_sha256"] for row in rows}),
    }
    expected = {
        "adversarial_blocked": 4096,
        "adversarial_cases": 4096,
        "adversarial_downstream_invocations": 0,
        "benign_cases": 1024,
        "benign_forwarded": 1024,
        "benign_oracle_allowed": 1024,
        "decision_agreements": 5120,
        "exact_sdk_source_bodies": 64,
        "false_blocks": 0,
        "false_forwards": 0,
        "logical_cases": 5120,
        "network_calls": 0,
        "oracle_blocked_adversarial": 4096,
        "unique_body_sha256": 5120,
    }
    if totals != expected:
        raise ValueError(f"compositional audit totals failed: {totals}")

    benign_summary = {}
    for transform in BENIGN_TRANSFORMS:
        selected = [row for row in benign if row["operator_chain"] == [transform]]
        benign_summary[transform] = {
            "cases": len(selected),
            "forwarded": sum(row["guard_decision"] == "forwarded" for row in selected),
            "oracle_allowed": sum(row["oracle_allowed"] for row in selected),
        }
    depth_summary = {}
    for depth in range(1, 6):
        selected = [row for row in adversarial if row["operator_depth"] == depth]
        depth_summary[str(depth)] = {
            "blocked": sum(row["guard_decision"] == "blocked" for row in selected),
            "cases": len(selected),
            "oracle_blocked": sum(not row["oracle_allowed"] for row in selected),
        }

    return {
        "audit_name": "exact-body compositional evidence-contract audit",
        "audit_schema_version": 1,
        "dependencies": {
            "google_genai": genai.__version__,
            "httpx": httpx.__version__,
            "openai": openai.__version__,
            "pillow": PIL.__version__,
        },
        "benign_summary": benign_summary,
        "compositional_depth_summary": depth_summary,
        "exact_sdk_body_manifest": [
            {
                "body_sha256": sha256_bytes(row.raw_body),
                "case_id": row.case_id,
                "contract_sha256": row.contract.sha256,
                "correlation_id": row.correlation_id,
                "client_response_binding": row.response_binding.commitment(),
                "sdk_family": row.sdk_family,
                "task_id": row.contract.task_id,
                "url": row.url,
            }
            for row in captures
        ],
        "independent_oracle": {
            "implementation_path": "code/independent_contract_oracle.py",
            "implementation_sha256": sha256_bytes(ORACLE_PATH.read_bytes()),
        },
        "logical_audit": sorted(rows, key=lambda row: row["audit_case_id"]),
        "operator_manifest": {
            "adversarial": list(ADVERSARIAL_OPERATORS),
            "benign": list(BENIGN_TRANSFORMS),
            "composition_seed": SEED,
        },
        "scope": (
            "The current OpenAI and Google Gen AI SDKs serialize 64 control requests. "
            "Their exact httpx request bodies are captured before guard logic, then replayed "
            "under benign history, ordering, and JSON encodings or mutated with deterministic "
            "one- through five-fault compositions. A separately implemented oracle and the "
            "enforcement guard inspect identical bytes against sidecar caller commitments. "
            "This is a local conformance and false-block audit; it makes no provider call and "
            "does not establish provider acceptance or production incident frequency."
        ),
        "totals": totals,
    }


def run() -> dict[str, Any]:
    spec = load_contract_spec()
    contracts = generate_contracts(spec)
    task_registry = MappingProxyType({row.task_id: row for row in contracts})
    captures = capture_current_sdk_bodies(contracts, spec)
    chains = adversarial_chains()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        batches = list(
            executor.map(
                lambda row: audit_captured_body(row, task_registry, chains),
                captures,
            )
        )
    rows = [item for batch in batches for item in batch]
    return summarize(captures, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        parser.error(f"output directory does not exist: {args.out.parent}")
    result = run()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
