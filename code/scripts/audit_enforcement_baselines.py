#!/usr/bin/env python3
"""Compare request-enforcement baselines on three frozen local corpora.

The corpus adapter reuses only released fixture and SDK serialization helpers.
Every reconstructed body is checked against its frozen SHA-256 before any
baseline sees it. Candidate decisions below are standalone implementations and
do not call the released guard or either released contract decision oracle.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib
import importlib.metadata
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx
import openai
import PIL
import pydantic
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import types
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

import draft202012_integrated_envelope as integrated_envelope
import pydantic_integrated_envelope as pydantic_envelope


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
SCRIPTS_ROOT = CODE_ROOT / "scripts"

RELEASE_FILE_SHA256 = {
    "code/evidence_contract.py": (
        "8a45a573b83c745c87e55d90aa91dbacbf6f1afb73ce7a90fb860f07bcd1ac68"
    ),
    "code/scripts/audit_compositional_requests.py": (
        "349e27921d47d19a8bb491c55d0a7ab8b8678c61419d5d5125fb52c6692a20ab"
    ),
    "code/scripts/audit_evidence_binding_matrix.py": (
        "0156d09e2890e5462ea116746c82e4f3da1378b74fa71a5271977c14b16ac25f"
    ),
    "code/scripts/audit_evidence_mutation_properties.py": (
        "121bd8d75386bd350eedaa11578b18c144a371e7ebe27bdd15907e1f028a7677"
    ),
    "data/compositional_request_audit.json": (
        "7f8ddd0625b7cbdf7dd0b6bc3ffba43034941432f61d0a9ab78fa1db28e417fd"
    ),
    "data/configs/evidence_field_contract.json": (
        "050749139fff0e02e0391b1f5789abaaef8a8674450f23c0ca323a7f6a202b12"
    ),
    "data/evidence_binding_matrix_audit.json": (
        "f0fcfb15b16e4b6a07d6448e5f0e15befd4d84a316074d8d848f55b94c6a3158"
    ),
    "data/evidence_mutation_property_audit.json": (
        "82764ad562774997e20ffcae98a7dd13a905dd535fc5a758bad5e7a7028958ca"
    ),
}

EXPECTED_DENOMINATORS = {
    "frozen_matrix": {"controls": 30, "total": 85, "violations": 55},
    "generated_exact_body": {
        "controls": 64,
        "total": 960,
        "violations": 896,
    },
    "compositional": {
        "controls": 1024,
        "total": 5120,
        "violations": 4096,
    },
}

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
SUPPORTED_MIMES = frozenset(MIME_BY_FORMAT.values())
TASK_ID_PATTERN = re.compile(r"(?m)^evidence_task_id: ([a-z0-9-]+)$")
CASE_ID_PATTERN = re.compile(r"(?m)^evidence_case_id: ([a-z0-9-]+)$")
TASK_METADATA_LINE_PATTERN = re.compile(
    r"^evidence_(?:task_id|case_id): [a-z0-9-]+$"
)


class AuditDriftError(RuntimeError):
    """Raised before scoring when a frozen input or denominator drifts."""


@dataclass(frozen=True)
class ExpectedImage:
    mime_type: str
    sha256: str


@dataclass(frozen=True)
class ExpectedBinding:
    client_response_binding: Mapping[str, Any]
    expected_response_binding: Mapping[str, Any]
    images: tuple[ExpectedImage, ...]
    response_schema_required: bool
    serialized_response_schema: Mapping[str, Any] | None
    task_text: str

    @property
    def active_field(self) -> str:
        return str(self.expected_response_binding["active_field"])

    @property
    def contract_sha256(self) -> str:
        return str(self.expected_response_binding["contract_sha256"])

    @property
    def contract_version(self) -> int:
        return int(self.expected_response_binding["contract_version"])

    @property
    def correlation_id(self) -> str:
        return str(self.expected_response_binding["correlation_id"])

    @property
    def task_id(self) -> str:
        return str(self.expected_response_binding["task_id"])

    @property
    def task_text_sha256(self) -> str:
        return str(self.expected_response_binding["task_text_sha256"])


@dataclass(frozen=True)
class CorpusCase:
    audit_case_id: str
    body: bytes
    expected: ExpectedBinding
    is_control: bool
    registry: Mapping[str, ExpectedBinding]
    sdk_family: str
    source_case_id: str


@dataclass(frozen=True)
class InlineImage:
    base64_valid: bool
    data: bytes
    declared_mime_type: str
    decoded_mime_type: str | None


@dataclass(frozen=True)
class RequestView:
    active_images: tuple[InlineImage, ...]
    active_references: tuple[str, ...]
    active_task_id: str
    active_task_text: str
    body: Mapping[str, Any]
    global_images: tuple[InlineImage, ...]
    global_references: tuple[str, ...]
    global_task_id: str
    out_of_scope_image_count: int
    response_schema: Mapping[str, Any] | None
    shape_reason: str
    shape_valid: bool


Decision = tuple[bool, str]
DecisionFunction = Callable[[CorpusCase, RequestView], Decision]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditDriftError(message)


def verify_body_sha(body: bytes, expected_sha256: str, case_id: str) -> None:
    """Fail closed if a reconstructed body is not the frozen body."""
    actual = sha256_bytes(body)
    if actual != expected_sha256:
        raise AuditDriftError(
            f"body SHA-256 drift for {case_id}: expected {expected_sha256}, got {actual}"
        )


def verify_release_inputs() -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative_path, expected in RELEASE_FILE_SHA256.items():
        path = ARTIFACT_ROOT / relative_path
        _require(path.is_file(), f"missing frozen release input: {relative_path}")
        actual = sha256_bytes(path.read_bytes())
        _require(
            actual == expected,
            f"release input SHA-256 drift for {relative_path}: "
            f"expected {expected}, got {actual}",
        )
        verified[relative_path] = actual
    return verified


def _load_json(relative_path: str) -> dict[str, Any]:
    value = json.loads((ARTIFACT_ROOT / relative_path).read_text())
    _require(isinstance(value, dict), f"{relative_path} is not a JSON object")
    return value


def _load_corpus_generators() -> tuple[Any, Any, Any]:
    """Load released serialization helpers only after commitment verification."""
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    matrix = importlib.import_module("audit_evidence_binding_matrix")
    generated = importlib.import_module("audit_evidence_mutation_properties")
    compositional = importlib.import_module("audit_compositional_requests")
    return matrix, generated, compositional


def _expected_binding(
    source_contract: Any,
    *,
    correlation_id: str,
    response_schema_required: bool,
    client_response_binding: Mapping[str, Any] | None = None,
    serialized_response_schema: Mapping[str, Any] | None = None,
) -> ExpectedBinding:
    images = tuple(
        ExpectedImage(str(image.mime_type), sha256_bytes(bytes(image.data)))
        for image in source_contract.images
    )
    task_text = str(source_contract.task_text)
    commitment = {
        "active_field": str(source_contract.active_field),
        "contract_version": int(source_contract.contract_version),
        "ordered_image_mime_types": [image.mime_type for image in images],
        "ordered_image_sha256": [image.sha256 for image in images],
        "task_id": str(source_contract.task_id),
        "task_text_sha256": sha256_bytes(task_text.encode("utf-8")),
    }
    digest = sha256_bytes(canonical_json_bytes(commitment))
    released_digest = getattr(source_contract, "sha256", None)
    if released_digest is not None:
        _require(
            str(released_digest) == digest,
            f"source contract digest drift for {source_contract.task_id}",
        )
    expected_response_binding = {
        "task_id": commitment["task_id"],
        "task_text_sha256": commitment["task_text_sha256"],
        "contract_sha256": digest,
        "contract_version": commitment["contract_version"],
        "active_field": commitment["active_field"],
        "correlation_id": correlation_id,
    }
    observed_response_binding = (
        dict(client_response_binding)
        if isinstance(client_response_binding, Mapping)
        else expected_response_binding
    )
    if response_schema_required:
        _require(
            isinstance(serialized_response_schema, Mapping),
            f"serialized response schema is absent for {source_contract.task_id}",
        )
    normalized_response_schema = (
        dict(serialized_response_schema)
        if isinstance(serialized_response_schema, Mapping)
        else None
    )
    if normalized_response_schema is not None and isinstance(
        normalized_response_schema.get("required"), list
    ):
        normalized_response_schema["required"] = sorted(
            normalized_response_schema["required"]
        )
    return ExpectedBinding(
        client_response_binding=observed_response_binding,
        expected_response_binding=expected_response_binding,
        images=images,
        response_schema_required=response_schema_required,
        serialized_response_schema=normalized_response_schema,
        task_text=task_text,
    )


def _openai_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "offline-baseline-capture",
            "object": "chat.completion",
            "created": 0,
            "model": str(body.get("model") or "offline"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )


def _google_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "{}"}]},
                    "finishReason": "STOP",
                }
            ]
        },
    )


def _capture_matrix_openai(
    matrix: Any, route: str, config: Mapping[str, Any], cases: tuple[Any, ...]
) -> dict[str, bytes]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _openai_response(request)

    client = OpenAI(
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    try:
        for case in cases:
            before = len(captured)
            client.chat.completions.create(
                model=config["model"],
                messages=matrix.openai_messages(case),
                extra_body={"provider": config["provider"]},
            )
            _require(
                len(captured) == before + 1,
                f"{route}/{case.case_id} did not serialize exactly one body",
            )
    finally:
        client.close()
    return {
        case.case_id: bytes(request.content)
        for case, request in zip(cases, captured, strict=True)
    }


def _google_client(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    credentials = AnonymousCredentials()
    credentials.token = "test"
    return genai.Client(
        vertexai=True,
        project="offline-project",
        location="global",
        credentials=credentials,
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )


def _capture_matrix_google(matrix: Any, cases: tuple[Any, ...]) -> dict[str, bytes]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _google_response(request)

    client = _google_client(handler)
    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=4000,
        response_mime_type="application/json",
    )
    try:
        for case in cases:
            before = len(captured)
            client.models.generate_content(
                model=matrix.VERTEX_MODEL,
                contents=matrix.vertex_contents(case),
                config=config,
            )
            _require(
                len(captured) == before + 1,
                f"{matrix.VERTEX_ROUTE}/{case.case_id} did not serialize one body",
            )
    finally:
        client.close()
    return {
        case.case_id: bytes(request.content)
        for case, request in zip(cases, captured, strict=True)
    }


def _validate_denominators(stratum: str, cases: list[CorpusCase]) -> None:
    actual = {
        "controls": sum(case.is_control for case in cases),
        "total": len(cases),
        "violations": sum(not case.is_control for case in cases),
    }
    _require(
        actual == EXPECTED_DENOMINATORS[stratum],
        f"{stratum} denominator drift: expected "
        f"{EXPECTED_DENOMINATORS[stratum]}, got {actual}",
    )
    identifiers = [case.audit_case_id for case in cases]
    _require(
        len(set(identifiers)) == len(identifiers),
        f"{stratum} case identifiers are not unique",
    )


def _load_frozen_matrix(matrix: Any, frozen: Mapping[str, Any]) -> list[CorpusCase]:
    cases = tuple(matrix.matrix_cases())
    source_by_id = {case.case_id: case for case in cases}
    _require(len(source_by_id) == 17, "matrix source case denominator drift")
    registry = {
        task_id: _expected_binding(
            contract,
            correlation_id=f"matrix-registry/{task_id}",
            response_schema_required=False,
        )
        for task_id, contract in matrix.TASKS.items()
    }
    serialized: dict[str, dict[str, bytes]] = {}
    for route, config in matrix.OPENAI_ROUTES.items():
        serialized[route] = _capture_matrix_openai(matrix, route, config, cases)
    serialized[matrix.VERTEX_ROUTE] = _capture_matrix_google(matrix, cases)

    transports = frozen.get("transports")
    _require(isinstance(transports, Mapping), "matrix transport manifest is absent")
    _require(
        set(transports) == set(serialized),
        "matrix route denominator or identity drift",
    )
    result: list[CorpusCase] = []
    for route in sorted(transports):
        route_audit = transports[route]
        rows = route_audit.get("decision_audit")
        _require(isinstance(rows, list) and len(rows) == 17, f"{route} row drift")
        frozen_hashes = route_audit.get("request_body_sha256")
        _require(
            frozen_hashes == [row.get("body_sha256") for row in rows],
            f"{route} body manifest is internally inconsistent",
        )
        for row in rows:
            case_id = str(row.get("case_id"))
            _require(case_id in source_by_id, f"unknown matrix case: {route}/{case_id}")
            source = source_by_id[case_id]
            _require(
                row.get("decision") == source.expected_decision,
                f"matrix label drift for {route}/{case_id}",
            )
            body = serialized[route][case_id]
            audit_case_id = f"{route}/{case_id}"
            verify_body_sha(body, str(row.get("body_sha256")), audit_case_id)
            result.append(
                CorpusCase(
                    audit_case_id=audit_case_id,
                    body=body,
                    expected=_expected_binding(
                        matrix.TASKS[source.contract_task_id],
                        correlation_id=audit_case_id,
                        response_schema_required=False,
                    ),
                    is_control=row.get("decision") == "forwarded",
                    registry=registry,
                    sdk_family=str(route_audit.get("sdk_family")),
                    source_case_id=case_id,
                )
            )
    _validate_denominators("frozen_matrix", result)
    matrix_totals = frozen.get("matrix")
    _require(
        isinstance(matrix_totals, Mapping)
        and matrix_totals.get("total_cases") == len(result),
        "matrix frozen total disagrees with reconstructed denominator",
    )
    return sorted(result, key=lambda case: case.audit_case_id)


def _capture_generated_openai(generated: Any, dispatches: tuple[Any, ...]) -> dict[str, bytes]:
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        correlation_id = request.headers.get("x-evidence-correlation-id", "")
        _require(correlation_id not in captured, f"duplicate generated body: {correlation_id}")
        captured[correlation_id] = bytes(request.content)
        return _openai_response(request)

    client = OpenAI(
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
    )
    try:
        for dispatch in dispatches:
            client.chat.completions.create(
                model=generated.OPENAI_CONFIG["model"],
                messages=generated.openai_messages(dispatch),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": (
                            f"evidence_response_v{dispatch.contract.contract_version}"
                        ),
                        "strict": True,
                        "schema": generated.response_json_schema(),
                    },
                },
                extra_body={"provider": generated.OPENAI_CONFIG["provider"]},
                extra_headers={"x-evidence-correlation-id": dispatch.correlation_id},
            )
    finally:
        client.close()
    _require(len(captured) == len(dispatches), "generated OpenAI capture drift")
    return captured


def _capture_generated_google(generated: Any, dispatches: tuple[Any, ...]) -> dict[str, bytes]:
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(bytes(request.content))
        return _google_response(request)

    client = _google_client(handler)
    try:
        for dispatch in dispatches:
            config = types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=4000,
                response_mime_type="application/json",
                response_json_schema=generated.response_json_schema(),
            )
            before = len(captured)
            client.models.generate_content(
                model=generated.VERTEX_MODEL,
                contents=generated.vertex_contents(dispatch),
                config=config,
            )
            _require(
                len(captured) == before + 1,
                f"{dispatch.correlation_id} did not serialize exactly one body",
            )
    finally:
        client.close()
    _require(len(captured) == len(dispatches), "generated Google capture drift")
    return {
        dispatch.correlation_id: body
        for dispatch, body in zip(dispatches, captured, strict=True)
    }


def _load_generated_exact_body(
    generated: Any,
    frozen: Mapping[str, Any],
) -> tuple[list[CorpusCase], dict[str, Any], tuple[Any, ...]]:
    spec = generated.load_contract_spec()
    contracts = tuple(generated.generate_contracts(spec))
    _require(len(contracts) == 32, "generated contract denominator drift")
    registry = {
        contract.task_id: _expected_binding(
            contract,
            correlation_id=f"generated-registry/{contract.task_id}",
            response_schema_required=False,
        )
        for contract in contracts
    }
    openai_dispatches = tuple(generated.build_dispatches("openai", contracts, spec))
    google_dispatches = tuple(
        generated.build_dispatches("google_genai", contracts, spec)
    )
    dispatches = (*openai_dispatches, *google_dispatches)
    dispatch_by_id = {dispatch.correlation_id: dispatch for dispatch in dispatches}
    _require(len(dispatch_by_id) == 960, "generated correlation denominator drift")
    bodies = {
        **_capture_generated_openai(generated, openai_dispatches),
        **_capture_generated_google(generated, google_dispatches),
    }
    rows = frozen.get("logical_audit")
    _require(isinstance(rows, list), "generated logical audit is absent")
    _require(len(rows) == len(dispatch_by_id), "generated frozen row denominator drift")
    frozen_by_id = {str(row.get("correlation_id")): row for row in rows}
    _require(
        len(frozen_by_id) == len(rows) and set(frozen_by_id) == set(dispatch_by_id),
        "generated frozen correlation identity drift",
    )
    result: list[CorpusCase] = []
    for correlation_id in sorted(dispatch_by_id):
        dispatch = dispatch_by_id[correlation_id]
        row = frozen_by_id[correlation_id]
        _require(
            row.get("case_id") == dispatch.case.case_id
            and row.get("sdk_family") == dispatch.sdk_family
            and bool(row.get("is_control")) == bool(dispatch.is_control),
            f"generated metadata drift for {correlation_id}",
        )
        body = bodies[correlation_id]
        verify_body_sha(body, str(row.get("body_sha256")), correlation_id)
        expected = _expected_binding(
            dispatch.contract,
            correlation_id=correlation_id,
            response_schema_required=True,
            client_response_binding=row.get("client_response_binding"),
            serialized_response_schema=row.get("serialized_response_schema"),
        )
        _require(
            row.get("contract_sha256") == expected.contract_sha256,
            f"generated sidecar contract drift for {correlation_id}",
        )
        result.append(
            CorpusCase(
                audit_case_id=correlation_id,
                body=body,
                expected=expected,
                is_control=bool(dispatch.is_control),
                registry=registry,
                sdk_family=str(dispatch.sdk_family),
                source_case_id=str(dispatch.case.case_id),
            )
        )
    _validate_denominators("generated_exact_body", result)
    totals = frozen.get("totals")
    _require(
        isinstance(totals, Mapping)
        and totals.get("logical_dispatches") == len(result)
        and totals.get("control_dispatches") == sum(case.is_control for case in result)
        and totals.get("mutation_dispatches") == sum(
            not case.is_control for case in result
        ),
        "generated frozen totals disagree with reconstructed denominators",
    )
    return result, spec, contracts


def _load_compositional(
    compositional: Any,
    frozen: Mapping[str, Any],
    spec: dict[str, Any],
    contracts: tuple[Any, ...],
) -> list[CorpusCase]:
    captures = tuple(compositional.capture_current_sdk_bodies(contracts, spec))
    manifest = frozen.get("exact_sdk_body_manifest")
    _require(isinstance(manifest, list) and len(manifest) == 64, "source body drift")
    manifest_by_id = {str(row.get("correlation_id")): row for row in manifest}
    capture_by_id = {capture.correlation_id: capture for capture in captures}
    _require(
        len(capture_by_id) == 64 and set(capture_by_id) == set(manifest_by_id),
        "compositional source body identity drift",
    )
    for correlation_id, capture in capture_by_id.items():
        row = manifest_by_id[correlation_id]
        verify_body_sha(
            bytes(capture.raw_body),
            str(row.get("body_sha256")),
            f"compositional-source/{correlation_id}",
        )
        _require(
            row.get("case_id") == capture.case_id
            and row.get("sdk_family") == capture.sdk_family,
            f"compositional source metadata drift for {correlation_id}",
        )

    registry = {
        contract.task_id: _expected_binding(
            contract,
            correlation_id=f"compositional-registry/{contract.task_id}",
            response_schema_required=False,
        )
        for contract in contracts
    }
    frozen_rows = frozen.get("logical_audit")
    _require(isinstance(frozen_rows, list), "compositional logical audit is absent")
    frozen_by_id = {str(row.get("audit_case_id")): row for row in frozen_rows}
    _require(
        len(frozen_by_id) == len(frozen_rows),
        "compositional frozen case identifiers are not unique",
    )
    chains = tuple(compositional.adversarial_chains())
    generated_rows: list[tuple[Any, str, str, tuple[str, ...], bytes]] = []
    for capture in captures:
        for index, transform in enumerate(compositional.BENIGN_TRANSFORMS):
            generated_rows.append(
                (
                    capture,
                    f"{capture.correlation_id}-benign-{index:02d}",
                    "benign",
                    (transform,),
                    bytes(compositional.benign_body(capture, transform)),
                )
            )
        for index, chain in enumerate(chains):
            generated_rows.append(
                (
                    capture,
                    f"{capture.correlation_id}-adversarial-{index:02d}",
                    "adversarial",
                    tuple(chain),
                    bytes(compositional.adversarial_body(capture, chain)),
                )
            )
    _require(
        len(generated_rows) == len(frozen_by_id),
        "compositional generated denominator drift",
    )
    result: list[CorpusCase] = []
    for capture, audit_case_id, case_kind, chain, body in generated_rows:
        _require(
            audit_case_id in frozen_by_id,
            f"unknown compositional case: {audit_case_id}",
        )
        row = frozen_by_id[audit_case_id]
        _require(
            row.get("case_kind") == case_kind
            and row.get("operator_chain") == list(chain)
            and row.get("source_case_id") == capture.case_id
            and row.get("sdk_family") == capture.sdk_family,
            f"compositional metadata drift for {audit_case_id}",
        )
        verify_body_sha(body, str(row.get("body_sha256")), audit_case_id)
        verify_body_sha(
            bytes(capture.raw_body),
            str(row.get("source_body_sha256")),
            f"{audit_case_id}/source",
        )
        expected = _expected_binding(
            capture.contract,
            correlation_id=capture.correlation_id,
            response_schema_required=True,
            client_response_binding=capture.response_binding.commitment(),
            serialized_response_schema=_serialized_response_schema(
                _response_schema(
                    json.loads(capture.raw_body),
                    capture.sdk_family,
                ),
                capture.sdk_family,
            ),
        )
        _require(
            row.get("contract_sha256") == expected.contract_sha256,
            f"compositional sidecar contract drift for {audit_case_id}",
        )
        result.append(
            CorpusCase(
                audit_case_id=audit_case_id,
                body=body,
                expected=expected,
                is_control=case_kind == "benign",
                registry=registry,
                sdk_family=str(capture.sdk_family),
                source_case_id=str(capture.case_id),
            )
        )
    _validate_denominators("compositional", result)
    totals = frozen.get("totals")
    _require(
        isinstance(totals, Mapping)
        and totals.get("logical_cases") == len(result)
        and totals.get("benign_cases") == sum(case.is_control for case in result)
        and totals.get("adversarial_cases") == sum(
            not case.is_control for case in result
        ),
        "compositional frozen totals disagree with reconstructed denominators",
    )
    return sorted(result, key=lambda case: case.audit_case_id)


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


def _inspect_inline_evidence(value: Any) -> tuple[tuple[InlineImage, ...], tuple[str, ...]]:
    images: list[InlineImage] = []
    references: list[str] = []

    def add_image(encoded: str, declared_mime_type: str) -> None:
        data, valid = _decode_base64(encoded)
        images.append(
            InlineImage(
                base64_valid=valid,
                data=data,
                declared_mime_type=declared_mime_type.lower(),
                decoded_mime_type=_decoded_mime_type(data) if valid else None,
            )
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
                add_image(encoded, header[5:].split(";", 1)[0])
            elif url:
                references.append(url)
        for key in ("inlineData", "inline_data"):
            inline = item.get(key)
            if not isinstance(inline, Mapping):
                continue
            consumed.add(key)
            add_image(
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
    return tuple(images), tuple(references)


def _text_fields(value: Any) -> list[str]:
    texts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, Mapping):
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


def _request_metadata(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        value = {"content": value}
    text = "\n".join(_text_fields(value))
    task_ids = TASK_ID_PATTERN.findall(text)
    case_ids = CASE_ID_PATTERN.findall(text)
    if len(task_ids) != 1 or len(case_ids) != 1:
        return "", ""
    return task_ids[0], case_ids[0]


def _canonical_task_text(value: Any) -> str:
    if isinstance(value, str):
        value = {"content": value}
    lines: list[str] = []
    for text in _text_fields(value):
        lines.extend(
            line.rstrip()
            for line in text.splitlines()
            if not TASK_METADATA_LINE_PATTERN.fullmatch(line.strip())
        )
    return "\n".join(lines).strip()


def _shape_only_typed_request(body: Mapping[str, Any], sdk_family: str) -> Decision:
    collection_key = "messages" if sdk_family == "openai" else "contents"
    content_key = "content" if sdk_family == "openai" else "parts"
    turns = body.get(collection_key)
    if not isinstance(turns, list):
        return False, "turn_collection_not_a_list"
    for turn in turns:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("role"), str):
            return False, "turn_shape_invalid"
        content = turn.get(content_key)
        if sdk_family == "openai" and isinstance(content, str):
            continue
        if not isinstance(content, list):
            return False, "turn_content_not_string_or_part_list"
        for part in content:
            if not isinstance(part, Mapping):
                return False, "part_not_an_object"
            if sdk_family == "openai":
                if part.get("type") == "text":
                    if not isinstance(part.get("text"), str):
                        return False, "text_part_shape_invalid"
                elif part.get("type") == "image_url" or "image_url" in part:
                    image_url = part.get("image_url")
                    if not isinstance(image_url, Mapping) or not isinstance(
                        image_url.get("url"), str
                    ):
                        return False, "image_url_part_shape_invalid"
                else:
                    return False, "unsupported_part_shape"
            else:
                variants = sum(
                    key in part
                    for key in ("text", "inlineData", "inline_data", "fileData", "file_data")
                )
                if variants != 1:
                    return False, "google_part_variant_invalid"
                if "text" in part and not isinstance(part.get("text"), str):
                    return False, "text_part_shape_invalid"
                for key in ("inlineData", "inline_data"):
                    if key not in part:
                        continue
                    inline = part[key]
                    mime_key = "mimeType" if key == "inlineData" else "mime_type"
                    if (
                        not isinstance(inline, Mapping)
                        or not isinstance(inline.get("data"), str)
                        or not isinstance(inline.get(mime_key), str)
                    ):
                        return False, "inline_data_part_shape_invalid"
                for key in ("fileData", "file_data"):
                    if key not in part:
                        continue
                    file_data = part[key]
                    uri_key = "fileUri" if key == "fileData" else "file_uri"
                    if not isinstance(file_data, Mapping) or not isinstance(
                        file_data.get(uri_key), str
                    ):
                        return False, "file_data_part_shape_invalid"
    response_container = (
        body.get("response_format")
        if sdk_family == "openai"
        else body.get("generationConfig")
    )
    if response_container is not None and not isinstance(response_container, Mapping):
        return False, "response_container_shape_invalid"
    return True, "shape_representable"


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


def _request_view(case: CorpusCase) -> RequestView:
    try:
        body = json.loads(case.body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = {}
        shape_valid, shape_reason = False, "malformed_json"
    else:
        if not isinstance(body, Mapping):
            body = {}
            shape_valid, shape_reason = False, "request_not_an_object"
        else:
            shape_valid, shape_reason = _shape_only_typed_request(body, case.sdk_family)
    global_images, global_references = _inspect_inline_evidence(body)
    global_task_id, _ = _request_metadata(body)
    collection_key = "messages" if case.sdk_family == "openai" else "contents"
    content_key = "content" if case.sdk_family == "openai" else "parts"
    turns = body.get(collection_key)
    turns = turns if isinstance(turns, list) else []
    user_indexes = [
        index
        for index, turn in enumerate(turns)
        if isinstance(turn, Mapping) and turn.get("role") == "user"
    ]
    if user_indexes:
        active = turns[user_indexes[-1]].get(content_key)
        active_images, active_references = _inspect_inline_evidence(active)
        active_task_id, _ = _request_metadata(active)
        active_task_text = _canonical_task_text(active)
    else:
        active_images, active_references = (), ()
        active_task_id, active_task_text = "", ""
    return RequestView(
        active_images=active_images,
        active_references=active_references,
        active_task_id=active_task_id,
        active_task_text=active_task_text,
        body=body,
        global_images=global_images,
        global_references=global_references,
        global_task_id=global_task_id,
        out_of_scope_image_count=len(global_images) - len(active_images),
        response_schema=_response_schema(body, case.sdk_family),
        shape_reason=shape_reason,
        shape_valid=shape_valid,
    )


def response_schema_only(_case: CorpusCase, _view: RequestView) -> Decision:
    return True, "response_schema_has_no_request_blocking_hook"


def shape_only_typed_builder(_case: CorpusCase, view: RequestView) -> Decision:
    return view.shape_valid, view.shape_reason


def _all_inline_images_decode(images: tuple[InlineImage, ...]) -> bool:
    return all(
        image.base64_valid
        and image.declared_mime_type in SUPPORTED_MIMES
        and image.decoded_mime_type == image.declared_mime_type
        for image in images
    )


def independent_decoded_input_validator(
    _case: CorpusCase, view: RequestView
) -> Decision:
    if not view.shape_valid:
        return False, view.shape_reason
    if not _all_inline_images_decode(view.global_images):
        return False, "provided_inline_image_failed_decode_or_mime_check"
    return True, "all_provided_inline_images_valid_or_none_provided"


def released_decoded_presence_gate(_case: CorpusCase, view: RequestView) -> Decision:
    if view.global_references or not view.global_images:
        return False, "no_materialized_image"
    if not _all_inline_images_decode(view.global_images):
        return False, "nondecodable_or_mime_mismatch"
    return True, "decoded_presence_satisfied"


def released_body_only_gate(case: CorpusCase, view: RequestView) -> Decision:
    if view.global_references:
        return False, "unresolved_reference"
    expected = case.registry.get(view.global_task_id)
    if expected is None:
        return False, "body_task_missing"
    if len(view.global_images) != len(expected.images):
        return False, "body_image_count_mismatch"
    if not _all_inline_images_decode(view.global_images):
        return False, "nondecodable_or_mime_mismatch"
    observed = tuple(
        (image.declared_mime_type, sha256_bytes(image.data))
        for image in view.global_images
    )
    wanted = tuple((image.mime_type, image.sha256) for image in expected.images)
    if observed != wanted:
        return False, "body_task_content_mismatch"
    return True, "body_self_consistency_satisfied"


def _serialized_response_schema(
    schema: Mapping[str, Any] | None, sdk_family: str
) -> dict[str, Any]:
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    required = schema.get("required") if isinstance(schema, Mapping) else None
    forbidden = (
        sorted(set(properties).intersection(BINDING_FIELDS))
        if isinstance(properties, Mapping)
        else []
    )
    provider_properties = {
        "primary_diagnosis": {"type": ["string", "null"]},
        "differentials": {"type": "array", "items": {"type": "string"}},
        "clinical_assessment": {"type": "string"},
    }
    return {
        "additional_properties": (
            schema.get("additionalProperties") if isinstance(schema, Mapping) else None
        ),
        "forbidden_binding_fields": forbidden,
        "path": (
            "$.response_format.json_schema.schema"
            if sdk_family == "openai"
            else "$.generationConfig.responseJsonSchema"
        ),
        "present": isinstance(schema, Mapping),
        "properties": sorted(properties) if isinstance(properties, Mapping) else [],
        "required": sorted(required) if isinstance(required, list) else [],
        "valid": bool(
            isinstance(schema, Mapping)
            and schema.get("type") == "object"
            and dict(properties) == provider_properties
            and isinstance(required, list)
            and set(required) == set(RESPONSE_FIELDS)
            and len(required) == len(RESPONSE_FIELDS)
            and schema.get("additionalProperties") is False
            and not forbidden
        ),
    }


def caller_owned_binding(case: CorpusCase, view: RequestView) -> Decision:
    expected = case.expected
    if view.active_task_id != expected.task_id:
        if view.global_task_id == expected.task_id:
            return False, "task_scope_mismatch"
        return False, "task_id_mismatch"
    if view.active_task_text != expected.task_text:
        return False, "task_text_mismatch"
    if view.active_references:
        return False, "unresolved_uri"
    if view.out_of_scope_image_count:
        return False, "evidence_scope_mismatch"
    if len(view.active_images) > len(expected.images):
        return False, "unexpected_extra_image"
    if len(view.active_images) < len(expected.images):
        return False, "missing_image"
    if not _all_inline_images_decode(view.active_images):
        corrupt = any(
            not image.base64_valid or image.decoded_mime_type is None
            for image in view.active_images
        )
        return False, "corrupt_image" if corrupt else "mime_mismatch"
    observed_mimes = tuple(image.declared_mime_type for image in view.active_images)
    wanted_mimes = tuple(image.mime_type for image in expected.images)
    if observed_mimes != wanted_mimes:
        return False, "mime_mismatch"
    observed_hashes = tuple(sha256_bytes(image.data) for image in view.active_images)
    wanted_hashes = tuple(image.sha256 for image in expected.images)
    if observed_hashes != wanted_hashes:
        if len(observed_hashes) > 1 and sorted(observed_hashes) == sorted(wanted_hashes):
            return False, "wrong_image_order"
        return False, "digest_mismatch"
    if dict(expected.client_response_binding) != dict(
        expected.expected_response_binding
    ):
        return False, "client_response_binding_mismatch"
    if expected.response_schema_required:
        observed_schema = _serialized_response_schema(
            view.response_schema, case.sdk_family
        )
        if (
            not observed_schema["valid"]
            or observed_schema != expected.serialized_response_schema
        ):
            return False, "response_contract_mismatch"
    return True, "caller_owned_binding_satisfied"


def _integrated_expected(case: CorpusCase) -> dict[str, Any]:
    expected = case.expected
    return {
        "client_response_binding": dict(expected.client_response_binding),
        "expected_response_binding": dict(expected.expected_response_binding),
        "images": [
            {"mime_type": image.mime_type, "sha256": image.sha256}
            for image in expected.images
        ],
        "response_schema_required": expected.response_schema_required,
        "serialized_response_schema": expected.serialized_response_schema,
    }


def draft202012_integrated_envelope(
    case: CorpusCase, _view: RequestView
) -> Decision:
    return integrated_envelope.evaluate_request(
        case.body,
        case.sdk_family,
        _integrated_expected(case),
    )


def pydantic_integrated_envelope(
    case: CorpusCase, _view: RequestView
) -> Decision:
    return pydantic_envelope.evaluate_request(
        case.body,
        case.sdk_family,
        _integrated_expected(case),
    )


BASELINES: tuple[tuple[str, DecisionFunction], ...] = (
    ("response_schema_only", response_schema_only),
    ("shape_only_typed_builder", shape_only_typed_builder),
    ("independent_decoded_input_validator", independent_decoded_input_validator),
    ("released_decoded_presence_gate", released_decoded_presence_gate),
    ("released_body_only_gate", released_body_only_gate),
    ("draft202012_integrated_envelope", draft202012_integrated_envelope),
    ("pydantic_integrated_envelope", pydantic_integrated_envelope),
    ("caller_owned_binding", caller_owned_binding),
)

BASELINE_DEFINITIONS = {
    "response_schema_only": {
        "request_blocking_capability": False,
        "specification": (
            "Output validation runs only after dispatch. It never inspects or blocks a "
            "request; conditional response behavior is tested in a separate local fixture audit."
        ),
    },
    "shape_only_typed_builder": {
        "request_blocking_capability": True,
        "specification": (
            "Accept exactly route-shaped turn, text, inline-image, and reference-image "
            "containers with string-valued fields. Missing evidence, remote references, "
            "arbitrary base64 payload content, task placement, task binding, and schema "
            "enum values are outside this builder's checks."
        ),
    },
    "independent_decoded_input_validator": {
        "request_blocking_capability": True,
        "specification": (
            "Independently parse every provided inline image with strict base64 and Pillow, "
            "and require declared MIME to equal decoded MIME. Text-only requests and remote "
            "references remain valid generic request forms; no task or caller binding is checked."
        ),
    },
    "released_decoded_presence_gate": {
        "request_blocking_capability": True,
        "specification": (
            "Released whole-body presence semantics: require at least one materialized inline "
            "image, reject any unresolved reference, and require every inline image to decode "
            "with declared/decoded MIME agreement."
        ),
    },
    "released_body_only_gate": {
        "request_blocking_capability": True,
        "specification": (
            "Released whole-body self-consistency semantics: select a registry entry from the "
            "task marker serialized in the body, then require exact image count, order, MIME, "
            "and digest. It does not compare against the caller's selected task."
        ),
    },
    "draft202012_integrated_envelope": {
        "request_blocking_capability": True,
        "specification": (
            "An independent adapter projects the exact request body, caller sidecar, "
            "response-contract metadata, and parser selection into a closed envelope. "
            "The pinned jsonschema Draft 2020-12 engine validates per-contract const and "
            "ordered-evidence constraints before dispatch. JSON Schema does not parse SDK "
            "dialects, decode images, or calculate digests; the adapter performs those steps."
        ),
    },
    "pydantic_integrated_envelope": {
        "request_blocking_capability": True,
        "specification": (
            "A second standalone SDK-body adapter materializes the same closed request "
            "envelope and validates it with pinned Pydantic v2 models and model validators. "
            "It imports neither the released guard/oracle nor the Draft 2020-12 adapter."
        ),
    },
    "caller_owned_binding": {
        "request_blocking_capability": True,
        "specification": (
            "Compare the last user turn against a caller-owned sidecar: exact task ID/text, "
            "active-turn-only materialized evidence, ordered MIME/digests, and, where present "
            "in the frozen corpus, response-schema task/contract/version/field/correlation enums."
        ),
    },
}


def _evaluate_stratum(cases: list[CorpusCase]) -> dict[str, Any]:
    views = {case.audit_case_id: _request_view(case) for case in cases}
    result: dict[str, Any] = {}
    for baseline_name, decision_function in BASELINES:
        rows = [
            (case, *decision_function(case, views[case.audit_case_id]))
            for case in cases
        ]
        forwarded_controls = sorted(
            case.audit_case_id for case, allowed, _ in rows if case.is_control and allowed
        )
        blocked_controls = sorted(
            case.audit_case_id
            for case, allowed, _ in rows
            if case.is_control and not allowed
        )
        blocked_violations = sorted(
            case.audit_case_id
            for case, allowed, _ in rows
            if not case.is_control and not allowed
        )
        forwarded_violations = sorted(
            case.audit_case_id
            for case, allowed, _ in rows
            if not case.is_control and allowed
        )
        result[baseline_name] = {
            "blocked_control_case_ids": blocked_controls,
            "blocked_controls": len(blocked_controls),
            "blocked_reason_counts": dict(
                sorted(Counter(reason for _, allowed, reason in rows if not allowed).items())
            ),
            "blocked_violation_case_ids": blocked_violations,
            "controls_forwarded": len(forwarded_controls),
            "controls_total": sum(case.is_control for case in cases),
            "forwarded_control_case_ids": forwarded_controls,
            "forwarded_violation_case_ids": forwarded_violations,
            "request_cases_scored": len(rows),
            "violations_blocked": len(blocked_violations),
            "violations_total": sum(not case.is_control for case in cases),
        }
    return result


def _body_manifest(cases: list[CorpusCase]) -> dict[str, Any]:
    rows = [
        {
            "audit_case_id": case.audit_case_id,
            "body_sha256": sha256_bytes(case.body),
            "is_control": case.is_control,
            "sdk_family": case.sdk_family,
            "source_case_id": case.source_case_id,
        }
        for case in sorted(cases, key=lambda item: item.audit_case_id)
    ]
    return {
        "case_to_body_manifest_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "controls": sum(case.is_control for case in cases),
        "reconstructed_bodies": len(cases),
        "unique_body_sha256": len({sha256_bytes(case.body) for case in cases}),
        "violations": sum(not case.is_control for case in cases),
    }


def _conditional_response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "image_present": {"type": "boolean"},
            "can_diagnose": {"type": "boolean"},
            "primary_diagnosis": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "string", "minLength": 1},
                ]
            },
            "differentials": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "image_present",
            "can_diagnose",
            "primary_diagnosis",
            "differentials",
        ],
        "if": {
            "properties": {"image_present": {"const": False}},
            "required": ["image_present"],
        },
        "then": {
            "properties": {
                "can_diagnose": {"const": False},
                "primary_diagnosis": {"type": "null"},
                "differentials": {"maxItems": 0},
            }
        },
        "additionalProperties": False,
    }


def _local_response_predicate(value: Any) -> bool:
    required = {
        "image_present",
        "can_diagnose",
        "primary_diagnosis",
        "differentials",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        return False
    image_present = value.get("image_present")
    can_diagnose = value.get("can_diagnose")
    primary = value.get("primary_diagnosis")
    differentials = value.get("differentials")
    if not isinstance(image_present, bool) or not isinstance(can_diagnose, bool):
        return False
    if primary is not None and (not isinstance(primary, str) or not primary):
        return False
    if not isinstance(differentials, list) or any(
        not isinstance(item, str) for item in differentials
    ):
        return False
    if not image_present:
        if can_diagnose or primary is not None or differentials:
            return False
    return True


def response_fixture_audit() -> dict[str, Any]:
    fixtures = {
        "absent-null-valid": (
            {
                "image_present": False,
                "can_diagnose": False,
                "primary_diagnosis": None,
                "differentials": [],
            },
            True,
        ),
        "absent-can-diagnose-true": (
            {
                "image_present": False,
                "can_diagnose": True,
                "primary_diagnosis": None,
                "differentials": [],
            },
            False,
        ),
        "absent-diagnosis-populated": (
            {
                "image_present": False,
                "can_diagnose": False,
                "primary_diagnosis": "A",
                "differentials": [],
            },
            False,
        ),
        "absent-differentials-populated": (
            {
                "image_present": False,
                "can_diagnose": False,
                "primary_diagnosis": None,
                "differentials": ["A"],
            },
            False,
        ),
        "extra-property": (
            {
                "image_present": True,
                "can_diagnose": True,
                "primary_diagnosis": "A",
                "differentials": [],
                "extra": True,
            },
            False,
        ),
        "present-diagnosis-valid": (
            {
                "image_present": True,
                "can_diagnose": True,
                "primary_diagnosis": "A",
                "differentials": [],
            },
            True,
        ),
        "self-asserted-present-valid": (
            {
                "image_present": True,
                "can_diagnose": True,
                "primary_diagnosis": "A",
                "differentials": ["B"],
            },
            True,
        ),
    }
    schema = _conditional_response_schema()
    try:
        jsonschema = importlib.import_module("jsonschema")
        validator_class = jsonschema.Draft202012Validator
    except (ImportError, AttributeError):
        validate = _local_response_predicate
        engine = "local conditional-response predicate (not standards validation)"
        standards_validation = False
        dependency_version = None
    else:
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        validate = validator.is_valid
        engine = "jsonschema.Draft202012Validator"
        standards_validation = True
        dependency_version = importlib.metadata.version("jsonschema")
    rows = [
        {
            "accepted": bool(validate(value)),
            "expected_valid": expected,
            "fixture_id": fixture_id,
        }
        for fixture_id, (value, expected) in sorted(fixtures.items())
    ]
    _require(
        all(row["accepted"] == row["expected_valid"] for row in rows),
        "conditional response fixture validation drift",
    )
    return {
        "accepted_fixture_ids": [row["fixture_id"] for row in rows if row["accepted"]],
        "engine": engine,
        "fixture_agreements": len(rows),
        "fixture_count": len(rows),
        "jsonschema_version": dependency_version,
        "rejected_fixture_ids": [row["fixture_id"] for row in rows if not row["accepted"]],
        "request_binding_probe": "self-asserted-present-valid",
        "request_binding_probe_accepted": next(
            row["accepted"]
            for row in rows
            if row["fixture_id"] == "self-asserted-present-valid"
        ),
        "request_cases_blocked": 0,
        "request_scoring_separate": True,
        "schema_dialect": (
            "JSON Schema Draft 2020-12" if standards_validation else None
        ),
        "standards_validation": standards_validation,
    }


def _stratum_report(cases: list[CorpusCase]) -> dict[str, Any]:
    return {
        "baselines": _evaluate_stratum(cases),
        "corpus_verification": _body_manifest(cases),
        "denominators": {
            "controls": sum(case.is_control for case in cases),
            "total": len(cases),
            "violations": sum(not case.is_control for case in cases),
        },
    }


def run() -> dict[str, Any]:
    verified_files = verify_release_inputs()
    frozen_matrix = _load_json("data/evidence_binding_matrix_audit.json")
    frozen_generated = _load_json("data/evidence_mutation_property_audit.json")
    frozen_compositional = _load_json("data/compositional_request_audit.json")
    matrix, generated, compositional = _load_corpus_generators()
    matrix_cases = _load_frozen_matrix(matrix, frozen_matrix)
    generated_cases, spec, contracts = _load_generated_exact_body(
        generated, frozen_generated
    )
    compositional_cases = _load_compositional(
        compositional,
        frozen_compositional,
        spec,
        contracts,
    )
    strata = {
        "compositional": _stratum_report(compositional_cases),
        "frozen_matrix": _stratum_report(matrix_cases),
        "generated_exact_body": _stratum_report(generated_cases),
    }
    for stratum, report in strata.items():
        _require(
            report["denominators"] == EXPECTED_DENOMINATORS[stratum],
            f"post-decision denominator drift for {stratum}",
        )
    return {
        "audit_name": "offline frozen-corpus enforcement baseline comparison",
        "audit_schema_version": 3,
        "baseline_definitions": BASELINE_DEFINITIONS,
        "claim_limits": [
            "Results are limited to the three local frozen case corpora and their labels.",
            "No provider endpoint or model is called; SDK serialization uses local mock transports.",
            "The comparison does not establish general superiority over all response schemas, typed builders, or decoded-input validators.",
            "The integrated Draft 2020-12 result is adapter-plus-validator equivalence on the author-defined corpus, not a production-gateway evaluation.",
            "The Pydantic v2 result is a second local adapter-plus-validator equivalence test, not an independently deployed client evaluation.",
        ],
        "dependencies": {
            "google_genai": genai.__version__,
            "httpx": httpx.__version__,
            "jsonschema": importlib.metadata.version("jsonschema"),
            "openai": openai.__version__,
            "pydantic": pydantic.__version__,
            "pillow": PIL.__version__,
        },
        "execution": {
            "credentials_required": False,
            "network_calls": 0,
            "predicted_counts_used": False,
            "provider_or_model_calls": 0,
        },
        "release_file_commitments": verified_files,
        "response_schema_fixture_audit": response_fixture_audit(),
        "strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        parser.error(f"output directory does not exist: {args.out.parent}")
    result = run()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        stratum: {
            name: {
                "controls_forwarded": values["controls_forwarded"],
                "violations_blocked": values["violations_blocked"],
            }
            for name, values in report["baselines"].items()
        }
        for stratum, report in result["strata"].items()
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
