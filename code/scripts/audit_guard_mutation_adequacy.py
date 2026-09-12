#!/usr/bin/env python3
"""Measure decision-level mutation adequacy of the released evidence guard.

The released guard has no source-mutation hook. This audit therefore records exact
offline request bodies while the released corpus runners execute, then evaluates a
fixed catalog of first-order semantic omission mutants in a parameterized
replica of the released decision pipeline. The replica delegates request inspection,
image decoding, response-schema inspection, and contract data to released code and
must agree with the released guard on every frozen case before mutants are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import httpx


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
DATA_ROOT = ARTIFACT_ROOT / "data"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for import_root in (CODE_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import audit_compositional_requests as compositional
import audit_evidence_binding_matrix as matrix
import audit_evidence_mutation_properties as generated
from evidence_contract import inspect_serialized_response_schema


OUTPUT_PATH = DATA_ROOT / "guard_mutation_adequacy_audit.json"
FROZEN_PATHS = {
    "frozen_matrix": DATA_ROOT / "evidence_binding_matrix_audit.json",
    "frozen_generated": DATA_ROOT / "evidence_mutation_property_audit.json",
    "frozen_compositional": DATA_ROOT / "compositional_request_audit.json",
}
RELEASED_GUARD = matrix.GuardedTransport


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    category: str
    omitted_semantics: str
    equivalence_status: str = "non_equivalent"
    equivalence_basis: str | None = None

    @property
    def is_equivalent(self) -> bool:
        return self.equivalence_status == "decision_signature_equivalent"


# Operators and equivalence classifications are fixed before corpus scoring in this
# audit. The catalog was not externally preregistered.
MUTANT_CATALOG = (
    MutantSpec(
        "omit_active_turn_task_scope",
        "active_turn_scope",
        "Accept a matching task marker and canonical task text from outside the active user turn.",
    ),
    MutantSpec(
        "omit_active_turn_image_scope",
        "active_turn_scope",
        "Accept materialized images from the whole request instead of the active user turn.",
    ),
    MutantSpec(
        "omit_task_text_binding",
        "task_text",
        "Do not compare canonical active-turn task text with the caller commitment.",
    ),
    MutantSpec(
        "omit_task_id_binding",
        "task_id",
        "Do not compare the active-turn task ID with the caller-owned task ID.",
    ),
    MutantSpec(
        "omit_reference_materialization",
        "reference_materialization",
        "Allow unresolved image references when otherwise valid inline evidence is present.",
    ),
    MutantSpec(
        "omit_image_count_binding",
        "image_count",
        "Compare overlapping image positions without requiring equal image counts.",
    ),
    MutantSpec(
        "omit_image_order_binding",
        "image_order",
        "Treat an equal image-digest multiset as satisfying the ordered digest commitment.",
    ),
    MutantSpec(
        "omit_base64_validation",
        "base64",
        "Remove the explicit strict-base64 validity branch.",
        "decision_signature_equivalent",
        (
            "The released decoder materializes every invalid base64 payload as empty bytes; "
            "the retained decode/MIME branches still block it, so only reason selection can change."
        ),
    ),
    MutantSpec(
        "omit_image_decode_validation",
        "decode",
        "Remove the explicit undecodable-image branch.",
        "decision_signature_equivalent",
        (
            "An undecodable image has decoded MIME None and is still blocked by the retained "
            "declared-versus-decoded MIME comparison, so only reason selection can change."
        ),
    ),
    MutantSpec(
        "omit_declared_mime_binding",
        "mime",
        "Do not compare the serialized declared MIME type with the decoded image MIME type.",
    ),
    MutantSpec(
        "omit_contract_mime_binding",
        "mime",
        "Do not compare decoded MIME types with the ordered caller contract MIME sequence.",
    ),
    MutantSpec(
        "omit_image_digest_binding",
        "digest",
        "Do not require observed image digests to match the caller-owned digest commitment.",
    ),
    MutantSpec(
        "omit_task_ownership_check",
        "task_ownership",
        "Remove the explicit known-other-task image ownership branch.",
        "decision_signature_equivalent",
        (
            "Known-other-task evidence still fails the retained expected digest comparison; "
            "the ownership branch only selects a more specific block reason."
        ),
    ),
    MutantSpec(
        "omit_serialized_response_schema_presence",
        "serialized_response_schema",
        "Allow dispatch when no serialized response schema is present.",
    ),
    MutantSpec(
        "omit_serialized_response_schema_shape_contract",
        "serialized_response_schema",
        "Do not require the released response-schema shape and required-field contract.",
    ),
    MutantSpec(
        "omit_client_response_binding_presence",
        "client_response_binding",
        "Allow dispatch when the caller-owned response-binding sidecar is absent.",
    ),
    MutantSpec(
        "omit_client_response_binding_shape_contract",
        "client_response_binding",
        "Do not require the caller-owned response-binding sidecar to have its exact field set.",
    ),
    MutantSpec(
        "omit_client_response_binding_task_id_binding",
        "client_response_binding",
        "Do not bind the sidecar task ID to the caller contract.",
    ),
    MutantSpec(
        "omit_client_response_binding_task_text_binding",
        "client_response_binding",
        "Do not bind the sidecar task-text digest to the caller contract.",
    ),
    MutantSpec(
        "omit_client_response_binding_contract_digest_binding",
        "client_response_binding",
        "Do not bind the sidecar contract digest to the caller contract.",
    ),
    MutantSpec(
        "omit_client_response_binding_version_binding",
        "client_response_binding",
        "Do not bind the sidecar contract version to the caller contract.",
    ),
    MutantSpec(
        "omit_client_response_binding_active_field_binding",
        "client_response_binding",
        "Do not bind the sidecar active field to the caller contract.",
    ),
    MutantSpec(
        "omit_client_response_binding_correlation_binding",
        "client_response_binding",
        "Do not bind the sidecar correlation ID to the dispatch correlation ID.",
    ),
    MutantSpec(
        "omit_downstream_suppression",
        "downstream_suppression",
        "Invoke the local downstream spy for blocked as well as forwarded requests.",
    ),
)


WHITE_BOX_RESPONSE_SCHEMA_VARIANTS = (
    "missing_serialized_response_schema",
    "invalid_serialized_response_schema",
    "missing_client_response_binding",
    "invalid_client_response_binding",
    "client_response_binding_task_id_mismatch",
    "client_response_binding_task_text_mismatch",
    "client_response_binding_contract_digest_mismatch",
    "client_response_binding_version_mismatch",
    "client_response_binding_active_field_mismatch",
    "client_response_binding_correlation_mismatch",
)


@dataclass(frozen=True)
class CapturedCall:
    route: str
    sdk_family: str
    raw_body: bytes
    method: str
    url: str
    expectation: Any
    task_registry: Mapping[str, Any]
    response_schema_required: bool
    serialized_response_schema: Mapping[str, Any] | None
    client_response_binding: Mapping[str, Any] | None
    baseline_record: Mapping[str, Any]


class CaptureSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[CapturedCall] = []

    def append(self, call: CapturedCall) -> None:
        with self._lock:
            self.calls.append(call)


def recording_guard_class(sink: CaptureSink) -> type[RELEASED_GUARD]:
    class RecordingGuard(RELEASED_GUARD):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            response = super().__call__(request)
            sink.append(
                CapturedCall(
                    route=self.route,
                    sdk_family=self.sdk_family,
                    raw_body=bytes(request.content),
                    method=request.method,
                    url=str(request.url),
                    expectation=self.expectation,
                    task_registry=self.task_registry,
                    response_schema_required=self.response_schema_inspector is not None,
                    serialized_response_schema=self.captured[0].get(
                        "serialized_response_schema"
                    ),
                    client_response_binding=self.captured[0].get(
                        "client_response_binding"
                    ),
                    baseline_record=dict(self.captured[0]),
                )
            )
            return response

    return RecordingGuard


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    adequacy_set: str
    sdk_family: str
    raw_body: bytes
    method: str
    url: str
    expectation: Any
    task_registry: Mapping[str, Any]
    response_schema_required: bool
    expected_decision: str
    expected_downstream_invocations: int
    expected_reason: str


@dataclass(frozen=True)
class ImageObservation:
    base64_valid: bool
    data: bytes
    declared_mime_type: str
    decoded_mime_type: str | None
    digest: str


@dataclass(frozen=True)
class PreparedCase:
    source: AuditCase
    global_images: tuple[ImageObservation, ...]
    active_images: tuple[ImageObservation, ...]
    global_references: tuple[str, ...]
    active_references: tuple[str, ...]
    global_task_id: str
    global_task_text: str
    scoped_task_id: str
    scoped_task_text: str
    out_of_scope_image_count: int
    expected_image_digests: tuple[str, ...]
    expected_image_mime_types: tuple[str, ...]
    owner_lookup: Mapping[tuple[str, ...], frozenset[str]]
    serialized_response_schema: Mapping[str, Any] | None
    client_response_binding: Mapping[str, Any] | None


@dataclass(frozen=True)
class PipelineOutcome:
    decision: str
    downstream_invocations: int
    diagnostic_reason: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: observed {actual!r}, frozen {expected!r}")


def _unique_call_map(
    calls: Sequence[CapturedCall], key_fn: Any
) -> dict[Any, CapturedCall]:
    result: dict[Any, CapturedCall] = {}
    for call in calls:
        key = key_fn(call)
        prior = result.get(key)
        if prior is None:
            result[key] = call
            continue
        _assert_equal(f"duplicate body for {key}", prior.raw_body, call.raw_body)
        _assert_equal(
            f"duplicate decision for {key}",
            prior.baseline_record["decision"],
            call.baseline_record["decision"],
        )
    return result


def collect_frozen_matrix() -> list[AuditCase]:
    frozen = load_json(FROZEN_PATHS["frozen_matrix"])
    sink = CaptureSink()
    with patch.object(matrix, "GuardedTransport", recording_guard_class(sink)):
        runtime = matrix.run()

    calls = _unique_call_map(
        sink.calls, lambda row: (row.route, row.expectation.case_id)
    )
    cases: list[AuditCase] = []
    for route, frozen_transport in sorted(frozen["transports"].items()):
        runtime_rows = {
            row["case_id"]: row
            for row in runtime["transports"][route]["decision_audit"]
        }
        for frozen_row in frozen_transport["decision_audit"]:
            case_name = frozen_row["case_id"]
            runtime_row = runtime_rows[case_name]
            for field in ("body_sha256", "decision", "downstream_invocations"):
                _assert_equal(
                    f"matrix {route}/{case_name} {field}",
                    runtime_row[field],
                    frozen_row[field],
                )
            call = calls[(route, case_name)]
            _assert_equal(
                f"matrix {route}/{case_name} captured hash",
                sha256_bytes(call.raw_body),
                frozen_row["body_sha256"],
            )
            cases.append(
                AuditCase(
                    case_id=f"matrix:{route}:{case_name}",
                    adequacy_set="frozen_matrix",
                    sdk_family=call.sdk_family,
                    raw_body=call.raw_body,
                    method=call.method,
                    url=call.url,
                    expectation=call.expectation,
                    task_registry=call.task_registry,
                    response_schema_required=False,
                    expected_decision=frozen_row["decision"],
                    expected_downstream_invocations=frozen_row[
                        "downstream_invocations"
                    ],
                    expected_reason=frozen_row["reason"],
                )
            )
    _assert_equal("frozen matrix denominator", len(cases), 85)
    return sorted(cases, key=lambda row: row.case_id)


def collect_frozen_generated() -> list[AuditCase]:
    frozen = load_json(FROZEN_PATHS["frozen_generated"])
    sink = CaptureSink()
    with patch.object(generated, "GuardedTransport", recording_guard_class(sink)):
        runtime = generated.run()

    calls = _unique_call_map(
        sink.calls,
        lambda row: (
            row.sdk_family,
            row.expectation.correlation_id,
            sha256_bytes(row.raw_body),
        ),
    )
    runtime_rows = {row["correlation_id"]: row for row in runtime["logical_audit"]}
    cases: list[AuditCase] = []
    for frozen_row in frozen["logical_audit"]:
        correlation_id = frozen_row["correlation_id"]
        runtime_row = runtime_rows[correlation_id]
        for field in ("body_sha256", "decision", "model_spy_invocations"):
            _assert_equal(
                f"generated {correlation_id} {field}",
                runtime_row[field],
                frozen_row[field],
            )
        key = (
            frozen_row["sdk_family"],
            correlation_id,
            frozen_row["body_sha256"],
        )
        call = calls[key]
        expected_invocations = 1 if frozen_row["decision"] == "forwarded" else 0
        _assert_equal(
            f"generated {correlation_id} invocation label",
            frozen_row["model_spy_invocations"],
            expected_invocations,
        )
        cases.append(
            AuditCase(
                case_id=f"generated:{correlation_id}",
                adequacy_set="frozen_generated",
                sdk_family=call.sdk_family,
                raw_body=call.raw_body,
                method=call.method,
                url=call.url,
                expectation=call.expectation,
                task_registry=call.task_registry,
                response_schema_required=True,
                expected_decision=frozen_row["decision"],
                expected_downstream_invocations=expected_invocations,
                expected_reason=frozen_row["reason"],
            )
        )
    _assert_equal("frozen generated denominator", len(cases), 960)
    return sorted(cases, key=lambda row: row.case_id)


def collect_frozen_compositional() -> list[AuditCase]:
    frozen = load_json(FROZEN_PATHS["frozen_compositional"])
    sink = CaptureSink()
    with patch.object(compositional, "GuardedTransport", recording_guard_class(sink)):
        runtime = compositional.run()

    calls = _unique_call_map(sink.calls, lambda row: sha256_bytes(row.raw_body))
    runtime_rows = {row["audit_case_id"]: row for row in runtime["logical_audit"]}
    cases: list[AuditCase] = []
    for frozen_row in frozen["logical_audit"]:
        audit_case_id = frozen_row["audit_case_id"]
        runtime_row = runtime_rows[audit_case_id]
        for field in ("body_sha256", "guard_decision", "downstream_invocations"):
            _assert_equal(
                f"compositional {audit_case_id} {field}",
                runtime_row[field],
                frozen_row[field],
            )
        call = calls[frozen_row["body_sha256"]]
        cases.append(
            AuditCase(
                case_id=f"compositional:{audit_case_id}",
                adequacy_set="frozen_compositional",
                sdk_family=call.sdk_family,
                raw_body=call.raw_body,
                method=call.method,
                url=call.url,
                expectation=call.expectation,
                task_registry=call.task_registry,
                response_schema_required=True,
                expected_decision=frozen_row["guard_decision"],
                expected_downstream_invocations=frozen_row[
                    "downstream_invocations"
                ],
                expected_reason=frozen_row["guard_reason"],
            )
        )
    _assert_equal("frozen compositional denominator", len(cases), 5120)
    return sorted(cases, key=lambda row: row.case_id)


def collect_frozen_cases() -> dict[str, list[AuditCase]]:
    return {
        "frozen_matrix": collect_frozen_matrix(),
        "frozen_generated": collect_frozen_generated(),
        "frozen_compositional": collect_frozen_compositional(),
    }


def _response_schema(body: dict[str, Any], sdk_family: str) -> dict[str, Any]:
    if sdk_family == "openai":
        return body["response_format"]["json_schema"]["schema"]
    return body["generationConfig"]["responseJsonSchema"]


def _mutate_response_schema_case(
    base: AuditCase, variant: str
) -> tuple[bytes, Any]:
    body = json.loads(base.raw_body)
    if variant == "missing_serialized_response_schema":
        if base.sdk_family == "openai":
            body.pop("response_format", None)
        else:
            body["generationConfig"].pop("responseJsonSchema", None)
        return json.dumps(body, separators=(",", ":")).encode("utf-8"), base.expectation

    contract = base.expectation.contract
    binding = base.expectation.client_response_binding
    if variant == "invalid_serialized_response_schema":
        schema = _response_schema(body, base.sdk_family)
        properties = schema["properties"]
        schema["required"].remove("differentials")
    elif variant == "missing_client_response_binding":
        binding = None
    elif variant == "invalid_client_response_binding":
        binding = {}
    elif variant == "client_response_binding_task_id_mismatch":
        binding = {**binding, "task_id": "white-box-task-drift"}
    elif variant == "client_response_binding_task_text_mismatch":
        binding = {**binding, "task_text_sha256": "0" * 64}
    elif variant == "client_response_binding_contract_digest_mismatch":
        binding = {**binding, "contract_sha256": "0" * 64}
    elif variant == "client_response_binding_version_mismatch":
        binding = {**binding, "contract_version": contract.contract_version + 1}
    elif variant == "client_response_binding_active_field_mismatch":
        binding = {**binding, "active_field": "white-box-active-field-drift"}
    elif variant == "client_response_binding_correlation_mismatch":
        binding = {**binding, "correlation_id": "white-box-correlation-drift"}
    else:
        raise ValueError(f"unknown response-schema variant: {variant}")
    return (
        json.dumps(body, separators=(",", ":")).encode("utf-8"),
        replace(base.expectation, client_response_binding=binding),
    )


def _released_white_box_outcome(
    base: AuditCase, body_bytes: bytes, expectation: Any
) -> PipelineOutcome:
    formatter = matrix.openai_response if base.sdk_family == "openai" else matrix.vertex_response
    downstream = matrix.DownstreamSpy("white-box-response-schema", formatter)
    guard = RELEASED_GUARD(
        route="white-box-response-schema",
        sdk_family=base.sdk_family,
        expectation=expectation,
        blocked_response_formatter=formatter,
        downstream=downstream,
        task_registry=base.task_registry,
        response_schema_inspector=inspect_serialized_response_schema,
    )
    guard(
        httpx.Request(
            base.method,
            base.url,
            headers={"content-type": "application/json"},
            content=body_bytes,
        )
    )
    record = guard.captured[0]
    return PipelineOutcome(
        decision=record["decision"],
        downstream_invocations=len(downstream.invocations),
        diagnostic_reason=record["reason"],
    )


def build_white_box_response_schema_cases(
    generated_cases: Sequence[AuditCase],
) -> list[AuditCase]:
    bases: dict[str, AuditCase] = {}
    for case in generated_cases:
        if case.expected_decision != "forwarded":
            continue
        bases.setdefault(case.sdk_family, case)
    _assert_equal("white-box SDK families", sorted(bases), ["google_genai", "openai"])

    cases: list[AuditCase] = []
    for sdk_family, base in sorted(bases.items()):
        for variant in WHITE_BOX_RESPONSE_SCHEMA_VARIANTS:
            body_bytes, expectation = _mutate_response_schema_case(base, variant)
            released = _released_white_box_outcome(base, body_bytes, expectation)
            _assert_equal(
                f"white-box {sdk_family}/{variant} released signature",
                (released.decision, released.downstream_invocations),
                ("blocked", 0),
            )
            cases.append(
                AuditCase(
                    case_id=f"whitebox-response-schema:{sdk_family}:{variant}",
                    adequacy_set="white_box_response_schema",
                    sdk_family=sdk_family,
                    raw_body=body_bytes,
                    method=base.method,
                    url=base.url,
                    expectation=expectation,
                    task_registry=base.task_registry,
                    response_schema_required=True,
                    expected_decision="blocked",
                    expected_downstream_invocations=0,
                    expected_reason=released.diagnostic_reason,
                )
            )
    _assert_equal("white-box response-schema denominator", len(cases), 20)
    return sorted(cases, key=lambda row: row.case_id)


def _image_observation(
    row: Mapping[str, Any], decode_cache: dict[bytes, str | None]
) -> ImageObservation:
    data = row["data"]
    if data not in decode_cache:
        decode_cache[data] = matrix.decoded_mime_type(data)
    return ImageObservation(
        base64_valid=bool(row["base64_valid"]),
        data=data,
        declared_mime_type=str(row["declared_mime_type"]),
        decoded_mime_type=decode_cache[data],
        digest=sha256_bytes(data),
    )


def prepare_cases(cases: Sequence[AuditCase]) -> list[PreparedCase]:
    decode_cache: dict[bytes, str | None] = {}
    owner_cache: dict[int, Mapping[tuple[str, ...], frozenset[str]]] = {}
    prepared: list[PreparedCase] = []
    for case in cases:
        body = json.loads(case.raw_body)
        global_images_raw, global_references = matrix.inspect_inline_evidence(body)
        global_task_id, _ = matrix.request_metadata(body)
        scope = matrix.active_user_scope(body, case.sdk_family)
        contract = case.expectation.contract
        registry_key = id(case.task_registry)
        if registry_key not in owner_cache:
            owner_cache[registry_key] = {
                sequence: frozenset(owners)
                for sequence, owners in matrix.task_digest_sequences(
                    case.task_registry
                ).items()
            }
        serialized_response_schema = None
        if case.response_schema_required:
            serialized_response_schema = inspect_serialized_response_schema(
                body, case.sdk_family
            )
        prepared.append(
            PreparedCase(
                source=case,
                global_images=tuple(
                    _image_observation(row, decode_cache)
                    for row in global_images_raw
                ),
                active_images=tuple(
                    _image_observation(row, decode_cache) for row in scope["images"]
                ),
                global_references=tuple(global_references),
                active_references=tuple(scope["references"]),
                global_task_id=global_task_id,
                global_task_text=matrix.canonical_task_text(body),
                scoped_task_id=scope["scoped_task_id"],
                scoped_task_text=scope["scoped_task_text"],
                out_of_scope_image_count=len(global_images_raw) - len(scope["images"]),
                expected_image_digests=tuple(
                    sha256_bytes(image.data) for image in contract.images
                ),
                expected_image_mime_types=tuple(
                    image.mime_type for image in contract.images
                ),
                owner_lookup=owner_cache[registry_key],
                serialized_response_schema=serialized_response_schema,
                client_response_binding=case.expectation.client_response_binding,
            )
        )
    return prepared


def _evidence_decision(case: PreparedCase, omitted: str | None) -> tuple[str, str]:
    contract = case.source.expectation.contract
    task_id = case.scoped_task_id
    task_text = case.scoped_task_text
    if (
        omitted == "omit_active_turn_task_scope"
        and task_id != contract.task_id
        and case.global_task_id == contract.task_id
    ):
        task_id = case.global_task_id
        task_text = case.global_task_text

    images = case.active_images
    references = case.active_references
    out_of_scope = case.out_of_scope_image_count
    if omitted == "omit_active_turn_image_scope" and out_of_scope:
        images = case.global_images
        references = case.global_references
        out_of_scope = 0

    if task_id != contract.task_id and omitted != "omit_task_id_binding":
        return "blocked", "task_id_or_scope_mismatch"
    if task_text != contract.task_text and omitted != "omit_task_text_binding":
        return "blocked", "task_text_mismatch"
    if references and omitted != "omit_reference_materialization":
        return "blocked", "unresolved_reference"
    if out_of_scope:
        return "blocked", "evidence_scope_mismatch"

    count_matches = len(images) == len(case.expected_image_digests)
    if not count_matches and omitted != "omit_image_count_binding":
        return "blocked", "image_count_mismatch"

    for image in images:
        if not image.base64_valid and omitted != "omit_base64_validation":
            return "blocked", "invalid_base64"
        if image.decoded_mime_type is None and omitted != "omit_image_decode_validation":
            return "blocked", "undecodable_image"
        if (
            image.declared_mime_type != image.decoded_mime_type
            and omitted != "omit_declared_mime_binding"
        ):
            return "blocked", "declared_mime_mismatch"

    if omitted != "omit_contract_mime_binding":
        for image, expected_mime in zip(
            images, case.expected_image_mime_types, strict=False
        ):
            if image.decoded_mime_type != expected_mime:
                return "blocked", "contract_mime_mismatch"

    observed_digests = tuple(image.digest for image in images)
    same_digest_multiset = (
        len(observed_digests) == len(case.expected_image_digests)
        and sorted(observed_digests) == sorted(case.expected_image_digests)
    )
    if (
        same_digest_multiset
        and observed_digests != case.expected_image_digests
        and omitted != "omit_image_order_binding"
    ):
        return "blocked", "image_order_mismatch"

    if omitted == "omit_image_order_binding" and same_digest_multiset:
        digest_matches = True
    elif omitted == "omit_image_count_binding":
        digest_matches = all(
            observed == expected
            for observed, expected in zip(
                observed_digests, case.expected_image_digests, strict=False
            )
        )
    else:
        digest_matches = observed_digests == case.expected_image_digests

    owners = case.owner_lookup.get(observed_digests, frozenset())
    if (
        owners
        and contract.task_id not in owners
        and omitted != "omit_task_ownership_check"
    ):
        return "blocked", "wrong_task_owner"
    if not digest_matches and omitted != "omit_image_digest_binding":
        return "blocked", "image_digest_mismatch"
    return "forwarded", "evidence_binding_satisfied"


def evaluate_pipeline(case: PreparedCase, mutant_id: str | None = None) -> PipelineOutcome:
    decision, reason = _evidence_decision(case, mutant_id)
    contract = case.source.expectation.contract
    serialized = case.serialized_response_schema
    if case.source.response_schema_required:
        if not serialized or not serialized["present"]:
            if mutant_id != "omit_serialized_response_schema_presence":
                decision, reason = "blocked", "response_schema_missing"
        else:
            if (
                not serialized["valid"]
                and mutant_id != "omit_serialized_response_schema_shape_contract"
            ):
                decision, reason = "blocked", "response_schema_invalid"
            else:
                binding = case.client_response_binding
                expected_binding = {
                    "task_id": contract.task_id,
                    "task_text_sha256": sha256_bytes(
                        contract.task_text.encode("utf-8")
                    ),
                    "contract_sha256": contract.sha256,
                    "contract_version": contract.contract_version,
                    "active_field": contract.active_field,
                    "correlation_id": case.source.expectation.correlation_id,
                }
                if binding is None:
                    if mutant_id != "omit_client_response_binding_presence":
                        decision, reason = "blocked", "client_response_binding_missing"
                elif (
                    not isinstance(binding, Mapping)
                    or set(binding) != set(expected_binding)
                ):
                    if mutant_id != "omit_client_response_binding_shape_contract":
                        decision, reason = "blocked", "client_response_binding_invalid"
                elif (
                    binding["task_id"] != expected_binding["task_id"]
                    and mutant_id != "omit_client_response_binding_task_id_binding"
                ):
                    decision, reason = "blocked", "client_response_binding_mismatch"
                elif (
                    binding["task_text_sha256"]
                    != expected_binding["task_text_sha256"]
                    and mutant_id
                    != "omit_client_response_binding_task_text_binding"
                ):
                    decision, reason = "blocked", "client_response_binding_mismatch"
                elif (
                    binding["contract_sha256"] != expected_binding["contract_sha256"]
                    and mutant_id
                    != "omit_client_response_binding_contract_digest_binding"
                ):
                    decision, reason = "blocked", "client_response_binding_mismatch"
                elif (
                    binding["contract_version"] != expected_binding["contract_version"]
                    and mutant_id != "omit_client_response_binding_version_binding"
                ):
                    decision, reason = "blocked", "client_response_binding_mismatch"
                elif (
                    binding["active_field"] != expected_binding["active_field"]
                    and mutant_id
                    != "omit_client_response_binding_active_field_binding"
                ):
                    decision, reason = "blocked", "client_response_binding_mismatch"
                elif (
                    binding["correlation_id"] != expected_binding["correlation_id"]
                    and mutant_id
                    != "omit_client_response_binding_correlation_binding"
                ):
                    decision, reason = "blocked", "client_response_binding_mismatch"

    invoke_blocked = mutant_id == "omit_downstream_suppression"
    downstream_invocations = int(decision == "forwarded" or invoke_blocked)
    return PipelineOutcome(decision, downstream_invocations, reason)


def is_killed(expected: PipelineOutcome, observed: PipelineOutcome) -> bool:
    """Return a kill only for decision or downstream-invocation disagreement."""
    return (
        expected.decision,
        expected.downstream_invocations,
    ) != (
        observed.decision,
        observed.downstream_invocations,
    )


def expected_outcome(case: PreparedCase) -> PipelineOutcome:
    return PipelineOutcome(
        decision=case.source.expected_decision,
        downstream_invocations=case.source.expected_downstream_invocations,
        diagnostic_reason=case.source.expected_reason,
    )


def _source_manifest() -> dict[str, Any]:
    paths = {
        **FROZEN_PATHS,
        "released_guard": SCRIPTS_ROOT / "audit_evidence_binding_matrix.py",
        "generated_corpus_generator": SCRIPTS_ROOT
        / "audit_evidence_mutation_properties.py",
        "compositional_corpus_generator": SCRIPTS_ROOT
        / "audit_compositional_requests.py",
        "response_contract": CODE_ROOT / "evidence_contract.py",
    }
    return {
        name: {
            "path": str(path.relative_to(ARTIFACT_ROOT)),
            "sha256": sha256_bytes(path.read_bytes()),
        }
        for name, path in sorted(paths.items())
    }


def run_audit() -> dict[str, Any]:
    frozen_sets = collect_frozen_cases()
    white_box_cases = build_white_box_response_schema_cases(
        frozen_sets["frozen_generated"]
    )
    all_sets = {
        **frozen_sets,
        "white_box_response_schema": white_box_cases,
    }
    prepared_sets = {
        name: prepare_cases(cases) for name, cases in all_sets.items()
    }

    fidelity: dict[str, Any] = {}
    for set_name, cases in prepared_sets.items():
        disagreements = []
        for case in cases:
            if is_killed(expected_outcome(case), evaluate_pipeline(case)):
                disagreements.append(case.source.case_id)
        fidelity[set_name] = {
            "decision_and_downstream_agreements": len(cases) - len(disagreements),
            "denominator": len(cases),
            "disagreement_case_ids": disagreements,
        }
        if disagreements:
            raise ValueError(
                f"unmutated semantic replica disagreed in {set_name}: {disagreements[:5]}"
            )

    mutant_rows: list[dict[str, Any]] = []
    for mutant in MUTANT_CATALOG:
        set_results: dict[str, Any] = {}
        all_kills: list[str] = []
        for set_name, cases in prepared_sets.items():
            killing_ids = [
                case.source.case_id
                for case in cases
                if is_killed(
                    expected_outcome(case),
                    evaluate_pipeline(case, mutant.mutant_id),
                )
            ]
            killing_ids.sort()
            all_kills.extend(killing_ids)
            set_results[set_name] = {
                "denominator": len(cases),
                "killing_case_count": len(killing_ids),
                "killing_case_ids": killing_ids,
            }
        survived = not all_kills
        if mutant.is_equivalent and not survived:
            raise ValueError(
                "fixed decision-signature-equivalent mutant was killed: "
                f"{mutant.mutant_id}"
            )
        mutant_rows.append(
            {
                "mutant_id": mutant.mutant_id,
                "category": mutant.category,
                "omitted_semantics": mutant.omitted_semantics,
                "implementation_mode": "parameterized_semantic_omission_mutant",
                "equivalence_status": mutant.equivalence_status,
                "equivalence_basis": mutant.equivalence_basis,
                "status": "survived" if survived else "killed",
                "total_killing_cases": len(all_kills),
                "adequacy_sets": set_results,
            }
        )

    survivors = [row["mutant_id"] for row in mutant_rows if row["status"] == "survived"]
    equivalent_survivors = [
        mutant.mutant_id
        for mutant in MUTANT_CATALOG
        if mutant.is_equivalent and mutant.mutant_id in survivors
    ]
    non_equivalent_survivors = [
        mutant.mutant_id
        for mutant in MUTANT_CATALOG
        if not mutant.is_equivalent and mutant.mutant_id in survivors
    ]
    if non_equivalent_survivors:
        raise ValueError(
            "non-equivalent semantic omission mutants survived: "
            + ", ".join(non_equivalent_survivors)
        )

    frozen_case_count = sum(len(rows) for rows in frozen_sets.values())
    white_box_count = len(white_box_cases)
    total_case_count = frozen_case_count + white_box_count
    non_equivalent_count = sum(not mutant.is_equivalent for mutant in MUTANT_CATALOG)
    killed_non_equivalent = non_equivalent_count - len(non_equivalent_survivors)
    mutant_decisions = len(MUTANT_CATALOG) * total_case_count
    return {
        "audit_name": "released guard semantic omission mutation adequacy",
        "audit_schema_version": 1,
        "adequacy_sets": {
            "frozen_matrix": {
                "denominator": len(frozen_sets["frozen_matrix"]),
                "expected_labels": "data/evidence_binding_matrix_audit.json",
                "label_status": "frozen",
            },
            "frozen_generated": {
                "denominator": len(frozen_sets["frozen_generated"]),
                "expected_labels": "data/evidence_mutation_property_audit.json",
                "label_status": "frozen",
            },
            "frozen_compositional": {
                "denominator": len(frozen_sets["frozen_compositional"]),
                "expected_labels": "data/compositional_request_audit.json",
                "label_status": "frozen",
            },
            "white_box_response_schema": {
                "denominator": white_box_count,
                "expected_labels": (
                    "fixed local serialized-response-schema and "
                    "client-response-binding variants"
                ),
                "label_status": "separate_white_box_adequacy_set",
                "variants_per_sdk_family": list(WHITE_BOX_RESPONSE_SCHEMA_VARIANTS),
            },
        },
        "baseline_fidelity": fidelity,
        "kill_criterion": {
            "fields": ["forwarded_or_blocked_decision", "downstream_invocation_count"],
            "reason_text_considered": False,
            "definition": (
                "A mutant is killed only when its forwarded/blocked decision or local "
                "downstream-spy invocation count differs from the frozen expectation."
            ),
        },
        "mutation_model": {
            "catalog_frozen_before_scoring": True,
            "catalog_size": len(MUTANT_CATALOG),
            "externally_preregistered": False,
            "label": "semantic omission mutants",
            "limitation": (
                "The released GuardedTransport has no source-mutation hook. These are not "
                "AST or bytecode mutations of the released source. They are first-order, "
                "parameterized semantic omissions in a replica that delegates inspection "
                "to released helpers and is required to match every unmutated decision and "
                "downstream count before mutation scoring."
            ),
            "released_guard_edited": False,
        },
        "mutants": mutant_rows,
        "network_and_credentials": {
            "credentials_required": False,
            "model_calls": 0,
            "network_calls": 0,
            "provider_calls": 0,
            "transport": "httpx.MockTransport and direct local guard calls only",
        },
        "source_manifest": _source_manifest(),
        "summary": {
            "baseline_decisions": total_case_count,
            "equivalent_mutants": len(MUTANT_CATALOG) - non_equivalent_count,
            "equivalent_survivors": equivalent_survivors,
            "frozen_expected_decisions": frozen_case_count,
            "killed_non_equivalent_mutants": killed_non_equivalent,
            "mutant_count": len(MUTANT_CATALOG),
            "mutant_decision_evaluations": mutant_decisions,
            "mutation_score": {
                "denominator_non_equivalent_mutants": non_equivalent_count,
                "numerator_killed_non_equivalent_mutants": killed_non_equivalent,
                "ratio": killed_non_equivalent / non_equivalent_count,
            },
            "non_equivalent_mutants": non_equivalent_count,
            "non_equivalent_survivors": non_equivalent_survivors,
            "survivors": survivors,
            "total_decision_evaluations": total_case_count + mutant_decisions,
            "white_box_response_schema_decisions": white_box_count,
        },
    }


def write_report(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    report = run_audit()
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and fail if the selected output differs instead of writing it",
    )
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        parser.error(f"output directory does not exist: {args.out.parent}")
    report = run_audit()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != serialized:
            raise SystemExit(f"mutation adequacy audit is stale: {args.out}")
    else:
        args.out.write_text(serialized)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
