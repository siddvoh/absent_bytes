#!/usr/bin/env python3
"""Generate and exercise evidence-contract mutations through two SDK families."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier, Lock
from types import MappingProxyType
from typing import Any, Callable, Mapping

import httpx
import openai
import PIL
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import types
from openai import OpenAI
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evidence_contract import (
    BoundImage,
    BoundResponseParser,
    ClientOwnedResponseBinding,
    ClientOwnedResponseEnvelope,
    EvidenceFieldContract,
    ParserContractMismatch,
    RESPONSE_FIELDS,
    TrustedResponseReceipt,
    inspect_serialized_response_schema,
    mock_structured_response,
    response_json_schema,
)
from independent_contract_oracle import assess_serialized_request

from audit_evidence_binding_matrix import (
    DispatchExpectation,
    EvidencePart,
    GuardedTransport,
    JPEG_IMAGE,
    MatrixCase,
    MULTI_PNG_A,
    MULTI_PNG_B,
    OPENAI_ROUTES,
    PNG_IMAGE,
    REMOTE_PNG,
    TRANSFORMED_JPEG,
    VERTEX_MODEL,
    WEBP_IMAGE,
    openai_image_content,
    openai_response,
    sha256_bytes,
    vertex_image_parts,
    vertex_response,
)


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ARTIFACT_ROOT / "data/configs/evidence_field_contract.json"
ORACLE_PATH = ARTIFACT_ROOT / "code/independent_contract_oracle.py"
OPENAI_ROUTE = "openrouter_openai_shared_client"
GOOGLE_ROUTE = "google_vertex_sdk"
WORKERS = 32
OPENAI_CONFIG = OPENAI_ROUTES["openrouter_openai_sdk"]
RESPONSE_RECEIPT_KEY_ID = "generated-mutation-response-fixture-v1"
RESPONSE_RECEIPT_KEY = hashlib.sha256(
    b"generated mutation response fixture key v1"
).digest()
RESPONSE_RECEIPT_METADATA_FIELDS = (
    "contract_sha256",
    "correlation_id",
    "dispatch_body_sha256",
    "key_id",
    "mac_sha256",
    "provider_payload_sha256",
    "provider_response_id",
    "receipt_version",
)


@dataclass(frozen=True)
class NamedEvidence:
    name: str
    part: EvidencePart


@dataclass(frozen=True)
class GeneratedDispatch:
    correlation_id: str
    sdk_family: str
    mutation_id: str
    contract: EvidenceFieldContract
    case: MatrixCase
    response_binding: ClientOwnedResponseBinding
    is_control: bool


@dataclass
class AttemptDownstream:
    route: str
    inject_retry: bool
    dispatch: GeneratedDispatch
    response_formatter: Callable[[dict[str, Any], dict[str, Any]], httpx.Response]
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        request: httpx.Request,
        content: dict[str, Any],
        body: dict[str, Any],
        case_id: str,
    ) -> httpx.Response:
        kind = "retry_injector" if self.inject_retry else "model_spy"
        self.invocations.append(
            {
                "body_sha256": sha256_bytes(request.content),
                "case_id": case_id,
                "kind": kind,
                "route": self.route,
            }
        )
        if self.inject_retry:
            return httpx.Response(
                429,
                headers={"retry-after": "0", "x-should-retry": "true"},
                json={
                    "error": {
                        "code": "rate_limit",
                        "message": "synthetic retry",
                        "type": "rate_limit_error",
                    }
                },
            )
        response = self.response_formatter(
            mock_structured_response(self.dispatch.contract), body
        )
        response_body = response.json()
        provider_response_id = (
            f"offline-evidence-mutation::{self.dispatch.correlation_id}"
        )
        if self.dispatch.sdk_family == "openai":
            response_body["id"] = provider_response_id
        elif self.dispatch.sdk_family == "google_genai":
            response_body["responseId"] = provider_response_id
        else:
            raise ValueError(f"unsupported SDK family: {self.dispatch.sdk_family}")
        return httpx.Response(response.status_code, json=response_body)


def load_contract_spec() -> dict[str, Any]:
    spec = json.loads(CONTRACT_PATH.read_text())
    mutation_ids = [row["mutation_id"] for row in spec["mutations"]]
    if len(mutation_ids) != 14 or len(set(mutation_ids)) != 14:
        raise ValueError("the evidence contract must define 14 unique mutations")
    if spec["generator"] != {"contract_count": 32, "seed": 20260715}:
        raise ValueError("the frozen generator parameters changed")
    if tuple(spec["active_fields"]) != RESPONSE_FIELDS:
        raise ValueError("active fields must match the reference response fields")
    return spec


def fixture_pool() -> tuple[NamedEvidence, ...]:
    return (
        NamedEvidence("png-one", PNG_IMAGE),
        NamedEvidence("png-two-a", MULTI_PNG_A),
        NamedEvidence("png-two-b", MULTI_PNG_B),
        NamedEvidence("png-remote", REMOTE_PNG),
        NamedEvidence("jpeg-inline", JPEG_IMAGE),
        NamedEvidence("jpeg-transformed", TRANSFORMED_JPEG),
        NamedEvidence("webp-inline", WEBP_IMAGE),
    )


def generate_contracts(spec: dict[str, Any]) -> tuple[EvidenceFieldContract, ...]:
    selected_signatures = {
        ("image/png", "image/png"),
        ("image/png", "image/jpeg"),
        ("image/png", "image/webp"),
        ("image/jpeg", "image/png"),
    }
    candidates: list[tuple[NamedEvidence, NamedEvidence]] = []
    for left in fixture_pool():
        for right in fixture_pool():
            if sha256_bytes(left.part.data) == sha256_bytes(right.part.data):
                continue
            signature = (left.part.mime_type, right.part.mime_type)
            if signature in selected_signatures:
                candidates.append((left, right))
    if len(candidates) != spec["generator"]["contract_count"]:
        raise ValueError(f"expected 32 generated contracts, found {len(candidates)}")
    random.Random(spec["generator"]["seed"]).shuffle(candidates)
    return tuple(
        EvidenceFieldContract(
            task_id=f"generated-task-{index:02d}",
            task_text=(
                "Describe the bound image evidence for generated task "
                f"{index:02d}."
            ),
            images=(
                BoundImage(left.part.data, left.part.mime_type),
                BoundImage(right.part.data, right.part.mime_type),
            ),
            active_field=spec["active_fields"][index % len(spec["active_fields"])],
            contract_version=spec["contract_version"],
        )
        for index, (left, right) in enumerate(candidates)
    )


def replacement_image(contract_index: int, mime_type: str) -> EvidencePart:
    color = (
        (31 + contract_index * 37) % 256,
        (83 + contract_index * 53) % 256,
        (149 + contract_index * 71) % 256,
    )
    image = Image.new("RGB", (4, 4), color)
    output = io.BytesIO()
    if mime_type == "image/png":
        image.save(output, format="PNG", optimize=False)
    elif mime_type == "image/jpeg":
        image.save(
            output,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    else:
        raise ValueError(f"unsupported replacement MIME type: {mime_type}")
    return EvidencePart(output.getvalue(), mime_type)


def owner_for_wrong_task(
    contract: EvidenceFieldContract,
    contracts: tuple[EvidenceFieldContract, ...],
) -> EvidenceFieldContract:
    expected_digests = tuple(sha256_bytes(image.data) for image in contract.images)
    expected_mimes = tuple(image.mime_type for image in contract.images)
    for candidate in sorted(contracts, key=lambda item: item.task_id):
        candidate_digests = tuple(
            sha256_bytes(image.data) for image in candidate.images
        )
        candidate_mimes = tuple(image.mime_type for image in candidate.images)
        if candidate.task_id == contract.task_id or candidate_mimes != expected_mimes:
            continue
        if candidate_digests == expected_digests:
            continue
        if sorted(candidate_digests) == sorted(expected_digests):
            continue
        return candidate
    raise ValueError(f"no distinct same-MIME owner for {contract.task_id}")


def mutation_case(
    contract_index: int,
    contract: EvidenceFieldContract,
    contracts: tuple[EvidenceFieldContract, ...],
    mutation: dict[str, Any],
) -> tuple[MatrixCase, str, EvidenceFieldContract]:
    mutation_id = mutation["mutation_id"]
    case_id = f"generated-{contract_index:02d}-{mutation_id}"
    images = contract.images
    serialized_task_id = contract.task_id
    serialized_task_text = contract.task_text
    active_field = contract.active_field
    guard_contract = contract
    unresolved_uri = None
    message_layout = "active_user_turn"

    if mutation_id == "missing-image":
        images = ()
    elif mutation_id == "corrupt-bytes":
        images = (
            EvidencePart(b"deterministically-corrupt-image", images[0].mime_type),
            images[1],
        )
    elif mutation_id == "declared-mime-mismatch":
        wrong_mime = (
            "image/jpeg" if images[0].mime_type != "image/jpeg" else "image/png"
        )
        images = (EvidencePart(images[0].data, wrong_mime), images[1])
    elif mutation_id == "caller-contract-mime-substitution":
        wrong_mime = (
            "image/jpeg" if images[0].mime_type != "image/jpeg" else "image/png"
        )
        guard_contract = EvidenceFieldContract(
            task_id=contract.task_id,
            task_text=contract.task_text,
            images=(
                BoundImage(images[0].data, wrong_mime),
                *contract.images[1:],
            ),
            active_field=contract.active_field,
            contract_version=contract.contract_version,
        )
    elif mutation_id == "digest-replacement":
        replacement = replacement_image(contract_index, images[0].mime_type)
        images = (replacement, images[1])
    elif mutation_id == "wrong-task-images":
        images = owner_for_wrong_task(contract, contracts).images
    elif mutation_id == "unresolved-reference":
        images = ()
        unresolved_uri = "gs://offline-unresolved/generated-image.png"
    elif mutation_id == "extra-image":
        extra = next(
            named.part
            for named in fixture_pool()
            if sha256_bytes(named.part.data)
            not in {sha256_bytes(image.data) for image in images}
        )
        images = (*images, extra)
    elif mutation_id == "reversed-order":
        images = tuple(reversed(images))
    elif mutation_id == "substituted-task-id":
        serialized_task_id = contracts[(contract_index + 1) % len(contracts)].task_id
    elif mutation_id == "task-text-substitution":
        serialized_task_text = "Summarize only the transport metadata."
    elif mutation_id == "active-field-substitution":
        active_field = RESPONSE_FIELDS[
            (RESPONSE_FIELDS.index(contract.active_field) + 1) % len(RESPONSE_FIELDS)
        ]
    elif mutation_id == "task-outside-active-turn":
        message_layout = "task_marker_in_system_or_model_role"
    elif mutation_id == "images-in-prior-turn":
        message_layout = "image_in_prior_user_turn"
    else:
        raise ValueError(f"unknown mutation: {mutation_id}")

    return (
        MatrixCase(
            case_id=case_id,
            serialized_task_id=serialized_task_id,
            images=tuple(images),
            expected_decision="blocked",
            expected_reason=mutation["expected_reason"],
            input_flow=mutation_id,
            unresolved_uri=unresolved_uri,
            expected_task_id=contract.task_id,
            message_layout=message_layout,
            serialized_task_text=serialized_task_text,
        ),
        active_field,
        guard_contract,
    )


def build_dispatches(
    sdk_family: str,
    contracts: tuple[EvidenceFieldContract, ...],
    spec: dict[str, Any],
) -> tuple[GeneratedDispatch, ...]:
    sdk_prefix = "openai" if sdk_family == "openai" else "google"
    dispatches: list[GeneratedDispatch] = []
    for index, contract in enumerate(contracts):
        control_correlation = f"{sdk_prefix}-g{index:02d}-control"
        control = MatrixCase(
            case_id=f"generated-{index:02d}-control",
            serialized_task_id=contract.task_id,
            images=contract.images,
            expected_decision="forwarded",
            expected_reason="evidence_binding_satisfied",
            input_flow="control",
            expected_task_id=contract.task_id,
            serialized_task_text=contract.task_text,
        )
        dispatches.append(
            GeneratedDispatch(
                correlation_id=control_correlation,
                sdk_family=sdk_family,
                mutation_id="control",
                contract=contract,
                case=control,
                response_binding=ClientOwnedResponseBinding.from_contract(
                    contract,
                    correlation_id=control_correlation,
                ),
                is_control=True,
            )
        )
        for mutation in spec["mutations"]:
            case, active_field, guard_contract = mutation_case(
                index, contract, contracts, mutation
            )
            correlation_id = (
                f"{sdk_prefix}-g{index:02d}-{mutation['mutation_id']}"
            )
            binding = ClientOwnedResponseBinding.from_contract(
                guard_contract,
                correlation_id=correlation_id,
            )
            if active_field != guard_contract.active_field:
                binding = ClientOwnedResponseBinding(
                    **{
                        **binding.commitment(),
                        "active_field": active_field,
                    }
                )
            dispatches.append(
                GeneratedDispatch(
                    correlation_id=correlation_id,
                    sdk_family=sdk_family,
                    mutation_id=mutation["mutation_id"],
                    contract=guard_contract,
                    case=case,
                    response_binding=binding,
                    is_control=False,
                )
            )
    if len(dispatches) != 480 or len({row.correlation_id for row in dispatches}) != 480:
        raise ValueError("generated dispatch cardinality failed")
    return tuple(dispatches)


def prompt_for(dispatch: GeneratedDispatch) -> str:
    return "\n".join(
        (
            f"evidence_task_id: {dispatch.case.serialized_task_id}",
            f"evidence_case_id: {dispatch.case.case_id}",
            dispatch.case.serialized_task_text,
        )
    )


def openai_messages(dispatch: GeneratedDispatch) -> list[dict[str, Any]]:
    text = {"type": "text", "text": prompt_for(dispatch)}
    images = openai_image_content(dispatch.case)
    if dispatch.case.message_layout == "task_marker_in_system_or_model_role":
        return [
            {"role": "system", "content": prompt_for(dispatch)},
            {"role": "user", "content": images},
        ]
    if dispatch.case.message_layout == "image_in_prior_user_turn":
        return [
            {"role": "user", "content": images},
            {"role": "user", "content": [text]},
        ]
    return [{"role": "user", "content": [text, *images]}]


def vertex_contents(dispatch: GeneratedDispatch) -> list[types.Content]:
    text = types.Part.from_text(text=prompt_for(dispatch))
    images = vertex_image_parts(dispatch.case)
    if dispatch.case.message_layout == "task_marker_in_system_or_model_role":
        return [
            types.Content(role="model", parts=[text]),
            types.Content(role="user", parts=images),
        ]
    if dispatch.case.message_layout == "image_in_prior_user_turn":
        return [
            types.Content(role="user", parts=images),
            types.Content(role="user", parts=[text]),
        ]
    return [types.Content(role="user", parts=[text, *images])]


def expected_blocked_response(dispatch: GeneratedDispatch) -> dict[str, str]:
    return {
        "decision": dispatch.case.expected_decision,
        "reason": dispatch.case.expected_reason,
    }


def contract_sha256(contract: EvidenceFieldContract) -> str:
    return contract.sha256


class SharedOpenAITransport:
    def __init__(
        self,
        dispatches: Mapping[str, GeneratedDispatch],
        task_registry: Mapping[str, EvidenceFieldContract],
    ) -> None:
        self.dispatches = MappingProxyType(dict(dispatches))
        self.task_registry = MappingProxyType(dict(task_registry))
        self.attempt_counts: Counter[str] = Counter()
        self.attempts: list[dict[str, Any]] = []
        self.lock = Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        correlation_id = request.headers.get("x-evidence-correlation-id", "")
        try:
            dispatch = self.dispatches[correlation_id]
        except KeyError as error:
            raise ValueError(f"unknown correlation ID: {correlation_id!r}") from error
        with self.lock:
            attempt_index = self.attempt_counts[correlation_id]
            self.attempt_counts[correlation_id] += 1
        inject_retry = dispatch.is_control and attempt_index == 0
        downstream = AttemptDownstream(
            OPENAI_ROUTE,
            inject_retry,
            dispatch,
            openai_response,
        )
        oracle = assess_serialized_request(
            request.content,
            "openai",
            dispatch.contract.commitment(),
            dispatch.contract.task_text,
            dispatch.correlation_id,
            dispatch.response_binding.commitment(),
        )
        guard = GuardedTransport(
            route=OPENAI_ROUTE,
            sdk_family="openai",
            expectation=DispatchExpectation(
                dispatch.case.case_id,
                dispatch.contract,
                dispatch.correlation_id,
                dispatch.response_binding.commitment(),
            ),
            blocked_response_formatter=openai_response,
            downstream=downstream,
            task_registry=self.task_registry,
            response_schema_inspector=inspect_serialized_response_schema,
        )
        response = guard(request)
        record = guard.captured[0]
        invocation_kinds = [row["kind"] for row in downstream.invocations]
        attempt = {
            "active_message_index": record["active_message_index"],
            "active_message_role": record["active_message_role"],
            "active_scope_path": record["active_scope_path"],
            "attempt_index": attempt_index,
            "body_sha256": record["body_sha256"],
            "case_id": dispatch.case.case_id,
            "contract_sha256": contract_sha256(dispatch.contract),
            "correlation_id": correlation_id,
            "decision": record["decision"],
            "expected_task_id": record["expected_task_id"],
            "expected_task_text_sha256": record["expected_task_text_sha256"],
            "guard_shared_contract_object": (
                guard.expectation.contract is dispatch.contract
            ),
            "invocation_kinds": invocation_kinds,
            "model_spy_invocations": invocation_kinds.count("model_spy"),
            "oracle_allowed": oracle.allowed,
            "oracle_reason": oracle.reason,
            "reason": record["reason"],
            "retry_injected": "retry_injector" in invocation_kinds,
            "scoped_image_count": record["scoped_image_count"],
            "scoped_task_id": record["scoped_task_id"],
            "scoped_task_text_sha256": record["scoped_task_text_sha256"],
            "sdk_family": "openai",
            "serialized_task_id": record["serialized_task_id"],
            "client_response_binding": record["client_response_binding"],
            "serialized_response_schema": record["serialized_response_schema"],
            "status_code": response.status_code,
        }
        with self.lock:
            self.attempts.append(attempt)
        return response

    def final_successful_attempt(self, correlation_id: str) -> dict[str, Any]:
        with self.lock:
            rows = sorted(
                (
                    dict(row)
                    for row in self.attempts
                    if row["correlation_id"] == correlation_id
                ),
                key=lambda row: row["attempt_index"],
            )
        if not rows or rows[-1]["status_code"] != 200:
            raise ValueError(f"{correlation_id} has no final successful SDK attempt")
        return rows[-1]


def verify_logical_response(
    dispatch: GeneratedDispatch,
    response: dict[str, Any],
    *,
    dispatch_body_sha256: str,
    provider_response_id: str | None,
) -> dict[str, Any]:
    if not dispatch.is_control:
        expected = expected_blocked_response(dispatch)
        if response != expected:
            raise ValueError(
                f"{dispatch.correlation_id} produced {response}, expected {expected}"
            )
        return {
            "authenticated_response_receipt_verified": False,
            "client_owned_response_receipt": None,
            "parser_contract_object_shared": False,
            "parser_dispatch_body_sha256": None,
            "parser_response_binding_object_shared": False,
            "parser_drift_attempted": False,
            "parser_drift_rejected_before_extraction": False,
            "parser_field": None,
            "parser_invocations": 0,
            "parser_selected_contract_field": False,
            "provider_payload_sha256": None,
            "provider_response_id": None,
            "receipt_dispatch_body_match": False,
            "receipt_metadata_client_owned": False,
            "receipt_payload_mismatch": False,
            "receipt_provider_response_id_match": False,
            "receipt_response_binding_match": False,
            "wrong_field_extractions": 0,
        }

    if not provider_response_id:
        raise ValueError(
            f"{dispatch.correlation_id} returned no provider response ID"
        )
    receipt = TrustedResponseReceipt.issue(
        dispatch.response_binding,
        dispatch_body_sha256=dispatch_body_sha256,
        provider_response_id=provider_response_id,
        provider_payload=response,
        key_id=RESPONSE_RECEIPT_KEY_ID,
        signing_key=RESPONSE_RECEIPT_KEY,
    )
    receipt_commitment = receipt.commitment()
    if set(receipt_commitment) != set(RESPONSE_RECEIPT_METADATA_FIELDS):
        raise ValueError("trusted response receipt metadata fields changed")
    receipt_metadata_client_owned = not set(response).intersection(
        RESPONSE_RECEIPT_METADATA_FIELDS
    )
    if not receipt_metadata_client_owned:
        raise ValueError("trusted response receipt metadata entered provider payload")
    parser = BoundResponseParser(
        dispatch.contract,
        dispatch.response_binding,
        dispatch.contract.active_field,
        dispatch_body_sha256,
        RESPONSE_RECEIPT_KEY,
    )
    envelope = ClientOwnedResponseEnvelope.from_trusted_completion(
        dispatch.response_binding,
        response,
        receipt,
    )
    selected = parser.parse(envelope)
    observed_payload_sha256 = TrustedResponseReceipt.payload_sha256(response)
    receipt_payload_mismatch = (
        selected.provider_payload_sha256 != observed_payload_sha256
    )
    receipt_dispatch_body_match = (
        selected.dispatch_body_sha256 == dispatch_body_sha256
    )
    receipt_provider_response_id_match = (
        selected.provider_response_id == provider_response_id
    )
    receipt_response_binding_match = (
        receipt.contract_sha256 == dispatch.response_binding.contract_sha256
        and receipt.correlation_id == dispatch.response_binding.correlation_id
        and envelope.binding is dispatch.response_binding
    )
    if (
        receipt_payload_mismatch
        or not receipt_dispatch_body_match
        or not receipt_provider_response_id_match
        or not receipt_response_binding_match
    ):
        raise ValueError(
            f"{dispatch.correlation_id} lost trusted response receipt association"
        )
    expected_value = mock_structured_response(dispatch.contract)[
        dispatch.contract.active_field
    ]
    if selected.field_name != dispatch.contract.active_field:
        raise ValueError(f"{dispatch.correlation_id} selected the wrong field")
    if selected.value != expected_value:
        raise ValueError(f"{dispatch.correlation_id} extracted the wrong value")
    drift_field = next(
        field for field in RESPONSE_FIELDS if field != dispatch.contract.active_field
    )
    try:
        BoundResponseParser(
            dispatch.contract,
            dispatch.response_binding,
            drift_field,
            dispatch_body_sha256,
            RESPONSE_RECEIPT_KEY,
        )
    except ParserContractMismatch:
        drift_rejected = True
    else:
        drift_rejected = False
    if not drift_rejected:
        raise ValueError(f"{dispatch.correlation_id} accepted parser drift")
    if parser.contract is not dispatch.contract:
        raise ValueError(
            f"{dispatch.correlation_id} did not share the immutable contract"
        )
    return {
        "authenticated_response_receipt_verified": True,
        "client_owned_response_receipt": receipt_commitment,
        "parser_contract_object_shared": True,
        "parser_dispatch_body_sha256": selected.dispatch_body_sha256,
        "parser_response_binding_object_shared": (
            parser.response_binding is dispatch.response_binding
        ),
        "parser_drift_attempted": True,
        "parser_drift_rejected_before_extraction": True,
        "parser_field": selected.field_name,
        "parser_invocations": 1,
        "parser_selected_contract_field": True,
        "provider_payload_sha256": selected.provider_payload_sha256,
        "provider_response_id": selected.provider_response_id,
        "receipt_dispatch_body_match": receipt_dispatch_body_match,
        "receipt_metadata_client_owned": receipt_metadata_client_owned,
        "receipt_payload_mismatch": receipt_payload_mismatch,
        "receipt_provider_response_id_match": receipt_provider_response_id_match,
        "receipt_response_binding_match": receipt_response_binding_match,
        "wrong_field_extractions": 0,
    }


def run_openai_dispatches(
    dispatches: tuple[GeneratedDispatch, ...],
    task_registry: Mapping[str, EvidenceFieldContract],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dispatch_by_id = {row.correlation_id: row for row in dispatches}
    transport = SharedOpenAITransport(dispatch_by_id, task_registry)
    http_client = httpx.Client(transport=httpx.MockTransport(transport))
    client = OpenAI(
        api_key="test",
        base_url="https://openrouter.test/api/v1",
        http_client=http_client,
        max_retries=1,
    )

    controls = tuple(row for row in dispatches if row.is_control)
    mutations = tuple(row for row in dispatches if not row.is_control)
    barrier = Barrier(len(controls))

    def invoke(
        dispatch: GeneratedDispatch, start_barrier: Barrier | None
    ) -> tuple[str, dict[str, Any]]:
        if start_barrier is not None:
            start_barrier.wait()
        completion = client.chat.completions.create(
            model=OPENAI_CONFIG["model"],
            messages=openai_messages(dispatch),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        f"evidence_response_v{dispatch.contract.contract_version}"
                    ),
                    "strict": True,
                    "schema": response_json_schema(),
                },
            },
            extra_body={"provider": OPENAI_CONFIG["provider"]},
            extra_headers={"x-evidence-correlation-id": dispatch.correlation_id},
        )
        final_success = transport.final_successful_attempt(dispatch.correlation_id)
        return (
            dispatch.correlation_id,
            verify_logical_response(
                dispatch,
                json.loads(completion.choices[0].message.content or "{}"),
                dispatch_body_sha256=final_success["body_sha256"],
                provider_response_id=completion.id,
            ),
        )

    try:
        with ThreadPoolExecutor(
            max_workers=WORKERS,
            thread_name_prefix="shared-openai-control",
        ) as executor:
            control_results = list(
                executor.map(lambda row: invoke(row, barrier), controls)
            )
        with ThreadPoolExecutor(
            max_workers=WORKERS,
            thread_name_prefix="shared-openai-mutation",
        ) as executor:
            mutation_results = list(
                executor.map(lambda row: invoke(row, None), mutations)
            )
    finally:
        client.close()

    attempts = sorted(
        transport.attempts,
        key=lambda row: (row["correlation_id"], row["attempt_index"]),
    )
    logical: list[dict[str, Any]] = []
    parser_results = dict([*control_results, *mutation_results])
    attempts_by_correlation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_correlation[attempt["correlation_id"]].append(attempt)
    for dispatch in sorted(dispatches, key=lambda row: row.correlation_id):
        rows = attempts_by_correlation[dispatch.correlation_id]
        final = rows[-1]
        logical.append(
            {
                "attempt_count": len(rows),
                "body_sha256": final["body_sha256"],
                "case_id": dispatch.case.case_id,
                "contract_active_field": dispatch.contract.active_field,
                "contract_task_id": dispatch.contract.task_id,
                "contract_task_text_sha256": dispatch.contract.task_text_sha256,
                "contract_sha256": final["contract_sha256"],
                "correlation_id": dispatch.correlation_id,
                "decision": final["decision"],
                "is_control": dispatch.is_control,
                "guard_shared_contract_object": all(
                    row["guard_shared_contract_object"] for row in rows
                ),
                "model_spy_invocations": sum(
                    row["model_spy_invocations"] for row in rows
                ),
                "mutation_id": dispatch.mutation_id,
                "reason": final["reason"],
                "oracle_allowed": final["oracle_allowed"],
                "oracle_reason": final["oracle_reason"],
                "retry_injections": sum(row["retry_injected"] for row in rows),
                "sdk_family": "openai",
                "client_response_binding": final["client_response_binding"],
                "serialized_response_schema": final[
                    "serialized_response_schema"
                ],
                **parser_results[dispatch.correlation_id],
            }
        )
    return logical, attempts


def run_google_dispatch(
    dispatch: GeneratedDispatch,
    task_registry: Mapping[str, EvidenceFieldContract],
) -> tuple[dict[str, Any], dict[str, Any]]:
    downstream = AttemptDownstream(
        GOOGLE_ROUTE,
        False,
        dispatch,
        vertex_response,
    )
    guard = GuardedTransport(
        route=GOOGLE_ROUTE,
        sdk_family="google_genai",
        expectation=DispatchExpectation(
            dispatch.case.case_id,
            dispatch.contract,
            dispatch.correlation_id,
            dispatch.response_binding.commitment(),
        ),
        blocked_response_formatter=vertex_response,
        downstream=downstream,
        task_registry=task_registry,
        response_schema_inspector=inspect_serialized_response_schema,
    )
    oracle_capture: list[Any] = []

    def independently_checked_guard(request: httpx.Request) -> httpx.Response:
        oracle_capture.append(
            assess_serialized_request(
                request.content,
                "google_genai",
                dispatch.contract.commitment(),
                dispatch.contract.task_text,
                dispatch.correlation_id,
                dispatch.response_binding.commitment(),
            )
        )
        return guard(request)

    credentials = AnonymousCredentials()
    credentials.token = "test"
    client = genai.Client(
        vertexai=True,
        project="offline-project",
        location="global",
        credentials=credentials,
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(
                transport=httpx.MockTransport(independently_checked_guard)
            )
        ),
    )
    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=4000,
        response_mime_type="application/json",
        response_json_schema=response_json_schema(),
    )
    try:
        result = client.models.generate_content(
            model=VERTEX_MODEL,
            contents=vertex_contents(dispatch),
            config=config,
        )
        response = json.loads(result.text or "{}")
        record = guard.captured[0]
        parser_result = verify_logical_response(
            dispatch,
            response,
            dispatch_body_sha256=record["body_sha256"],
            provider_response_id=result.response_id,
        )
    finally:
        client.close()
    if len(oracle_capture) != 1:
        raise ValueError(
            f"{dispatch.correlation_id} produced {len(oracle_capture)} oracle checks"
        )
    oracle = oracle_capture[0]
    model_spy_invocations = len(downstream.invocations)
    logical = {
        "attempt_count": 1,
        "body_sha256": record["body_sha256"],
        "case_id": dispatch.case.case_id,
        "contract_active_field": dispatch.contract.active_field,
        "contract_sha256": contract_sha256(dispatch.contract),
        "contract_task_id": dispatch.contract.task_id,
        "contract_task_text_sha256": dispatch.contract.task_text_sha256,
        "correlation_id": dispatch.correlation_id,
        "decision": record["decision"],
        "is_control": dispatch.is_control,
        "guard_shared_contract_object": (
            guard.expectation.contract is dispatch.contract
        ),
        "model_spy_invocations": model_spy_invocations,
        "mutation_id": dispatch.mutation_id,
        "reason": record["reason"],
        "oracle_allowed": oracle.allowed,
        "oracle_reason": oracle.reason,
        "retry_injections": 0,
        "sdk_family": "google_genai",
        "client_response_binding": record["client_response_binding"],
        "serialized_response_schema": record["serialized_response_schema"],
        **parser_result,
    }
    attempt = {
        "active_message_index": record["active_message_index"],
        "active_message_role": record["active_message_role"],
        "active_scope_path": record["active_scope_path"],
        "attempt_index": 0,
        "body_sha256": record["body_sha256"],
        "case_id": dispatch.case.case_id,
        "contract_sha256": contract_sha256(dispatch.contract),
        "correlation_id": dispatch.correlation_id,
        "decision": record["decision"],
        "expected_task_id": record["expected_task_id"],
        "expected_task_text_sha256": record["expected_task_text_sha256"],
        "guard_shared_contract_object": (
            guard.expectation.contract is dispatch.contract
        ),
        "invocation_kinds": ["model_spy"] if model_spy_invocations else [],
        "model_spy_invocations": model_spy_invocations,
        "oracle_allowed": oracle.allowed,
        "oracle_reason": oracle.reason,
        "reason": record["reason"],
        "retry_injected": False,
        "scoped_image_count": record["scoped_image_count"],
        "scoped_task_id": record["scoped_task_id"],
        "scoped_task_text_sha256": record["scoped_task_text_sha256"],
        "sdk_family": "google_genai",
        "serialized_task_id": record["serialized_task_id"],
        "client_response_binding": record["client_response_binding"],
        "serialized_response_schema": record["serialized_response_schema"],
        "status_code": 200,
    }
    return logical, attempt


def run_google_dispatches(
    dispatches: tuple[GeneratedDispatch, ...],
    task_registry: Mapping[str, EvidenceFieldContract],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with ThreadPoolExecutor(
        max_workers=WORKERS,
        thread_name_prefix="google-generated-property",
    ) as executor:
        runs = list(
            executor.map(
                lambda row: run_google_dispatch(row, task_registry),
                dispatches,
            )
        )
    logical = sorted((row[0] for row in runs), key=lambda row: row["correlation_id"])
    attempts = sorted((row[1] for row in runs), key=lambda row: row["correlation_id"])
    return logical, attempts


def contract_commitment(contract: EvidenceFieldContract) -> dict[str, Any]:
    return contract.commitment()


def validate_and_summarize(
    spec: dict[str, Any],
    contracts: tuple[EvidenceFieldContract, ...],
    logical: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    controls = [row for row in logical if row["is_control"]]
    mutations = [row for row in logical if not row["is_control"]]
    blocked_model_spy = sum(
        row["model_spy_invocations"]
        for row in mutations
        if row["decision"] == "blocked"
    )
    totals = {
        "authenticated_response_receipts_verified": sum(
            row["authenticated_response_receipt_verified"] for row in controls
        ),
        "blocked_mutations": sum(row["decision"] == "blocked" for row in mutations),
        "client_owned_response_receipts": sum(
            row["client_owned_response_receipt"] is not None for row in controls
        ),
        "control_dispatches": len(controls),
        "forwarded_controls": sum(row["decision"] == "forwarded" for row in controls),
        "independent_oracle_agreements": sum(
            row["oracle_allowed"] == (row["decision"] == "forwarded")
            for row in logical
        ),
        "independent_oracle_allowed_controls": sum(
            row["oracle_allowed"] for row in controls
        ),
        "independent_oracle_blocked_mutations": sum(
            not row["oracle_allowed"] for row in mutations
        ),
        "independent_oracle_physical_checks": len(attempts),
        "logical_dispatches": len(logical),
        "model_spy_invocations": sum(row["model_spy_invocations"] for row in logical),
        "mutation_dispatches": len(mutations),
        "mutation_model_spy_invocations": blocked_model_spy,
        "network_calls": 0,
        "physical_attempts": len(attempts),
        "parser_contract_object_shared": sum(
            row["parser_contract_object_shared"] for row in controls
        ),
        "parser_drift_attempts": sum(
            row["parser_drift_attempted"] for row in controls
        ),
        "parser_drift_rejections": sum(
            row["parser_drift_rejected_before_extraction"] for row in controls
        ),
        "parser_invocations": sum(row["parser_invocations"] for row in logical),
        "parser_response_binding_object_shared": sum(
            row["parser_response_binding_object_shared"] for row in controls
        ),
        "parser_selected_contract_field": sum(
            row["parser_selected_contract_field"] for row in controls
        ),
        "retry_injections": sum(row["retry_injected"] for row in attempts),
        "receipt_payload_mismatches": sum(
            row["receipt_payload_mismatch"] for row in controls
        ),
        "unique_logical_body_sha256": len({row["body_sha256"] for row in logical}),
        "wrong_field_extractions": sum(
            row["wrong_field_extractions"] for row in logical
        ),
    }
    expected_totals = {
        "authenticated_response_receipts_verified": 64,
        "blocked_mutations": 896,
        "client_owned_response_receipts": 64,
        "control_dispatches": 64,
        "forwarded_controls": 64,
        "independent_oracle_agreements": 960,
        "independent_oracle_allowed_controls": 64,
        "independent_oracle_blocked_mutations": 896,
        "independent_oracle_physical_checks": 992,
        "logical_dispatches": 960,
        "model_spy_invocations": 64,
        "mutation_dispatches": 896,
        "mutation_model_spy_invocations": 0,
        "network_calls": 0,
        "physical_attempts": 992,
        "parser_contract_object_shared": 64,
        "parser_drift_attempts": 64,
        "parser_drift_rejections": 64,
        "parser_invocations": 64,
        "parser_response_binding_object_shared": 64,
        "parser_selected_contract_field": 64,
        "retry_injections": 32,
        "receipt_payload_mismatches": 0,
        "unique_logical_body_sha256": 960,
        "wrong_field_extractions": 0,
    }
    if totals != expected_totals:
        raise ValueError(f"generated property totals failed: {totals}")

    oracle_audit = {
        "decision_agreements": totals["independent_oracle_agreements"],
        "implementation_path": "code/independent_contract_oracle.py",
        "implementation_sha256": sha256_bytes(ORACLE_PATH.read_bytes()),
        "logical_dispatches": len(logical),
        "oracle_allowed_controls": totals["independent_oracle_allowed_controls"],
        "oracle_blocked_mutations": totals[
            "independent_oracle_blocked_mutations"
        ],
        "physical_checks": totals["independent_oracle_physical_checks"],
        "reason_counts": dict(
            sorted(Counter(row["oracle_reason"] for row in logical).items())
        ),
    }

    mutation_summary: dict[str, dict[str, Any]] = {}
    for mutation in spec["mutations"]:
        rows = [
            row for row in mutations if row["mutation_id"] == mutation["mutation_id"]
        ]
        reasons = Counter(row["reason"] for row in rows)
        summary = {
            "blocked": sum(row["decision"] == "blocked" for row in rows),
            "expected_reason": mutation["expected_reason"],
            "logical_dispatches": len(rows),
            "model_spy_invocations": sum(row["model_spy_invocations"] for row in rows),
            "observed_reasons": dict(sorted(reasons.items())),
        }
        expected_summary = {
            "blocked": 64,
            "expected_reason": mutation["expected_reason"],
            "logical_dispatches": 64,
            "model_spy_invocations": 0,
            "observed_reasons": {mutation["expected_reason"]: 64},
        }
        if summary != expected_summary:
            raise ValueError(
                f"mutation {mutation['mutation_id']} failed: {summary}"
            )
        mutation_summary[mutation["mutation_id"]] = summary

    openai_controls = [
        row for row in logical if row["sdk_family"] == "openai" and row["is_control"]
    ]
    attempts_by_correlation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_correlation[attempt["correlation_id"]].append(attempt)
    correlated_retries = 0
    same_body_retries = 0
    same_contract_retries = 0
    for row in openai_controls:
        rows = sorted(
            attempts_by_correlation[row["correlation_id"]],
            key=lambda item: item["attempt_index"],
        )
        if len(rows) != 2 or [item["status_code"] for item in rows] != [429, 200]:
            raise ValueError(f"retry sequence failed for {row['correlation_id']}")
        if all(item["correlation_id"] == row["correlation_id"] for item in rows):
            correlated_retries += 1
        if len({item["body_sha256"] for item in rows}) == 1:
            same_body_retries += 1
        if len({item["contract_sha256"] for item in rows}) == 1:
            same_contract_retries += 1
    retry_audit = {
        "correlated_retries": correlated_retries,
        "forced_retry_controls": len(openai_controls),
        "same_body_retries": same_body_retries,
        "same_contract_retries": same_contract_retries,
    }
    if retry_audit != {
        "correlated_retries": 32,
        "forced_retry_controls": 32,
        "same_body_retries": 32,
        "same_contract_retries": 32,
    }:
        raise ValueError(f"shared-client retry audit failed: {retry_audit}")

    parser_audit = {
        "authenticated_response_receipts_verified": totals[
            "authenticated_response_receipts_verified"
        ],
        "control_responses_with_all_candidate_fields": sum(
            row["is_control"] and row["parser_invocations"] == 1
            for row in logical
        ),
        "contract_field_selections": totals["parser_selected_contract_field"],
        "drift_attempts": totals["parser_drift_attempts"],
        "drift_rejections_before_extraction": totals["parser_drift_rejections"],
        "parser_invocations": totals["parser_invocations"],
        "receipt_payload_mismatches": totals["receipt_payload_mismatches"],
        "selections_by_field": dict(
            sorted(Counter(row["parser_field"] for row in controls).items())
        ),
        "shared_immutable_contract_objects": totals[
            "parser_contract_object_shared"
        ],
        "shared_immutable_response_bindings": totals[
            "parser_response_binding_object_shared"
        ],
        "wrong_field_extractions": totals["wrong_field_extractions"],
    }
    if parser_audit != {
        "authenticated_response_receipts_verified": 64,
        "control_responses_with_all_candidate_fields": 64,
        "contract_field_selections": 64,
        "drift_attempts": 64,
        "drift_rejections_before_extraction": 64,
        "parser_invocations": 64,
        "receipt_payload_mismatches": 0,
        "selections_by_field": {
            "clinical_assessment": 20,
            "differentials": 22,
            "primary_diagnosis": 22,
        },
        "shared_immutable_contract_objects": 64,
        "shared_immutable_response_bindings": 64,
        "wrong_field_extractions": 0,
    }:
        raise ValueError(f"bound response parser audit failed: {parser_audit}")

    provider_schema_receipt_metadata_fields = sorted(
        {
            field_name
            for row in controls
            for field_name in row["serialized_response_schema"]["properties"]
            if field_name in RESPONSE_RECEIPT_METADATA_FIELDS
        }
    )
    response_receipt_audit = {
        "authenticated_receipts_verified": totals[
            "authenticated_response_receipts_verified"
        ],
        "client_owned_receipt_sidecars": totals["client_owned_response_receipts"],
        "control_provider_schemas_without_receipt_metadata": sum(
            not set(row["serialized_response_schema"]["properties"]).intersection(
                RESPONSE_RECEIPT_METADATA_FIELDS
            )
            for row in controls
        ),
        "fixture_key_id": RESPONSE_RECEIPT_KEY_ID,
        "fixture_verification_key_sha256": sha256_bytes(RESPONSE_RECEIPT_KEY),
        "provider_schema_receipt_metadata_fields": (
            provider_schema_receipt_metadata_fields
        ),
        "receipt_metadata_absent_from_provider_schema": (
            not provider_schema_receipt_metadata_fields
        ),
        "receipt_metadata_client_owned": all(
            row["receipt_metadata_client_owned"] for row in controls
        ),
        "receipt_metadata_fields": list(RESPONSE_RECEIPT_METADATA_FIELDS),
        "receipt_payload_mismatches": totals["receipt_payload_mismatches"],
        "receipts_bound_to_final_successful_body": sum(
            row["receipt_dispatch_body_match"]
            and row["parser_dispatch_body_sha256"] == row["body_sha256"]
            for row in controls
        ),
        "receipts_bound_to_client_response_binding": sum(
            row["receipt_response_binding_match"] for row in controls
        ),
        "receipts_bound_to_returned_response_id": sum(
            row["receipt_provider_response_id_match"] for row in controls
        ),
    }
    if response_receipt_audit != {
        "authenticated_receipts_verified": 64,
        "client_owned_receipt_sidecars": 64,
        "control_provider_schemas_without_receipt_metadata": 64,
        "fixture_key_id": "generated-mutation-response-fixture-v1",
        "fixture_verification_key_sha256": sha256_bytes(RESPONSE_RECEIPT_KEY),
        "provider_schema_receipt_metadata_fields": [],
        "receipt_metadata_absent_from_provider_schema": True,
        "receipt_metadata_client_owned": True,
        "receipt_metadata_fields": list(RESPONSE_RECEIPT_METADATA_FIELDS),
        "receipt_payload_mismatches": 0,
        "receipts_bound_to_client_response_binding": 64,
        "receipts_bound_to_final_successful_body": 64,
        "receipts_bound_to_returned_response_id": 64,
    }:
        raise ValueError(
            f"trusted response receipt audit failed: {response_receipt_audit}"
        )

    return {
        "audit_name": "deterministic generated evidence-contract mutation properties",
        "audit_schema_version": 6,
        "contract_manifest": {
            "contract_name": spec["contract_name"],
            "contract_sha256": sha256_bytes(CONTRACT_PATH.read_bytes()),
            "contract_version": spec["contract_version"],
            "failure_action": spec["failure_action"],
            "protected_fields": spec["contract_fields"],
            "request_scope": spec["request_scope"],
            "retry_invariant": spec["retry_invariant"],
        },
        "contracts": [contract_commitment(contract) for contract in contracts],
        "dependencies": {
            "google_genai": genai.__version__,
            "httpx": httpx.__version__,
            "openai": openai.__version__,
            "pillow": PIL.__version__,
        },
        "dispatch": {
            "client_response_receipt_sidecar_per_successful_control": True,
            "google_client_per_dispatch": True,
            "mutation_operators": len(spec["mutations"]),
            "openai_client_instances": 1,
            "openai_max_retries": 1,
            "openai_shared_client": True,
            "seed": spec["generator"]["seed"],
            "client_response_binding_sidecar_per_dispatch": True,
            "provider_semantic_response_schema_per_dispatch": True,
            "provider_response_schema_contains_receipt_metadata": False,
            "task_contracts": len(contracts),
            "workers_per_sdk_family": WORKERS,
        },
        "logical_audit": sorted(
            logical, key=lambda row: (row["sdk_family"], row["correlation_id"])
        ),
        "mutation_summary": mutation_summary,
        "network_calls": 0,
        "independent_oracle_audit": oracle_audit,
        "parser_audit": parser_audit,
        "physical_attempt_audit": sorted(
            attempts,
            key=lambda row: (
                row["sdk_family"],
                row["correlation_id"],
                row["attempt_index"],
            ),
        ),
        "response_receipt_audit": response_receipt_audit,
        "retry_audit": retry_audit,
        "scope": (
            "Deterministically generated task ID, canonical task text, evidence, "
            "active-field, role, turn, order, MIME, digest, count, and reference "
            "mutations are exercised "
            "through current OpenAI and Google Gen AI SDK serialization with local "
            "mock transports. A separately implemented request oracle independently "
            "checks every serialized body and is compared with the enforcement guard. "
            "The provider-facing response schema contains only semantic fields; "
            "authenticated receipt metadata remains in a client-owned sidecar. Each "
            "control's returned synthetic payload is bound to the final successful SDK "
            "body hash, returned response ID, and response binding before parser "
            "verification. The post-response parser shares the immutable caller "
            "contract, selects its field, and rejects parser drift before extraction. "
            "OpenAI controls share one concurrent "
            "client and receive one forced retry. This is a regression property audit, "
            "not evidence of "
            "production provider receipt, production effectiveness, or exhaustive "
            "mutation coverage."
        ),
        "sdk_families": ["google_genai", "openai"],
        "totals": totals,
    }


def run() -> dict[str, Any]:
    spec = load_contract_spec()
    contracts = generate_contracts(spec)
    task_registry = MappingProxyType({row.task_id: row for row in contracts})
    openai_dispatches = build_dispatches("openai", contracts, spec)
    google_dispatches = build_dispatches("google_genai", contracts, spec)
    openai_logical, openai_attempts = run_openai_dispatches(
        openai_dispatches, task_registry
    )
    google_logical, google_attempts = run_google_dispatches(
        google_dispatches, task_registry
    )
    return validate_and_summarize(
        spec,
        contracts,
        [*openai_logical, *google_logical],
        [*openai_attempts, *google_attempts],
    )


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
                "blocked_mutations": result["totals"]["blocked_mutations"],
                "correlated_retries": result["retry_audit"]["correlated_retries"],
                "forwarded_controls": result["totals"]["forwarded_controls"],
                "network_calls": result["network_calls"],
                "out": str(args.out),
                "response_receipts_verified": result["response_receipt_audit"][
                    "authenticated_receipts_verified"
                ],
                "parser_drift_rejections": result["parser_audit"][
                    "drift_rejections_before_extraction"
                ],
                "parser_field_selections": result["parser_audit"][
                    "contract_field_selections"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
