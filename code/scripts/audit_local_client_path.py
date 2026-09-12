#!/usr/bin/env python3
"""Audit a deterministic local queue-to-client evidence path."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Literal

import httpx
import openai
from openai import OpenAI

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evidence_contract import (
    BoundImage,
    BoundResponseParser,
    ClientOwnedResponseBinding,
    ClientOwnedResponseEnvelope,
    DispatchAnchorMismatch,
    EnvelopeContractMismatch,
    EvidenceFieldContract,
    LocalDispatchEnvelope,
    ParserContractMismatch,
    ProducerDispatchAnchor,
    RESPONSE_FIELDS,
    TrustedResponseReceipt,
    canonical_json_bytes,
    inspect_serialized_response_schema,
    response_json_schema,
    verify_detached_dispatch,
)
from audit_evidence_binding_matrix import (
    DispatchExpectation,
    EvidencePart,
    GuardedTransport,
    JPEG_IMAGE,
    MULTI_PNG_A,
    MULTI_PNG_B,
    OPENAI_ROUTES,
    PNG_IMAGE,
    WEBP_IMAGE,
    decoded_mime_type,
    openai_response,
)


TASK_TEXT = "Describe the bound image evidence for the local-client audit."
MODEL = OPENAI_ROUTES["openrouter_openai_sdk"]["model"]
PROVIDER = OPENAI_ROUTES["openrouter_openai_sdk"]["provider"]
ANCHOR_KEY_ENV = "MEDICAL_MIRAGE_LOCAL_ANCHOR_KEY"
LOOPBACK_URL_ENV = "MEDICAL_MIRAGE_LOOPBACK_URL"
RESPONSE_RECEIPT_KEY_ENV = "MEDICAL_MIRAGE_RESPONSE_RECEIPT_KEY"
ANCHOR_KEY_ID = "local-producer-fixture-v1"
ANCHOR_KEY = hashlib.sha256(b"local producer fixture key v1").digest()
FORGED_ANCHOR_KEY = hashlib.sha256(b"untrusted queue writer fixture key").digest()
RESPONSE_RECEIPT_KEY_ID = "local-response-adapter-fixture-v1"
RESPONSE_RECEIPT_KEY = hashlib.sha256(b"local response adapter fixture key v1").digest()
FIXTURES = {
    "png": PNG_IMAGE,
    "jpeg": JPEG_IMAGE,
    "webp": WEBP_IMAGE,
    "two-png": (MULTI_PNG_A, MULTI_PNG_B),
}


@dataclass
class SinkSpy:
    writes: list[dict[str, str]] = field(default_factory=list)

    def write(self, correlation_id: str, field_name: str, value: Any) -> None:
        self.writes.append(
            {
                "correlation_id": correlation_id,
                "field_name": field_name,
                "value_sha256": sha256(str(value).encode("utf-8")),
            }
        )


class LoopbackHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "unknown loopback path"}})
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.request_count += 1
        mode = self.headers.get("x-local-upstream-mode", "bound")
        if mode == "retry_429":
            self.send_json(
                429,
                {
                    "error": {
                        "code": "rate_limit",
                        "message": "synthetic loopback retry",
                        "type": "rate_limit_error",
                    }
                },
            )
            return

        task_id = next(
            line.removeprefix("evidence_task_id: ")
            for message in body["messages"]
            for part in message["content"]
            if part.get("type") == "text"
            for line in part["text"].splitlines()
            if line.startswith("evidence_task_id: ")
        )
        case_id = next(
            line.removeprefix("evidence_case_id: ")
            for message in body["messages"]
            for part in message["content"]
            if part.get("type") == "text"
            for line in part["text"].splitlines()
            if line.startswith("evidence_case_id: ")
        )
        response_suffix = f"::{case_id}" if case_id.startswith("response-pair-") else ""
        response = {
            "clinical_assessment": f"decoy-assessment::{task_id}{response_suffix}",
            "differentials": [f"decoy-differential::{task_id}{response_suffix}"],
            "primary_diagnosis": f"bound-primary::{task_id}{response_suffix}",
        }
        self.send_json(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": json.dumps(response, sort_keys=True),
                            "role": "assistant",
                        },
                    }
                ],
                "created": 0,
                "id": f"loopback-{task_id}-{case_id}",
                "model": body.get("model", MODEL),
                "object": "chat.completion",
                "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
            },
        )


@dataclass(frozen=True)
class LoopbackEndpoint:
    server: ThreadingHTTPServer
    thread: Thread

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def request_count(self) -> int:
        return self.server.request_count


@contextmanager
def loopback_endpoint() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
    server.request_count = 0
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield LoopbackEndpoint(server, thread)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@dataclass
class LoopbackUpstream:
    invocations: list[dict[str, str]] = field(default_factory=list)

    def __call__(
        self,
        request: httpx.Request,
        _content: dict[str, Any],
        _body: dict[str, Any],
        case_id: str,
    ) -> httpx.Response:
        self.invocations.append(
            {"body_sha256": sha256(request.content), "case_id": case_id}
        )
        forwarded = httpx.Request(
            request.method,
            request.url,
            headers=request.headers,
            content=request.content,
        )
        with httpx.Client() as client:
            return client.send(forwarded)


@dataclass
class GuardingHTTPTransport(httpx.BaseTransport):
    guard: GuardedTransport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.guard(request)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialize(part: EvidencePart) -> BoundImage:
    if decoded_mime_type(part.data) != part.mime_type:
        raise ValueError("in-memory fixture did not materialize as its declared MIME")
    return BoundImage(part.data, part.mime_type)


def materialized_images(fixture_name: str) -> tuple[BoundImage, ...]:
    fixture = FIXTURES[fixture_name]
    parts = fixture if isinstance(fixture, tuple) else (fixture,)
    return tuple(materialize(part) for part in parts)


def envelope_for(case_id: str, fixture_name: str) -> LocalDispatchEnvelope:
    images = materialized_images(fixture_name)
    contract = EvidenceFieldContract(
        task_id=f"local-client-{case_id}",
        task_text=TASK_TEXT,
        images=images,
        active_field="primary_diagnosis",
        contract_version=1,
    )
    return LocalDispatchEnvelope(f"local-correlation-{case_id}", contract)


def response_pair_envelopes() -> tuple[LocalDispatchEnvelope, LocalDispatchEnvelope]:
    contract = EvidenceFieldContract(
        task_id="local-client-response-pair-shared",
        task_text=TASK_TEXT,
        images=materialized_images("png"),
        active_field="primary_diagnosis",
        contract_version=1,
    )
    return (
        LocalDispatchEnvelope("local-correlation-response-pair-a", contract),
        LocalDispatchEnvelope("local-correlation-response-pair-b", contract),
    )


def anchor_for(
    envelope: LocalDispatchEnvelope,
    *,
    signing_key: bytes = ANCHOR_KEY,
) -> ProducerDispatchAnchor:
    return ProducerDispatchAnchor.issue(
        envelope,
        key_id=ANCHOR_KEY_ID,
        signing_key=signing_key,
    )


def prompt(contract: EvidenceFieldContract, case_id: str, task_text: str) -> dict[str, str]:
    return {
        "type": "text",
        "text": "\n".join(
            (
                f"evidence_task_id: {contract.task_id}",
                f"evidence_case_id: {case_id}",
                task_text,
            )
        ),
    }


def image_parts(contract: EvidenceFieldContract) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:"
                + image.mime_type
                + ";base64,"
                + base64.b64encode(image.data).decode("ascii")
            },
        }
        for image in contract.images
    ]


def messages_for(
    contract: EvidenceFieldContract,
    case_id: str,
    body_mode: Literal["bound", "text_only", "wrong_task_text", "wrong_turn"],
) -> list[dict[str, Any]]:
    task_text = (
        "Describe unrelated transport metadata."
        if body_mode == "wrong_task_text"
        else TASK_TEXT
    )
    text = prompt(contract, case_id, task_text)
    images = image_parts(contract)
    if body_mode == "text_only":
        return [{"role": "user", "content": [text]}]
    if body_mode == "wrong_turn":
        return [
            {"role": "user", "content": images},
            {"role": "user", "content": [text]},
        ]
    return [{"role": "user", "content": [text, *images]}]


def run_worker_local(
    queue_bytes: bytes,
    anchor_bytes: bytes,
    *,
    case_id: str,
    worker_id: str,
    body_mode: Literal["bound", "text_only", "wrong_task_text", "wrong_turn"],
    upstream_mode: Literal["bound", "retry_429"],
    client_response_binding_mutation: Literal["active_field"] | None = None,
) -> tuple[LocalDispatchEnvelope, dict[str, Any], dict[str, Any] | None]:
    verification_key_text = os.environ.get(ANCHOR_KEY_ENV)
    if not verification_key_text:
        raise DispatchAnchorMismatch("worker verification key is unavailable")
    try:
        verification_key = base64.b64decode(verification_key_text, validate=True)
    except ValueError as exc:
        raise DispatchAnchorMismatch("worker verification key is invalid") from exc
    response_key_text = os.environ.get(RESPONSE_RECEIPT_KEY_ENV)
    if not response_key_text:
        raise DispatchAnchorMismatch("response receipt signing key is unavailable")
    try:
        response_signing_key = base64.b64decode(response_key_text, validate=True)
    except ValueError as exc:
        raise DispatchAnchorMismatch("response receipt signing key is invalid") from exc
    envelope, anchor = verify_detached_dispatch(
        queue_bytes,
        anchor_bytes,
        verification_key=verification_key,
    )
    if envelope.to_json_bytes() != queue_bytes:
        raise ValueError("worker did not retain the exact queue bytes")
    loopback_url = os.environ.get(LOOPBACK_URL_ENV)
    if not loopback_url:
        raise ValueError("loopback endpoint URL is unavailable")
    upstream = LoopbackUpstream()
    client_response_binding = ClientOwnedResponseBinding.from_contract(
        envelope.contract,
        correlation_id=envelope.correlation_id,
    )
    client_response_binding_commitment = client_response_binding.commitment()
    if client_response_binding_mutation == "active_field":
        client_response_binding_commitment["active_field"] = next(
            field
            for field in RESPONSE_FIELDS
            if field != envelope.contract.active_field
        )
    guard = GuardedTransport(
        route="local-openai-sdk-loopback",
        sdk_family="openai",
        expectation=DispatchExpectation(
            case_id,
            envelope.contract,
            envelope.correlation_id,
            client_response_binding_commitment,
        ),
        blocked_response_formatter=openai_response,
        downstream=upstream,
        task_registry={envelope.contract.task_id: envelope.contract},
        response_schema_inspector=inspect_serialized_response_schema,
    )
    client = OpenAI(
        api_key="local-test",
        base_url=loopback_url,
        http_client=httpx.Client(transport=GuardingHTTPTransport(guard)),
        max_retries=0,
    )
    response: dict[str, Any] | None = None
    provider_response_id: str | None = None
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages_for(envelope.contract, case_id, body_mode),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "local_evidence_response",
                    "strict": True,
                    "schema": response_json_schema(),
                },
            },
            extra_body={"provider": PROVIDER},
            extra_headers={
                "x-evidence-correlation-id": envelope.correlation_id,
                "x-local-upstream-mode": upstream_mode,
            },
        )
        response = json.loads(completion.choices[0].message.content or "{}")
        provider_response_id = completion.id
        status_code = 200
    except openai.RateLimitError:
        status_code = 429
    finally:
        client.close()
    record = guard.captured[0]
    attempt = {
        "anchor_key_id": anchor.key_id,
        "client_id": f"{worker_id}-client",
        "contract_sha256": envelope.contract.sha256,
        "correlation_id": envelope.correlation_id,
        "detached_anchor_sha256": sha256(anchor_bytes),
        "detached_anchor_verified": True,
        "guard_decision": record["decision"],
        "guard_reason": record["reason"],
        "guard_transport_writes": 1,
        "loopback_http_requests": len(upstream.invocations),
        "queue_message_sha256": sha256(queue_bytes),
        "queue_rehydrated_exact_bytes": True,
        "sdk_body_sha256": record["body_sha256"],
        "status_code": status_code,
        "worker_id": worker_id,
    }
    for field_name in ("serialized_response_schema", "client_response_binding"):
        attempt[field_name] = record[field_name]
    trusted_completion: dict[str, Any] | None = None
    if record["decision"] == "forwarded" and response is not None:
        if not provider_response_id:
            raise ValueError("trusted response adapter requires a provider response ID")
        receipt = TrustedResponseReceipt.issue(
            client_response_binding,
            dispatch_body_sha256=record["body_sha256"],
            provider_response_id=provider_response_id,
            provider_payload=response,
            key_id=RESPONSE_RECEIPT_KEY_ID,
            signing_key=response_signing_key,
        )
        attempt["provider_payload_sha256"] = receipt.provider_payload_sha256
        attempt["provider_response_id"] = receipt.provider_response_id
        attempt["response_receipt_key_id"] = receipt.key_id
        attempt["response_receipt_mac_sha256"] = receipt.mac_sha256
        trusted_completion = {
            "provider_payload": response,
            "trusted_response_receipt": receipt.commitment(),
        }
    return envelope, attempt, trusted_completion


def invoke_worker_process(
    queue_bytes: bytes,
    anchor_bytes: bytes,
    *,
    case_id: str,
    worker_id: str,
    body_mode: Literal["bound", "text_only", "wrong_task_text", "wrong_turn"],
    upstream_mode: Literal["bound", "retry_429"],
    client_response_binding_mutation: Literal["active_field"] | None = None,
) -> dict[str, Any]:
    worker_input = {
        "anchor_bytes_base64": base64.b64encode(anchor_bytes).decode("ascii"),
        "body_mode": body_mode,
        "case_id": case_id,
        "queue_bytes_base64": base64.b64encode(queue_bytes).decode("ascii"),
        "client_response_binding_mutation": client_response_binding_mutation,
        "upstream_mode": upstream_mode,
        "worker_id": worker_id,
    }
    environment = os.environ.copy()
    environment[ANCHOR_KEY_ENV] = base64.b64encode(ANCHOR_KEY).decode("ascii")
    environment[RESPONSE_RECEIPT_KEY_ENV] = base64.b64encode(
        RESPONSE_RECEIPT_KEY
    ).decode("ascii")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=canonical_json_bytes(worker_input),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def run_worker(
    queue_bytes: bytes,
    anchor_bytes: bytes,
    *,
    case_id: str,
    worker_id: str,
    body_mode: Literal["bound", "text_only", "wrong_task_text", "wrong_turn"],
    upstream_mode: Literal["bound", "retry_429"],
    client_response_binding_mutation: Literal["active_field"] | None = None,
    response_binding_mutation: Literal["contract_sha256"] | None = None,
) -> tuple[LocalDispatchEnvelope, dict[str, Any], ClientOwnedResponseEnvelope | None]:
    worker_output = invoke_worker_process(
        queue_bytes,
        anchor_bytes,
        case_id=case_id,
        worker_id=worker_id,
        body_mode=body_mode,
        upstream_mode=upstream_mode,
        client_response_binding_mutation=client_response_binding_mutation,
    )
    if "rejection" in worker_output:
        raise ValueError(
            f"{case_id} was rejected before SDK serialization: "
            f"{worker_output['rejection']['reason']}"
        )
    envelope = LocalDispatchEnvelope.from_json_bytes(queue_bytes)
    attempt = worker_output["attempt"]
    attempt["worker_process_boundary"] = True
    trusted_completion = worker_output["response"]
    if attempt["guard_decision"] != "forwarded" or trusted_completion is None:
        return envelope, attempt, None
    response_binding = ClientOwnedResponseBinding(
        **attempt["client_response_binding"]
    )
    if response_binding_mutation == "contract_sha256":
        response_binding = ClientOwnedResponseBinding(
            **{**response_binding.commitment(), "contract_sha256": "0" * 64}
        )
    return (
        envelope,
        attempt,
        ClientOwnedResponseEnvelope.from_trusted_completion(
            response_binding,
            trusted_completion["provider_payload"],
            TrustedResponseReceipt(**trusted_completion["trusted_response_receipt"]),
        ),
    )


def run_rejected_worker(
    queue_bytes: bytes,
    anchor_bytes: bytes,
    *,
    case_id: str,
    worker_id: str,
) -> dict[str, Any]:
    worker_output = invoke_worker_process(
        queue_bytes,
        anchor_bytes,
        case_id=case_id,
        worker_id=worker_id,
        body_mode="bound",
        upstream_mode="bound",
    )
    rejection = worker_output.get("rejection")
    if not isinstance(rejection, dict):
        raise ValueError(f"{case_id} unexpectedly passed detached anchor verification")
    rejection["worker_process_boundary"] = True
    return rejection


def parse_after_guard(
    envelope: LocalDispatchEnvelope,
    response: ClientOwnedResponseEnvelope,
    parser_mode: Literal["bound", "parser_drift"],
    *,
    expected_dispatch_body_sha256: str,
) -> Any:
    if parser_mode == "parser_drift":
        drift_field = next(
            field for field in RESPONSE_FIELDS if field != envelope.contract.active_field
        )
        return BoundResponseParser(
            envelope.contract,
            ClientOwnedResponseBinding.from_contract(
                envelope.contract,
                correlation_id=envelope.correlation_id,
            ),
            drift_field,
            expected_dispatch_body_sha256,
            RESPONSE_RECEIPT_KEY,
        ).parse(response)
    return BoundResponseParser.from_contract(
        envelope.contract,
        correlation_id=envelope.correlation_id,
        expected_dispatch_body_sha256=expected_dispatch_body_sha256,
        response_receipt_verification_key=RESPONSE_RECEIPT_KEY,
    ).parse(response)


def control_case(
    case_id: str,
    fixture_name: str,
    sink: SinkSpy,
    *,
    retry: bool = False,
) -> dict[str, Any]:
    original = envelope_for(case_id, fixture_name)
    queue_bytes = original.to_json_bytes()
    anchor = anchor_for(original)
    anchor_bytes = anchor.to_json_bytes()
    attempts: list[dict[str, Any]] = []
    response: ClientOwnedResponseEnvelope | None = None
    envelope: LocalDispatchEnvelope | None = None
    modes = ("retry_429", "bound") if retry else ("bound",)
    for index, mode in enumerate(modes, start=1):
        envelope, attempt, response = run_worker(
            queue_bytes,
            anchor_bytes,
            case_id=case_id,
            worker_id=f"fresh-worker-{case_id}-{index}",
            body_mode="bound",
            upstream_mode=mode,
        )
        attempts.append(attempt)
    if envelope is None or response is None or attempts[-1]["status_code"] != 200:
        raise ValueError(f"{case_id} did not complete locally")
    if retry and [row["status_code"] for row in attempts] != [429, 200]:
        raise ValueError("forced retry did not produce the expected local handoff")
    selected = parse_after_guard(
        envelope,
        response,
        "bound",
        expected_dispatch_body_sha256=attempts[-1]["sdk_body_sha256"],
    )
    sink_writes = 0
    if retry:
        sink.write(selected.correlation_id, selected.field_name, selected.value)
        sink_writes = 1
    return {
        "attempts": attempts,
        "case_id": case_id,
        "category": "valid_fixture_control",
        "contract_sha256": original.contract.sha256,
        "correlation_id": original.correlation_id,
        "detached_anchor_sha256": sha256(anchor_bytes),
        "envelope_created_after_materialization": True,
        "fixture_mime_types": [image.mime_type for image in original.contract.images],
        "fixture_name": fixture_name,
        "outcome": "forwarded",
        "parser_invocations": 1,
        "queue_message_sha256": original.queue_message_sha256,
        "sdk_serializations": len(attempts),
        "sink_writes": sink_writes,
        "stage": "sdk_guard_and_bound_parser",
    }


def tampered_queue_bytes(envelope: LocalDispatchEnvelope, mutation: str) -> bytes:
    payload = copy.deepcopy(envelope.payload())
    image = payload["materialized_images"][0]
    if mutation == "dropped_bytes":
        image["data_base64"] = ""
        image["byte_length"] = 0
        image["sha256"] = sha256(b"")
    elif mutation == "replaced_bytes_and_digest":
        image["data_base64"] = base64.b64encode(MULTI_PNG_A.data).decode("ascii")
        image["byte_length"] = len(MULTI_PNG_A.data)
        image["sha256"] = sha256(MULTI_PNG_A.data)
    elif mutation == "caller_mime":
        image["mime_type"] = "image/jpeg"
    elif mutation == "task_text":
        payload["canonical_task_text"] = "Describe tampered evidence."
    elif mutation == "image_order":
        payload["materialized_images"].reverse()
    else:
        raise ValueError(f"unsupported queue mutation: {mutation}")
    return canonical_json_bytes(payload)


def queue_tampering_case(case_id: str, mutation: str, fixture_name: str) -> dict[str, Any]:
    envelope = envelope_for(case_id, fixture_name)
    queue_bytes = tampered_queue_bytes(envelope, mutation)
    try:
        LocalDispatchEnvelope.from_json_bytes(queue_bytes)
    except EnvelopeContractMismatch as error:
        reason = str(error)
    else:
        raise ValueError(f"{case_id} unexpectedly passed envelope verification")
    return {
        "attempts": [],
        "case_id": case_id,
        "category": "post_contract_queue_tampering",
        "contract_sha256": envelope.contract.sha256,
        "correlation_id": envelope.correlation_id,
        "envelope_created_after_materialization": True,
        "guard_transport_writes": 0,
        "loopback_http_requests": 0,
        "outcome": "envelope_rejected",
        "queue_message_sha256": sha256(queue_bytes),
        "reason": reason,
        "sdk_serializations": 0,
        "sink_writes": 0,
        "stage": "queue_envelope_verification",
        "tampering": mutation,
    }


def detached_anchor_case(
    case_id: str,
    tampering: Literal[
        "coherent_rewrite", "forged_anchor", "cross_job_anchor_mismatch"
    ],
) -> dict[str, Any]:
    original = envelope_for(case_id, "png")
    queue_envelope = original
    anchor = anchor_for(original)
    if tampering == "coherent_rewrite":
        queue_envelope = LocalDispatchEnvelope(
            original.correlation_id,
            EvidenceFieldContract(
                task_id=f"{original.contract.task_id}-rewritten",
                task_text="Describe coherently substituted evidence.",
                images=materialized_images("jpeg"),
                active_field="clinical_assessment",
                contract_version=2,
            ),
        )
    elif tampering == "forged_anchor":
        anchor = anchor_for(original, signing_key=FORGED_ANCHOR_KEY)
    elif tampering == "cross_job_anchor_mismatch":
        anchor = anchor_for(envelope_for(f"{case_id}-other-job", "jpeg"))
    else:
        raise ValueError(f"unsupported detached anchor mutation: {tampering}")
    queue_bytes = queue_envelope.to_json_bytes()
    anchor_bytes = anchor.to_json_bytes()
    if LocalDispatchEnvelope.from_json_bytes(queue_bytes) != queue_envelope:
        raise ValueError(f"{case_id} did not produce an internally valid queue envelope")
    rejection = run_rejected_worker(
        queue_bytes,
        anchor_bytes,
        case_id=case_id,
        worker_id=f"fresh-worker-{case_id}-1",
    )
    return {
        "attempts": [],
        "case_id": case_id,
        "category": "post_contract_detached_anchor_tampering",
        "contract_sha256": queue_envelope.contract.sha256,
        "correlation_id": queue_envelope.correlation_id,
        "detached_anchor_sha256": sha256(anchor_bytes),
        "envelope_created_after_materialization": True,
        "internally_valid_queue_envelope": True,
        "guard_transport_writes": 0,
        "loopback_http_requests": 0,
        "outcome": "detached_anchor_rejected",
        "queue_message_sha256": queue_envelope.queue_message_sha256,
        "reason": rejection["reason"],
        "sdk_serializations": 0,
        "sink_writes": 0,
        "stage": rejection["stage"],
        "tampering": tampering,
        "worker_process_boundary": rejection["worker_process_boundary"],
    }


def sdk_body_case(
    case_id: str,
    body_mode: Literal["text_only", "wrong_task_text", "wrong_turn"],
    *,
    client_response_binding_mutation: Literal["active_field"] | None = None,
) -> dict[str, Any]:
    envelope = envelope_for(case_id, "png")
    anchor_bytes = anchor_for(envelope).to_json_bytes()
    restored, attempt, response = run_worker(
        envelope.to_json_bytes(),
        anchor_bytes,
        case_id=case_id,
        worker_id=f"fresh-worker-{case_id}-1",
        body_mode=body_mode,
        upstream_mode="bound",
        client_response_binding_mutation=client_response_binding_mutation,
    )
    if response is not None or attempt["guard_decision"] != "blocked":
        raise ValueError(f"{case_id} was not stopped by the SDK-body guard")
    if attempt["loopback_http_requests"] != 0:
        raise ValueError(f"{case_id} reached the loopback endpoint")
    return {
        "attempts": [attempt],
        "case_id": case_id,
        "category": "post_contract_sdk_body_tampering",
        "contract_sha256": restored.contract.sha256,
        "correlation_id": restored.correlation_id,
        "detached_anchor_sha256": sha256(anchor_bytes),
        "envelope_created_after_materialization": True,
        "guard_transport_writes": 1,
        "loopback_http_requests": 0,
        "outcome": "blocked",
        "queue_message_sha256": restored.queue_message_sha256,
        "reason": attempt["guard_reason"],
        "sdk_serializations": 1,
        "sink_writes": 0,
        "stage": "sdk_body_guard",
        "tampering": (
            body_mode
            if client_response_binding_mutation is None
            else "active_field"
        ),
    }


def post_guard_case(
    case_id: str,
    parser_mode: Literal["bound", "parser_drift"],
    *,
    response_binding_mutation: Literal["contract_sha256"] | None = None,
) -> dict[str, Any]:
    envelope = envelope_for(case_id, "png")
    anchor_bytes = anchor_for(envelope).to_json_bytes()
    restored, attempt, response = run_worker(
        envelope.to_json_bytes(),
        anchor_bytes,
        case_id=case_id,
        worker_id=f"fresh-worker-{case_id}-1",
        body_mode="bound",
        upstream_mode="bound",
        response_binding_mutation=response_binding_mutation,
    )
    if response is None or attempt["guard_decision"] != "forwarded":
        raise ValueError(f"{case_id} did not reach the loopback endpoint")
    try:
        parse_after_guard(
            restored,
            response,
            parser_mode,
            expected_dispatch_body_sha256=attempt["sdk_body_sha256"],
        )
    except ParserContractMismatch as error:
        reason = str(error)
    else:
        raise ValueError(f"{case_id} unexpectedly reached the sink")
    return {
        "attempts": [attempt],
        "case_id": case_id,
        "category": "post_guard_response_or_parser_drift",
        "contract_sha256": restored.contract.sha256,
        "correlation_id": restored.correlation_id,
        "detached_anchor_sha256": sha256(anchor_bytes),
        "envelope_created_after_materialization": True,
        "guard_transport_writes": 1,
        "loopback_http_requests": 1,
        "outcome": "parser_rejected",
        "parser_invocations": 1,
        "queue_message_sha256": restored.queue_message_sha256,
        "reason": reason,
        "sdk_serializations": 1,
        "sink_writes": 0,
        "stage": "post_guard_bound_parser",
        "tampering": (
            "client_response_binding"
            if response_binding_mutation is not None
            else "parser_field"
        ),
    }


def response_association_pair_case(sink: SinkSpy) -> dict[str, Any]:
    envelopes = response_pair_envelopes()
    completions: list[
        tuple[LocalDispatchEnvelope, dict[str, Any], ClientOwnedResponseEnvelope]
    ] = []
    for label, envelope in zip(("a", "b"), envelopes, strict=True):
        restored, attempt, response = run_worker(
            envelope.to_json_bytes(),
            anchor_for(envelope).to_json_bytes(),
            case_id=f"response-pair-{label}",
            worker_id=f"fresh-worker-response-pair-{label}",
            body_mode="bound",
            upstream_mode="bound",
        )
        if response is None or attempt["guard_decision"] != "forwarded":
            raise ValueError(f"response pair {label} did not complete")
        completions.append((restored, attempt, response))

    selected = [
        parse_after_guard(
            envelope,
            response,
            "bound",
            expected_dispatch_body_sha256=attempt["sdk_body_sha256"],
        )
        for envelope, attempt, response in completions
    ]
    if selected[0].value == selected[1].value:
        raise ValueError("response pair payloads are not distinguishable")
    for parsed in selected:
        sink.write(parsed.correlation_id, parsed.field_name, parsed.value)

    (envelope_a, attempt_a, response_a), (
        envelope_b,
        attempt_b,
        response_b,
    ) = completions
    attacks = (
        ("complete-b-to-parser-a", envelope_a, attempt_a, response_b),
        ("complete-a-to-parser-b", envelope_b, attempt_b, response_a),
        (
            "payload-b-with-origin-receipt-under-a",
            envelope_a,
            attempt_a,
            ClientOwnedResponseEnvelope.from_trusted_completion(
                response_a.binding,
                response_b.provider_payload,
                response_b.receipt,
            ),
        ),
        (
            "payload-b-with-destination-receipt-under-a",
            envelope_a,
            attempt_a,
            ClientOwnedResponseEnvelope.from_trusted_completion(
                response_a.binding,
                response_b.provider_payload,
                response_a.receipt,
            ),
        ),
    )
    swap_results = []
    for variant, parser_envelope, parser_attempt, swapped_response in attacks:
        try:
            parse_after_guard(
                parser_envelope,
                swapped_response,
                "bound",
                expected_dispatch_body_sha256=parser_attempt["sdk_body_sha256"],
            )
        except ParserContractMismatch as error:
            swap_results.append(
                {"outcome": "parser_rejected", "reason": str(error), "variant": variant}
            )
        else:
            raise ValueError(f"response association attack passed: {variant}")

    return {
        "attempts": [attempt_a, attempt_b],
        "case_id": "response-association-pair",
        "category": "trusted_response_association",
        "contract_sha256": envelope_a.contract.sha256,
        "correlation_ids": [envelope_a.correlation_id, envelope_b.correlation_id],
        "distinct_provider_response_ids": len(
            {response_a.receipt.provider_response_id, response_b.receipt.provider_response_id}
        ),
        "envelope_created_after_materialization": True,
        "matched_parses": 2,
        "outcome": "matched_forwarded_swaps_rejected",
        "parser_invocations": 6,
        "queue_message_sha256": [
            envelope_a.queue_message_sha256,
            envelope_b.queue_message_sha256,
        ],
        "response_association_faults": 4,
        "response_association_rejections": len(swap_results),
        "sdk_serializations": 2,
        "sink_writes": 2,
        "stage": "trusted_completion_and_bound_parser",
        "swap_extractions": 0,
        "swap_results": swap_results,
        "swap_sink_writes": 0,
    }


def outside_detection_boundary_cases() -> list[dict[str, Any]]:
    try:
        materialize(EvidencePart(b"not-a-materialized-image", "image/png"))
    except ValueError:
        failed_materialization = {
            "attempts": [],
            "case_id": "outside-failed-materialization",
            "category": "pre_contract",
            "envelope_created": False,
            "guard_transport_writes": 0,
            "loopback_http_requests": 0,
            "outcome": "outside_detection_boundary",
            "sdk_serializations": 0,
            "sink_writes": 0,
            "stage": "pre_contract_materialization",
        }
    else:
        raise ValueError("invalid in-memory fixture unexpectedly materialized")
    wrong_association = EvidenceFieldContract(
        task_id="local-client-png-associated-task",
        task_text="Describe the PNG-associated evidence.",
        images=materialized_images("jpeg"),
        active_field="primary_diagnosis",
        contract_version=1,
    )
    envelope = LocalDispatchEnvelope("local-correlation-outside-association", wrong_association)
    if LocalDispatchEnvelope.from_json_bytes(envelope.to_json_bytes()) != envelope:
        raise ValueError("internally consistent pre-contract association changed")
    return [
        failed_materialization,
        {
            "attempts": [],
            "case_id": "outside-internally-consistent-wrong-association",
            "category": "pre_contract",
            "contract_sha256": wrong_association.sha256,
            "correlation_id": envelope.correlation_id,
            "envelope_created": True,
            "fixture_mime_types": ["image/jpeg"],
            "guard_transport_writes": 0,
            "loopback_http_requests": 0,
            "outcome": "outside_detection_boundary",
            "queue_message_sha256": envelope.queue_message_sha256,
            "sdk_serializations": 0,
            "sink_writes": 0,
            "stage": "pre_contract_association",
        },
    ]


def run_connected(endpoint: LoopbackEndpoint) -> dict[str, Any]:
    sink = SinkSpy()
    ledger = outside_detection_boundary_cases()
    ledger.extend(
        (
            control_case("control-png-retry", "png", sink, retry=True),
            control_case("control-jpeg", "jpeg", sink),
            control_case("control-webp", "webp", sink),
            response_association_pair_case(sink),
            queue_tampering_case("queue-dropped-bytes", "dropped_bytes", "png"),
            queue_tampering_case(
                "queue-replaced-bytes-digest", "replaced_bytes_and_digest", "png"
            ),
            queue_tampering_case("queue-caller-mime", "caller_mime", "png"),
            queue_tampering_case("queue-task-text", "task_text", "png"),
            queue_tampering_case("queue-image-order", "image_order", "two-png"),
            detached_anchor_case(
                "anchor-coherent-rewrite", "coherent_rewrite"
            ),
            detached_anchor_case("anchor-forged-mac", "forged_anchor"),
            detached_anchor_case(
                "anchor-cross-job-mismatch", "cross_job_anchor_mismatch"
            ),
            sdk_body_case("sdk-text-only", "text_only"),
            sdk_body_case("sdk-wrong-task-text", "wrong_task_text"),
            sdk_body_case("sdk-wrong-turn", "wrong_turn"),
            sdk_body_case(
                "sdk-active-field",
                "bound",
                client_response_binding_mutation="active_field",
            ),
            post_guard_case(
                "post-guard-client-response-binding",
                "bound",
                response_binding_mutation="contract_sha256",
            ),
            post_guard_case("post-guard-parser-field", "parser_drift"),
        )
    )
    controls = [row for row in ledger if row["category"] == "valid_fixture_control"]
    attempts = [attempt for row in ledger for attempt in row["attempts"]]
    retry_row = next(row for row in controls if row["case_id"] == "control-png-retry")
    retry_values = {
        key: {attempt[key] for attempt in retry_row["attempts"]}
        for key in (
            "anchor_key_id",
            "correlation_id",
            "contract_sha256",
            "detached_anchor_sha256",
            "queue_message_sha256",
            "sdk_body_sha256",
        )
    }
    if any(len(values) != 1 for values in retry_values.values()):
        raise ValueError("retry handoff changed a stable identity")
    if len({attempt["worker_id"] for attempt in retry_row["attempts"]}) != 2:
        raise ValueError("retry did not use fresh workers")
    if not all(attempt["worker_process_boundary"] for attempt in attempts):
        raise ValueError("an SDK attempt did not cross the worker process boundary")
    if len(sink.writes) != 3:
        raise ValueError("local audit requires three post-parser sink writes")
    response_pair = next(
        row for row in ledger if row["category"] == "trusted_response_association"
    )
    structural_counts = {
        "blocked_sdk_body_cases": sum(
            row["category"] == "post_contract_sdk_body_tampering" for row in ledger
        ),
        "detached_anchor_rejections": sum(
            row["category"] == "post_contract_detached_anchor_tampering"
            for row in ledger
        ),
        "fixture_false_positive_count": sum(
            row["outcome"] != "forwarded" for row in controls
        ),
        "fresh_worker_processes": len(attempts)
        + sum(
            row["category"] == "post_contract_detached_anchor_tampering"
            for row in ledger
        ),
        "guard_transport_writes": sum(
            attempt["guard_transport_writes"] for attempt in attempts
        ),
        "loopback_http_requests": sum(
            attempt["loopback_http_requests"] for attempt in attempts
        ),
        "outside_detection_boundary_cases": sum(
            row["outcome"] == "outside_detection_boundary" for row in ledger
        ),
        "physical_attempts": len(attempts),
        "post_guard_parser_rejections": sum(
            row["category"] == "post_guard_response_or_parser_drift" for row in ledger
        ),
        "post_contract_cases": sum(
            row["category"].startswith("post_contract")
            or row["category"] == "post_guard_response_or_parser_drift"
            for row in ledger
        ),
        "post_contract_sink_writes": sum(
            row["sink_writes"]
            for row in ledger
            if row["category"].startswith("post_contract")
            or row["category"] == "post_guard_response_or_parser_drift"
        ),
        "queue_envelope_rejections": sum(
            row["category"] == "post_contract_queue_tampering" for row in ledger
        ),
        "response_association_faults": response_pair["response_association_faults"],
        "response_association_rejections": response_pair[
            "response_association_rejections"
        ],
        "response_pair_matched_parses": response_pair["matched_parses"],
        "response_swap_extractions": response_pair["swap_extractions"],
        "response_swap_sink_writes": response_pair["swap_sink_writes"],
        "sdk_serializations": sum(row["sdk_serializations"] for row in ledger),
        "sink_writes": len(sink.writes),
        "trusted_response_receipts": sum(
            "response_receipt_mac_sha256" in attempt for attempt in attempts
        ),
        "valid_fixture_controls": len(controls),
        "blocked_fixture_controls": sum(
            row["outcome"] == "blocked" for row in controls
        ),
    }
    return {
        "audit_name": "deterministic loopback client queue handoff audit",
        "audit_schema_version": 4,
        "ledger": ledger,
        "network_calls": endpoint.request_count,
        "model_calls": 0,
        "scope": (
            "This deterministic loopback audit exercises current OpenAI SDK request "
            "serialization through a guarded transport and a real local TCP/HTTP endpoint, "
            "canonical local queue envelopes, "
            "a detached HMAC-SHA256 producer anchor whose verification key is supplied to "
            "the worker outside the queue message, the exact-body GuardedTransport, "
            "semantic response-schema binding, an authenticated trusted-adapter receipt "
            "over the dispatch-body and provider-payload digests, and the bound response "
            "parser. The response pair tests two matched completions plus four whole-envelope "
            "or raw-payload swap attacks. The public fixed keys are reproducibility fixtures, "
            "not production secrets or key-management tests. "
            "The two pre-contract examples are outside the detection boundary; this audit "
            "does not establish production effectiveness, semantic association correctness "
            "before contract creation, hosted provider receipt, or model behavior."
        ),
        "sink_audit": {"post_parser_writes": sink.writes},
        "structural_counts": structural_counts,
    }


def run() -> dict[str, Any]:
    previous = os.environ.get(LOOPBACK_URL_ENV)
    with loopback_endpoint() as endpoint:
        os.environ[LOOPBACK_URL_ENV] = endpoint.base_url
        try:
            return run_connected(endpoint)
        finally:
            if previous is None:
                os.environ.pop(LOOPBACK_URL_ENV, None)
            else:
                os.environ[LOOPBACK_URL_ENV] = previous


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raw = sys.stdin.buffer.read()
        worker_input = json.loads(raw)
        if canonical_json_bytes(worker_input) != raw:
            raise ValueError("worker input is not canonical JSON")
        queue_bytes = base64.b64decode(
            worker_input["queue_bytes_base64"], validate=True
        )
        anchor_bytes = base64.b64decode(
            worker_input["anchor_bytes_base64"], validate=True
        )
        try:
            _envelope, attempt, response = run_worker_local(
                queue_bytes,
                anchor_bytes,
                case_id=worker_input["case_id"],
                worker_id=worker_input["worker_id"],
                body_mode=worker_input["body_mode"],
                upstream_mode=worker_input["upstream_mode"],
                client_response_binding_mutation=worker_input[
                    "client_response_binding_mutation"
                ],
            )
        except DispatchAnchorMismatch as error:
            sys.stdout.buffer.write(
                canonical_json_bytes(
                    {
                        "rejection": {
                            "guard_transport_writes": 0,
                            "loopback_http_requests": 0,
                            "reason": str(error),
                            "sdk_serializations": 0,
                            "sink_writes": 0,
                            "stage": "detached_producer_anchor_verification",
                            "worker_id": worker_input["worker_id"],
                        }
                    }
                )
            )
            return
        sys.stdout.buffer.write(canonical_json_bytes({"attempt": attempt, "response": response}))
        return
    assert args.out is not None
    if not args.out.parent.is_dir():
        parser.error(f"output directory does not exist: {args.out.parent}")
    audit = run()
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit["structural_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
