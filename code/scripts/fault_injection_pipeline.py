from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


FAIL_CLOSED_RESPONSE = {
    "image_present": False,
    "can_diagnose": False,
    "primary_diagnosis": None,
    "differentials": [],
    "confidence": 0,
    "key_findings": "",
    "reasoning": "",
}


@dataclass(frozen=True)
class PipelineState:
    scenario: str
    stage: str
    rendered_attachment_count: int
    attachment_metadata: tuple[str, ...]
    image_bytes: tuple[bytes, ...]
    blob_pointer: str | None = None
    blob_available: bool | None = None
    signed_url_valid: bool | None = None
    retry_mode: str | None = None


@dataclass(frozen=True)
class SerializedRequest:
    rendered_prompt: str
    attachment_metadata: tuple[str, ...]
    image_parts: tuple[str, ...]
    serialized_image_byte_count: int
    serialized_body_sha256: str
    serialized_body: str


@dataclass(frozen=True)
class PreflightDecision:
    allowed: bool
    reason: str
    client_status: str
    response: dict | None


class ModelInvocationSpy:
    def __init__(self) -> None:
        self.calls: list[SerializedRequest] = []

    def invoke(self, request: SerializedRequest) -> dict:
        self.calls.append(request)
        return {"unexpected": "model invocation occurred"}


def initial_state(scenario: str, stage: str) -> PipelineState:
    return PipelineState(
        scenario=scenario,
        stage=stage,
        rendered_attachment_count=1,
        attachment_metadata=("image-1",),
        image_bytes=(b"synthetic-image-bytes",),
        blob_pointer="blob://image-1",
        blob_available=True,
        signed_url_valid=True,
    )


def partial_upload_failure() -> tuple[PipelineState, ...]:
    start = initial_state("partial_upload_failure", "ui_committed_attachment")
    end = replace(
        start,
        stage="upload_failed_after_ui_commit",
        image_bytes=(),
        blob_pointer=None,
        blob_available=False,
    )
    return start, end


def retry_reconstruction_loses_binary() -> tuple[PipelineState, ...]:
    start = initial_state("retry_reconstruction_loses_binary", "first_request_has_bytes")
    end = replace(
        start,
        stage="retry_rebuilt_from_text_metadata",
        image_bytes=(),
        retry_mode="text_metadata_only",
    )
    return start, end


def adapter_drops_binary() -> tuple[PipelineState, ...]:
    start = initial_state("adapter_drops_binary", "adapter_input_has_bytes")
    end = replace(start, stage="adapter_output_text_only", image_bytes=())
    return start, end


def agent_handoff_text_only() -> tuple[PipelineState, ...]:
    start = initial_state("agent_handoff_text_only", "originating_agent_has_bytes")
    end = replace(start, stage="handoff_forwards_rendered_text_only", image_bytes=())
    return start, end


def stale_ui_attachment_count() -> tuple[PipelineState, ...]:
    start = initial_state("stale_ui_attachment_count", "session_has_uploaded_blob")
    end = replace(
        start,
        stage="session_restored_without_blob",
        image_bytes=(),
        blob_available=False,
    )
    return start, end


def queue_blob_loss() -> tuple[PipelineState, ...]:
    start = initial_state("queue_blob_loss", "queue_enqueued_with_blob")
    end = replace(
        start,
        stage="queue_worker_cannot_resolve_blob",
        image_bytes=(),
        blob_available=False,
    )
    return start, end


def signed_url_expired() -> tuple[PipelineState, ...]:
    start = initial_state("signed_url_expired", "signed_url_resolves")
    end = replace(
        start,
        stage="signed_url_resolution_expired",
        image_bytes=(),
        signed_url_valid=False,
    )
    return start, end


def sdk_text_only_retry() -> tuple[PipelineState, ...]:
    start = initial_state("sdk_text_only_retry", "sdk_first_attempt_has_image_part")
    end = replace(
        start,
        stage="sdk_retry_copies_text_parts_only",
        image_bytes=(),
        retry_mode="text_parts_only",
    )
    return start, end


SCENARIOS: tuple[Callable[[], tuple[PipelineState, ...]], ...] = (
    partial_upload_failure,
    retry_reconstruction_loses_binary,
    adapter_drops_binary,
    agent_handoff_text_only,
    stale_ui_attachment_count,
    queue_blob_loss,
    signed_url_expired,
    sdk_text_only_retry,
)


def render_prompt(state: PipelineState) -> str:
    return (
        f"number of image attachments: {state.rendered_attachment_count}\n"
        "Describe this chest X ray and provide the final diagnosis."
    )


def snapshot(state: PipelineState) -> dict:
    return {
        "scenario": state.scenario,
        "stage": state.stage,
        "rendered_attachment_count": state.rendered_attachment_count,
        "attachment_metadata": list(state.attachment_metadata),
        "image_byte_count": sum(len(chunk) for chunk in state.image_bytes),
        "blob_pointer": state.blob_pointer,
        "blob_available": state.blob_available,
        "signed_url_valid": state.signed_url_valid,
        "retry_mode": state.retry_mode,
    }


def serialize_request(state: PipelineState) -> SerializedRequest:
    image_parts = tuple(chunk.hex() for chunk in state.image_bytes if chunk)
    body = {
        "messages": [{"role": "user", "content": render_prompt(state)}],
        "attachment_metadata": list(state.attachment_metadata),
        "image_parts": list(image_parts),
    }
    serialized = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return SerializedRequest(
        rendered_prompt=render_prompt(state),
        attachment_metadata=state.attachment_metadata,
        image_parts=image_parts,
        serialized_image_byte_count=sum(len(chunk) for chunk in state.image_bytes if chunk),
        serialized_body_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        serialized_body=serialized,
    )


def byte_first_preflight(request: SerializedRequest) -> PreflightDecision:
    if request.serialized_image_byte_count == 0 or not request.image_parts:
        return PreflightDecision(
            False,
            "missing_image_bytes",
            "blocked_missing_evidence",
            dict(FAIL_CLOSED_RESPONSE),
        )
    return PreflightDecision(True, "image_bytes_present", "passed_presence_check", None)


def invoke_with_preflight(
    request: SerializedRequest, model: ModelInvocationSpy
) -> tuple[dict, PreflightDecision]:
    decision = byte_first_preflight(request)
    if not decision.allowed:
        assert decision.response is not None
        return decision.response, decision
    return model.invoke(request), decision


def execute_fault_matrix() -> list[dict]:
    records: list[dict] = []
    for build_trace in SCENARIOS:
        states = build_trace()
        assert len(states) >= 2
        assert sum(len(chunk) for chunk in states[0].image_bytes) > 0
        assert sum(len(chunk) for chunk in states[-1].image_bytes) == 0
        trace = [snapshot(state) for state in states]
        trace_json = json.dumps(trace, sort_keys=True, separators=(",", ":"))
        request = serialize_request(states[-1])
        model = ModelInvocationSpy()
        response, decision = invoke_with_preflight(request, model)
        records.append(
            {
                "scenario": states[-1].scenario,
                "pathway_trace": trace,
                "pathway_trace_sha256": hashlib.sha256(trace_json.encode()).hexdigest(),
                "rendered_attachment_count": states[-1].rendered_attachment_count,
                "initial_image_byte_count": trace[0]["image_byte_count"],
                "serialized_image_byte_count": request.serialized_image_byte_count,
                "serialized_body_sha256": request.serialized_body_sha256,
                "serialized_body": request.serialized_body,
                "preflight_allowed": decision.allowed,
                "preflight_reason": decision.reason,
                "client_status": decision.client_status,
                "model_invocation_count": len(model.calls),
                "response": response,
                "full_null_response": response == FAIL_CLOSED_RESPONSE,
            }
        )
    return records


def write_outputs(out_root: Path) -> None:
    records = execute_fault_matrix()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "fault_injection_traces.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "| Scenario | Initial bytes | Final bytes | Client status | Model calls | Full-null response |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in records:
        lines.append(
            f"| {row['scenario']} | {row['initial_image_byte_count']} | "
            f"{row['serialized_image_byte_count']} | {row['client_status']} | "
            f"{row['model_invocation_count']} | {str(row['full_null_response']).lower()} |"
        )
    (out_root / "fault_injection_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build/fault_injection",
    )
    args = parser.parse_args()
    write_outputs(args.out_root)
    print(f"wrote {args.out_root}")


if __name__ == "__main__":
    main()
