#!/usr/bin/env python3
"""Score four request guards on released loss and source-regression cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
DATA_ROOT = ARTIFACT_ROOT / "data"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_enforcement_baselines as baselines
import audit_independent_client_sources as source_regressions


Decision = tuple[bool, str]
Guard = Callable[[baselines.CorpusCase, baselines.RequestView], Decision]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def simple_presence(
    _case: baselines.CorpusCase, view: baselines.RequestView
) -> Decision:
    if view.global_references or not view.global_images:
        return False, "no_inline_image"
    return True, "inline_image_present"


GUARDS: tuple[tuple[str, Guard], ...] = (
    ("simple_presence", simple_presence),
    ("decoded_presence", baselines.released_decoded_presence_gate),
    ("body_local_consistency", baselines.released_body_only_gate),
    ("caller_owned_binding", baselines.caller_owned_binding),
)


def expected_binding(
    task_id: str, task_text: str, mime_type: str, image_bytes: bytes
) -> baselines.ExpectedBinding:
    image = baselines.ExpectedImage(mime_type, sha256_bytes(image_bytes))
    commitment = {
        "active_field": "primary_diagnosis",
        "contract_version": 1,
        "ordered_image_mime_types": [mime_type],
        "ordered_image_sha256": [image.sha256],
        "task_id": task_id,
        "task_text_sha256": sha256_bytes(task_text.encode("utf-8")),
    }
    response_binding = {
        "task_id": task_id,
        "task_text_sha256": commitment["task_text_sha256"],
        "contract_sha256": sha256_bytes(canonical_json_bytes(commitment)),
        "contract_version": 1,
        "active_field": "primary_diagnosis",
        "correlation_id": f"boundary-comparison/{task_id}",
    }
    return baselines.ExpectedBinding(
        client_response_binding=response_binding,
        expected_response_binding=response_binding,
        images=(image,),
        response_schema_required=False,
        serialized_response_schema=None,
        task_text=task_text,
    )


def score_case(
    case: baselines.CorpusCase, view: baselines.RequestView
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, guard in GUARDS:
        allowed, reason = guard(case, view)
        result[name] = {
            "decision": "forwarded" if allowed else "blocked",
            "reason": reason,
        }
    return result


def loss_path_audit() -> dict[str, Any]:
    path = DATA_ROOT / "fault_injection_traces.json"
    traces = json.loads(path.read_text())
    if not isinstance(traces, list) or len(traces) != 8:
        raise RuntimeError("loss-path denominator drift")
    image_bytes = b"synthetic-image-bytes"
    rows = []
    for trace in traces:
        body = str(trace["serialized_body"]).encode("utf-8")
        if sha256_bytes(body) != trace["serialized_body_sha256"]:
            raise RuntimeError(f"loss-path body drift: {trace['scenario']}")
        parsed = json.loads(body)
        task_text = str(parsed["messages"][0]["content"])
        task_id = f"loss-path-{trace['scenario'].replace('_', '-')}"
        expected = expected_binding(task_id, task_text, "image/png", image_bytes)
        case = baselines.CorpusCase(
            audit_case_id=str(trace["scenario"]),
            body=body,
            expected=expected,
            is_control=False,
            registry={expected.task_id: expected},
            sdk_family="openai",
            source_case_id=str(trace["scenario"]),
        )
        rows.append(
            {
                "body_sha256": trace["serialized_body_sha256"],
                "decisions": score_case(case, baselines._request_view(case)),
                "scenario": trace["scenario"],
            }
        )
    totals = {
        name: {
            "faults_blocked": sum(
                row["decisions"][name]["decision"] == "blocked" for row in rows
            ),
            "faults_forwarded": sum(
                row["decisions"][name]["decision"] == "forwarded" for row in rows
            ),
        }
        for name, _ in GUARDS
    }
    return {
        "cases": rows,
        "scope": {
            "controls": 0,
            "faults": 8,
            "frozen_body_hashes": len({row["body_sha256"] for row in rows}),
            "note": (
                "These are eight pathway traces over one repeated final text-only "
                "body shape. They test fault blocking only and do not estimate false blocks."
            ),
        },
        "totals": totals,
    }


def expected_source_keys(audit: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    expected: dict[tuple[str, str], dict[str, str]] = {}
    projects = audit["projects"]
    for row in projects["open_webui"]["cases"]:
        for label, version in (("pre_fix", "v0.8.2"), ("post_fix", "v0.8.3")):
            audit_id = f"{row['case_id']}-{version}"
            digest = row[version]["body_sha256"]
            expected[(audit_id, digest)] = {
                "case_id": row["case_id"],
                "project": "open_webui",
                "stage": label,
            }
    for project, prefix in (("google_adk", "google-adk"), ("goose", "goose")):
        for row in projects[project]["cases"]:
            for label, source_label in (("pre_fix", "before"), ("post_fix", "after")):
                audit_id = f"{prefix}/{source_label}/{row['case_id']}"
                digest = row[source_label]["body_sha256"]
                expected[(audit_id, digest)] = {
                    "case_id": row["case_id"],
                    "project": project,
                    "stage": label,
                }
    for row in projects["openclaw"]["source_execution"]["cases"]:
        for label, source_label in (("pre_fix", "parent"), ("post_fix", "fixed")):
            digest = row[source_label]["command_input_sha256"]
            expected[(row["case_id"], digest)] = {
                "case_id": row["case_id"],
                "project": "openclaw",
                "stage": label,
            }
    return expected


def source_regression_audit() -> dict[str, Any]:
    captured: list[tuple[baselines.CorpusCase, baselines.RequestView]] = []
    original = baselines.caller_owned_binding

    def capture(
        case: baselines.CorpusCase, view: baselines.RequestView
    ) -> Decision:
        captured.append((case, view))
        return original(case, view)

    baselines.caller_owned_binding = capture
    try:
        source_audit = source_regressions.build_audit()
    finally:
        baselines.caller_owned_binding = original
    expected = expected_source_keys(source_audit)
    if len(expected) != 20 or len(captured) != 20:
        raise RuntimeError(
            f"source-regression denominator drift: {len(expected)} expected, "
            f"{len(captured)} captured"
        )
    rows = []
    seen: set[tuple[str, str]] = set()
    for case, view in captured:
        digest = sha256_bytes(canonical_json_bytes(view.body))
        key = (case.audit_case_id, digest)
        if key not in expected or key in seen:
            raise RuntimeError(f"unmatched source projection: {key}")
        seen.add(key)
        metadata = expected[key]
        rows.append(
            {
                **metadata,
                "audit_case_id": case.audit_case_id,
                "body_sha256": digest,
                "decisions": score_case(case, view),
            }
        )
    if seen != set(expected):
        raise RuntimeError("source-regression body commitments are incomplete")
    rows.sort(key=lambda row: (row["project"], row["case_id"], row["stage"]))
    totals: dict[str, dict[str, int]] = {}
    for name, _ in GUARDS:
        by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_stage[row["stage"]].append(row)
        totals[name] = {
            "post_fix_blocked": sum(
                row["decisions"][name]["decision"] == "blocked"
                for row in by_stage["post_fix"]
            ),
            "post_fix_forwarded": sum(
                row["decisions"][name]["decision"] == "forwarded"
                for row in by_stage["post_fix"]
            ),
            "pre_fix_blocked": sum(
                row["decisions"][name]["decision"] == "blocked"
                for row in by_stage["pre_fix"]
            ),
            "pre_fix_forwarded": sum(
                row["decisions"][name]["decision"] == "forwarded"
                for row in by_stage["pre_fix"]
            ),
        }
    return {
        "cases": rows,
        "scope": {
            "independent_projects": 4,
            "post_fix_controls": 10,
            "pre_fix_faults": 10,
            "source_audit_sha256": sha256_bytes(
                (DATA_ROOT / "independent_client_source_audit.json").read_bytes()
            ),
        },
        "totals": totals,
    }


def build_audit() -> dict[str, Any]:
    return {
        "audit_schema_version": 1,
        "caveats": [
            "The eight pathway traces share one repeated final body shape and have no matched controls in this audit.",
            "The source cases are exact released pre/post projections. They are not full application executions or an incident-frequency sample.",
            "Simple presence checks only for an inline image part. Decoded presence additionally validates bytes and MIME. Body-local consistency trusts the task marker inside the body. Caller-owned binding compares the body with external caller state.",
        ],
        "execution": {
            "credentials_required": False,
            "network_calls": 0,
            "provider_or_model_calls": 0,
        },
        "guard_definitions": {
            "body_local_consistency": "Select expected state from the task marker inside the body, then compare count, order, MIME, and digest.",
            "caller_owned_binding": "Compare the active turn with caller-owned task, evidence, and field state.",
            "decoded_presence": "Require at least one inline image and validate decoding plus declared and decoded MIME agreement.",
            "simple_presence": "Require at least one inline image part without decoding or task binding.",
        },
        "loss_paths": loss_path_audit(),
        "source_regressions": source_regression_audit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "boundary_guard_comparison_audit.json",
    )
    args = parser.parse_args()
    audit = build_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        "boundary guard comparison passed: "
        "8 loss paths and 10 pre/post source pairs scored across 4 guards"
    )


if __name__ == "__main__":
    main()
