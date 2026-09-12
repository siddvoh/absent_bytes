#!/usr/bin/env python3
"""Audit pinned before/after source paths from four independent clients."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
DATA_ROOT = ARTIFACT_ROOT / "data"
THIRD_PARTY_ROOT = ARTIFACT_ROOT / "third_party"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_enforcement_baselines as baselines
import audit_openclaw_holdout as openclaw
import audit_open_webui_source_regression as open_webui


ADK_SOURCE = {
    "before": {
        "commit": "589f15cb273b1b83a87f0c3145880344ed5b1b91",
        "file": "third_party/google_adk/contents_before.py",
        "git_blob_sha1": "39902e85475a48f6ff87c585727779a28a5a173c",
        "sha256": "18b923d8b3898f575099efe20e384febd2cd82d1e685070f69cc2955fd93a67e",
    },
    "after": {
        "commit": "f35d129b4c59d381e95418725d6eaa072ca7720a",
        "file": "third_party/google_adk/contents_after.py",
        "git_blob_sha1": "ce0df37e39bd324b587bb634a54264fd45bc04f0",
        "sha256": "f9fa1ec5526c13b5cfe8f40405d385a178da0976c65823cab00b5d97c49cf340",
    },
}
ADK_FIX_URL = (
    "https://github.com/google/adk-python/commit/"
    "f35d129b4c59d381e95418725d6eaa072ca7720a"
)

GOOSE_SOURCE = {
    "before": {
        "commit": "18bc030e48e032ebf65a7488b316cfbd966c0a16",
        "file": "third_party/goose/openai_before.rs",
        "git_blob_sha1": "c696cb8e34d22415c40973b73de16da3f07ff03c",
        "sha256": "7f20b9a21ba546ad3dfbddb93669d1f2b9c85fd671c814d2387f8c5ff6e7df2b",
    },
    "after": {
        "commit": "984967141376c77f424bcedf4d7ab35756753911",
        "file": "third_party/goose/openai_after.rs",
        "git_blob_sha1": "28420888a01a7448dc68e839e774e17f50dbb23e",
        "sha256": "2b024fecbbc99f51a2890ff19fc83e5887d09cb757edbb283817e0f2a402d9f2",
    },
}
GOOSE_ISSUE_URL = "https://github.com/aaif-goose/goose/issues/8067"
GOOSE_FIX_URL = (
    "https://github.com/aaif-goose/goose/commit/"
    "984967141376c77f424bcedf4d7ab35756753911"
)

PNG_BYTES = open_webui.PNG_BYTES
JPEG_BYTES = open_webui.JPEG_BYTES


class SourceDriftError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode("ascii") + value).hexdigest()


def load_source(provenance: Mapping[str, str]) -> tuple[Path, str]:
    path = ARTIFACT_ROOT / provenance["file"]
    value = path.read_bytes()
    if sha256_bytes(value) != provenance["sha256"]:
        raise SourceDriftError(f"SHA-256 drift for {provenance['file']}")
    if git_blob_sha1(value) != provenance["git_blob_sha1"]:
        raise SourceDriftError(f"Git blob drift for {provenance['file']}")
    return path, value.decode("utf-8")


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
        "correlation_id": f"source-regression/{task_id}",
    }
    return baselines.ExpectedBinding(
        client_response_binding=response_binding,
        expected_response_binding=response_binding,
        images=(image,),
        response_schema_required=False,
        serialized_response_schema=None,
        task_text=task_text,
    )


def prompt_for(task_id: str, case_id: str, task_text: str) -> str:
    return "\n".join(
        (
            f"evidence_task_id: {task_id}",
            f"evidence_case_id: {case_id}",
            task_text,
        )
    )


def image_part(mime_type: str, image_bytes: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def evaluate_projected_body(
    case_id: str,
    expected: baselines.ExpectedBinding,
    content: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {"messages": [{"role": "user", "content": content}]}
    body_bytes = canonical_json_bytes(body)
    case = baselines.CorpusCase(
        audit_case_id=case_id,
        body=body_bytes,
        expected=expected,
        is_control=True,
        registry={expected.task_id: expected},
        sdk_family="openai",
        source_case_id=case_id,
    )
    allowed, reason = baselines.caller_owned_binding(
        case, baselines._request_view(case)
    )
    return {
        "body_sha256": sha256_bytes(body_bytes),
        "contract_decision": "forwarded" if allowed else "blocked",
        "contract_reason": reason,
        "guarded_dispatch_invocations": int(allowed),
        "image_parts": sum(part.get("type") == "image_url" for part in content),
        "text_parts": sum(part.get("type") == "text" for part in content),
    }


def compile_adk_runtime(label: str) -> dict[str, Any]:
    path, source = load_source(ADK_SOURCE[label])
    tree = ast.parse(source, filename=str(path))
    names = {"_contains_empty_content"}
    if label == "after":
        names.add("_is_part_invisible")
    nodes = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != names:
        raise SourceDriftError(f"ADK {label} function set drift")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Event": object,
        "types": SimpleNamespace(Part=object),
    }
    exec(compile(module, str(path), "exec"), namespace)
    ast_sha256 = sha256_bytes(
        "\n".join(
            ast.dump(node, include_attributes=False)
            for node in sorted(nodes, key=lambda item: item.name)
        ).encode("utf-8")
    )
    return {
        "contains_empty": namespace["_contains_empty_content"],
        "function_ast_sha256": ast_sha256,
        "path": path,
        "source": source,
    }


def adk_part(**values: Any) -> SimpleNamespace:
    fields = {
        "file_data": None,
        "function_call": None,
        "function_response": None,
        "inline_data": None,
        "text": None,
        "thought": False,
    }
    fields.update(values)
    return SimpleNamespace(**fields)


def adk_event(parts: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        actions=None,
        content=SimpleNamespace(role="user", parts=parts),
        input_transcription=None,
        output_transcription=None,
    )


def run_adk_case(
    runtime: Mapping[str, Any],
    label: str,
    case_id: str,
    mime_type: str,
    image_bytes: bytes,
) -> dict[str, Any]:
    task_id = f"task-{case_id}"
    task_text = "Describe the attached local image."
    prompt = prompt_for(task_id, case_id, task_text)
    event = adk_event(
        [
            adk_part(text=""),
            adk_part(inline_data=SimpleNamespace(data=image_bytes, mime_type=mime_type)),
            adk_part(text=prompt),
        ]
    )
    filtered = bool(runtime["contains_empty"](event))
    content = [] if filtered else [
        image_part(mime_type, image_bytes),
        {"type": "text", "text": prompt},
    ]
    result = evaluate_projected_body(
        f"google-adk/{label}/{case_id}",
        expected_binding(task_id, task_text, mime_type, image_bytes),
        content,
    )
    return {
        **result,
        "event_filtered_as_empty": filtered,
        "event_retained": not filtered,
    }


def build_adk_audit() -> dict[str, Any]:
    before = compile_adk_runtime("before")
    after = compile_adk_runtime("after")
    diff = list(
        difflib.ndiff(before["source"].splitlines(), after["source"].splitlines())
    )
    fixtures = (
        ("google-adk-png", "image/png", PNG_BYTES),
        ("google-adk-jpeg", "image/jpeg", JPEG_BYTES),
    )
    rows = [
        {
            "case_id": case_id,
            "image_sha256": sha256_bytes(image_bytes),
            "mime_type": mime_type,
            "before": run_adk_case(
                before, "before", case_id, mime_type, image_bytes
            ),
            "after": run_adk_case(after, "after", case_id, mime_type, image_bytes),
        }
        for case_id, mime_type, image_bytes in fixtures
    ]
    if any(
        not row["before"]["event_filtered_as_empty"]
        or row["before"]["contract_decision"] != "blocked"
        for row in rows
    ):
        raise SourceDriftError("ADK parent did not reproduce the first-part filter")
    if any(
        not row["after"]["event_retained"]
        or row["after"]["contract_decision"] != "forwarded"
        for row in rows
    ):
        raise SourceDriftError("ADK fix did not retain and forward mixed content")
    return {
        "cases": rows,
        "execution": {
            "body_projection": "author_defined_openai_style",
            "credentials_required": False,
            "exact_upstream_function_ast_executed": True,
            "full_adk_runtime_started": False,
            "network_calls": 0,
            "provider_or_model_calls": 0,
        },
        "source": {
            "fix_url": ADK_FIX_URL,
            "project": "google/adk-python",
            "source_files": ADK_SOURCE,
            "upstream_diff": {
                "deletions": sum(line.startswith("- ") for line in diff),
                "insertions": sum(line.startswith("+ ") for line in diff),
            },
            "before_function_ast_sha256": before["function_ast_sha256"],
            "after_function_ast_sha256": after["function_ast_sha256"],
        },
        "totals": {
            "affected_cases": len(rows),
            "after_contract_forwarded": sum(
                row["after"]["contract_decision"] == "forwarded" for row in rows
            ),
            "after_events_retained": sum(row["after"]["event_retained"] for row in rows),
            "before_contract_blocked": sum(
                row["before"]["contract_decision"] == "blocked" for row in rows
            ),
            "before_events_retained": sum(
                row["before"]["event_retained"] for row in rows
            ),
        },
    }


def extract_rust_function(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise SourceDriftError(f"missing Rust function: {signature}")
    brace = source.find("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise SourceDriftError(f"unterminated Rust function: {signature}")


def goose_semantics(label: str) -> dict[str, Any]:
    path, source = load_source(GOOSE_SOURCE[label])
    function = extract_rust_function(source, "pub fn format_messages")
    old_fragments = (
        "let mut text_array = Vec::new();",
        "text_array.push(text.text.clone());",
        "} else if !text_array.is_empty() {",
    )
    fixed_fragments = (
        "let mut has_non_text_content = false;",
        'content_array.push(json!({"type": "text", "text": text.text}));',
        "if has_non_text_content {",
        "test_format_messages_with_text_and_image_preserves_order",
    )
    if label == "before":
        if not all(fragment in function for fragment in old_fragments):
            raise SourceDriftError("Goose parent branch structure drift")
        if fixed_fragments[0] in function:
            raise SourceDriftError("Goose parent unexpectedly contains fixed branch")
        text_destination = "text_array"
    else:
        if not all(fragment in source for fragment in fixed_fragments):
            raise SourceDriftError("Goose fixed branch structure drift")
        if old_fragments[0] in function:
            raise SourceDriftError("Goose fix unexpectedly retains split text array")
        text_destination = "content_array"
    if "content_array.push(convert_image(image, image_format));" not in function:
        raise SourceDriftError("Goose image branch drift")
    return {
        "function_sha256": sha256_bytes(function.encode("utf-8")),
        "image_destination": "content_array",
        "path": path,
        "source": source,
        "text_destination": text_destination,
    }


def interpret_goose_mixed(
    semantics: Mapping[str, Any], sequence: tuple[str, str]
) -> list[str]:
    buckets: dict[str, list[str]] = {"content_array": [], "text_array": []}
    for item in sequence:
        destination = (
            semantics["text_destination"]
            if item == "text"
            else semantics["image_destination"]
        )
        buckets[destination].append(item)
    if buckets["content_array"]:
        return buckets["content_array"]
    return buckets["text_array"]


def run_goose_case(
    semantics: Mapping[str, Any],
    label: str,
    case_id: str,
    mime_type: str,
    image_bytes: bytes,
    sequence: tuple[str, str],
) -> dict[str, Any]:
    task_id = f"task-{case_id}"
    task_text = "Describe the attached local image."
    prompt = prompt_for(task_id, case_id, task_text)
    emitted = interpret_goose_mixed(semantics, sequence)
    content = [
        {"type": "text", "text": prompt}
        if item == "text"
        else image_part(mime_type, image_bytes)
        for item in emitted
    ]
    result = evaluate_projected_body(
        f"goose/{label}/{case_id}",
        expected_binding(task_id, task_text, mime_type, image_bytes),
        content,
    )
    return {
        **result,
        "emitted_order": emitted,
        "input_order": list(sequence),
        "task_text_retained": "text" in emitted,
    }


def build_goose_audit() -> dict[str, Any]:
    before = goose_semantics("before")
    after = goose_semantics("after")
    diff = list(
        difflib.ndiff(before["source"].splitlines(), after["source"].splitlines())
    )
    fixtures = []
    for image_label, mime_type, image_bytes in (
        ("png", "image/png", PNG_BYTES),
        ("jpeg", "image/jpeg", JPEG_BYTES),
    ):
        for order_label, sequence in (
            ("text-image", ("text", "image")),
            ("image-text", ("image", "text")),
        ):
            fixtures.append(
                (f"goose-{image_label}-{order_label}", mime_type, image_bytes, sequence)
            )
    rows = [
        {
            "case_id": case_id,
            "image_sha256": sha256_bytes(image_bytes),
            "mime_type": mime_type,
            "before": run_goose_case(
                before, "before", case_id, mime_type, image_bytes, sequence
            ),
            "after": run_goose_case(
                after, "after", case_id, mime_type, image_bytes, sequence
            ),
        }
        for case_id, mime_type, image_bytes, sequence in fixtures
    ]
    if any(
        row["before"]["task_text_retained"]
        or row["before"]["contract_decision"] != "blocked"
        for row in rows
    ):
        raise SourceDriftError("Goose parent did not reproduce mixed-message loss")
    if any(
        not row["after"]["task_text_retained"]
        or row["after"]["contract_decision"] != "forwarded"
        or row["after"]["emitted_order"] != row["after"]["input_order"]
        for row in rows
    ):
        raise SourceDriftError("Goose fix did not preserve mixed-message order")
    return {
        "cases": rows,
        "execution": {
            "body_projection": "author_defined_openai_style",
            "credentials_required": False,
            "exact_rust_function_compiled": False,
            "full_goose_runtime_started": False,
            "network_calls": 0,
            "provider_or_model_calls": 0,
            "source_derived_branch_interpretation": True,
        },
        "source": {
            "fix_url": GOOSE_FIX_URL,
            "issue_url": GOOSE_ISSUE_URL,
            "project": "aaif-goose/goose",
            "source_files": GOOSE_SOURCE,
            "upstream_diff": {
                "deletions": sum(line.startswith("- ") for line in diff),
                "insertions": sum(line.startswith("+ ") for line in diff),
            },
            "before_function_sha256": before["function_sha256"],
            "after_function_sha256": after["function_sha256"],
        },
        "totals": {
            "affected_cases": len(rows),
            "after_contract_forwarded": sum(
                row["after"]["contract_decision"] == "forwarded" for row in rows
            ),
            "after_task_text_retained": sum(
                row["after"]["task_text_retained"] for row in rows
            ),
            "before_contract_blocked": sum(
                row["before"]["contract_decision"] == "blocked" for row in rows
            ),
            "before_task_text_retained": sum(
                row["before"]["task_text_retained"] for row in rows
            ),
        },
    }


def build_audit() -> dict[str, Any]:
    open_webui_audit = open_webui.build_audit()
    adk_audit = build_adk_audit()
    goose_audit = build_goose_audit()
    openclaw_audit = openclaw.build_audit()
    openclaw_totals = openclaw_audit["source_execution"]["totals"]
    old_blocked = (
        open_webui_audit["totals"]["old_contract_blocked"]
        + adk_audit["totals"]["before_contract_blocked"]
        + goose_audit["totals"]["before_contract_blocked"]
        + openclaw_totals["parent_contract_blocked"]
    )
    fixed_forwarded = (
        open_webui_audit["totals"]["fixed_contract_forwarded"]
        + adk_audit["totals"]["after_contract_forwarded"]
        + goose_audit["totals"]["after_contract_forwarded"]
        + openclaw_totals["fixed_contract_forwarded"]
    )
    affected_cases = (
        open_webui_audit["totals"]["fixtures"]
        + adk_audit["totals"]["affected_cases"]
        + goose_audit["totals"]["affected_cases"]
        + openclaw_totals["affected_cases"]
    )
    if old_blocked != affected_cases or fixed_forwarded != affected_cases:
        raise SourceDriftError("cross-project contract totals drift")
    return {
        "audit_schema_version": 1,
        "caveats": [
            "Open WebUI and Google ADK execute exact verified Python function ASTs; OpenClaw executes exact verified TypeScript functions and records an upstream gateway regression test; Goose uses a verified source-derived branch interpretation because no Rust toolchain or standalone crate is released.",
            "The ADK and Goose event-to-request projections are author-defined OpenAI-style bodies and are not executions of their full transport stacks.",
            "OpenClaw was selected after the guard freeze as a non-random, non-preregistered holdout; its RequestView adapter was authored after selection and its upstream test uses a loopback gateway with a mocked agent command.",
            "All fixtures are synthetic and local. No full application, model, provider, clinical workflow, or incident-frequency estimate is tested.",
        ],
        "projects": {
            "google_adk": adk_audit,
            "goose": goose_audit,
            "openclaw": openclaw_audit,
            "open_webui": open_webui_audit,
        },
        "totals": {
            "affected_cases": affected_cases,
            "exact_python_ast_projects": 2,
            "exact_typescript_projects": 1,
            "fixed_contract_forwarded": fixed_forwarded,
            "independent_projects": 4,
            "old_contract_blocked": old_blocked,
            "post_freeze_holdout_projects": 1,
            "provider_or_model_calls": 0,
            "source_derived_rust_projects": 1,
        },
    }


def render_tex(audit: Mapping[str, Any]) -> str:
    projects = audit["projects"]
    ow = projects["open_webui"]["totals"]
    adk = projects["google_adk"]["totals"]
    goose = projects["goose"]["totals"]
    openclaw = projects["openclaw"]["source_execution"]["totals"]
    return "\n".join(
        (
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.1pt}",
            r"\caption{\textbf{Independent client source regressions.} Python rows execute exact verified function ASTs. OpenClaw is a non-random, post-freeze holdout using exact TypeScript functions, an upstream gateway test, and a local adapter. The Goose row is source-derived because the released Rust file is not standalone. ADK and Goose request bodies are local projections. No provider is contacted.}",
            r"\label{tab:independent-client-sources}",
            r"\begin{tabular}{@{}p{.18\linewidth}p{.25\linewidth}p{.20\linewidth}p{.20\linewidth}@{}}",
            r"\toprule",
            r"Project & Probe & Before fix & After fix \\",
            r"\midrule",
            f"Open WebUI & exact Python AST & 0/{ow['fixtures']} image parts; {ow['old_contract_blocked']}/{ow['fixtures']} blocked & {ow['fixed_image_parts']}/{ow['fixtures']} image parts; {ow['fixed_contract_forwarded']}/{ow['fixtures']} forwarded \\\\",
            f"Google ADK & exact Python AST & {adk['before_events_retained']}/{adk['affected_cases']} events; {adk['before_contract_blocked']}/{adk['affected_cases']} blocked & {adk['after_events_retained']}/{adk['affected_cases']} events; {adk['after_contract_forwarded']}/{adk['affected_cases']} forwarded \\\\",
            f"Goose & source-derived branch & {goose['before_task_text_retained']}/{goose['affected_cases']} task texts; {goose['before_contract_blocked']}/{goose['affected_cases']} blocked & {goose['after_task_text_retained']}/{goose['affected_cases']} task texts; {goose['after_contract_forwarded']}/{goose['affected_cases']} forwarded \\\\",
            f"OpenClaw & exact TypeScript + gateway test & {openclaw['parent_images_at_agent_command']}/{openclaw['affected_cases']} images; {openclaw['parent_contract_blocked']}/{openclaw['affected_cases']} blocked & {openclaw['fixed_images_at_agent_command']}/{openclaw['affected_cases']} images; {openclaw['fixed_contract_forwarded']}/{openclaw['affected_cases']} forwarded \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "independent_client_source_audit.json",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=ARTIFACT_ROOT / "data" / "derived" / "tables" / "independent_client_sources.tex",
    )
    args = parser.parse_args()
    audit = build_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.tex_out.write_text(render_tex(audit))
    print(
        "independent client source audit passed: "
        f"{audit['totals']['old_contract_blocked']}/"
        f"{audit['totals']['affected_cases']} affected pre-fix projections blocked; "
        f"{audit['totals']['fixed_contract_forwarded']}/"
        f"{audit['totals']['affected_cases']} fixed projections forwarded"
    )


if __name__ == "__main__":
    main()
