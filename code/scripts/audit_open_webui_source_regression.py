#!/usr/bin/env python3
"""Execute the pinned Open WebUI attachment path before and after its upstream fix."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
DATA_ROOT = ARTIFACT_ROOT / "data"
THIRD_PARTY_ROOT = ARTIFACT_ROOT / "third_party" / "open_webui"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_enforcement_baselines as baselines


SOURCE_PROVENANCE = {
    "v0.8.2": {
        "commit": "7c7fe443289c3d0307ebbcf320fb1d895c3ee79b",
        "file": "third_party/open_webui/middleware_v0_8_2.py",
        "git_blob_sha1": "7ab7537de2cae7d0c1806ab7b4665d530010ce3d",
        "sha256": "1ffbbfee1b1975cf7cd868899580e11dc39ffc303ee1e229c7f4260786a4256a",
    },
    "v0.8.3": {
        "commit": "b8112d72b95e480f946f0688bed29321b61e65af",
        "file": "third_party/open_webui/middleware_v0_8_3.py",
        "git_blob_sha1": "ec7af7733bad4ddb558f57c20edac06cf879943e",
        "sha256": "07fad22b951d912c0b436a098a8def0317810e8283a8f88d0f4eb73f9833b5ed",
    },
}
FIX_COMMIT = "f1053d94c7ef7b8b78682dd73586b65a84d202a1"
FIX_PARENT = "10cfddccd7ae017c7e268820b755e4033e88943a"
ISSUE_URL = "https://github.com/open-webui/open-webui/issues/21477"
FIX_URL = f"https://github.com/open-webui/open-webui/commit/{FIX_COMMIT}"

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
JPEG_BYTES = base64.b64decode(
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
)


class SourceDriftError(RuntimeError):
    pass


class FakeChats:
    rows: list[dict[str, Any]] = []

    @classmethod
    def get_messages_map_by_chat_id(cls, _chat_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(cls.rows)


def get_message_list(
    messages_map: list[dict[str, Any]], _message_id: str
) -> list[dict[str, Any]]:
    return copy.deepcopy(messages_map)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def load_source(version: str) -> tuple[Path, str, ast.Module]:
    provenance = SOURCE_PROVENANCE[version]
    path = ARTIFACT_ROOT / provenance["file"]
    source_bytes = path.read_bytes()
    if sha256_bytes(source_bytes) != provenance["sha256"]:
        raise SourceDriftError(f"{version} source SHA-256 drift")
    if git_blob_sha1(source_bytes) != provenance["git_blob_sha1"]:
        raise SourceDriftError(f"{version} Git blob drift")
    source = source_bytes.decode("utf-8")
    return path, source, ast.parse(source, filename=str(path))


def compile_function(
    tree: ast.Module, source_path: Path, name: str, namespace: dict[str, Any]
) -> Callable[..., Any]:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise SourceDriftError(f"expected one {name} definition in {source_path}")
    module = ast.Module(body=[copy.deepcopy(matches[0])], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


def compile_fixed_injection(
    tree: ast.Module, source_path: Path
) -> tuple[Callable[[dict[str, Any]], list[dict[str, Any]]], str]:
    process = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "process_chat_payload"
        ),
        None,
    )
    if process is None:
        raise SourceDriftError(f"missing process_chat_payload in {source_path}")
    candidates = []
    for node in ast.walk(process):
        if not isinstance(node, ast.For):
            continue
        structural = ast.dump(node, include_attributes=False)
        if "image_url" in structural and "files" in structural and "image_files" in structural:
            candidates.append(node)
    if len(candidates) != 1:
        raise SourceDriftError(
            f"expected one upstream image injection block in {source_path}"
        )
    injection = copy.deepcopy(candidates[0])
    wrapper = ast.FunctionDef(
        name="_run_upstream_image_injection",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="form_data")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            injection,
            ast.Return(
                value=ast.Subscript(
                    value=ast.Name(id="form_data", ctx=ast.Load()),
                    slice=ast.Constant(value="messages"),
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    structural_sha256 = sha256_bytes(
        ast.dump(injection, include_attributes=False).encode("utf-8")
    )
    return namespace["_run_upstream_image_injection"], structural_sha256


def source_runtime(version: str) -> dict[str, Any]:
    path, source, tree = load_source(version)
    namespace = {
        "Chats": FakeChats,
        "Optional": Optional,
        "convert_output_to_messages": lambda _value: [],
        "get_message_list": get_message_list,
    }
    load_messages = compile_function(
        tree, path, "load_messages_from_db", namespace
    )
    process_messages = compile_function(
        tree, path, "process_messages_with_output", namespace
    )
    injection = None
    injection_sha256 = None
    if version == "v0.8.3":
        injection, injection_sha256 = compile_fixed_injection(tree, path)
    return {
        "injection": injection,
        "injection_ast_sha256": injection_sha256,
        "load_messages": load_messages,
        "path": path,
        "process_messages": process_messages,
        "source": source,
    }


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


def count_image_parts(value: Any) -> int:
    if isinstance(value, list):
        return sum(count_image_parts(item) for item in value)
    if not isinstance(value, Mapping):
        return 0
    here = int(
        isinstance(value.get("image_url"), Mapping)
        and isinstance(value["image_url"].get("url"), str)
    )
    return here + sum(count_image_parts(item) for item in value.values())


def run_version(
    runtime: Mapping[str, Any],
    version: str,
    case_id: str,
    mime_type: str,
    image_bytes: bytes,
) -> dict[str, Any]:
    task_id = f"task-{case_id}"
    task_text = "Describe the attached local image."
    prompt = "\n".join(
        (
            f"evidence_task_id: {task_id}",
            f"evidence_case_id: {case_id}",
            task_text,
        )
    )
    data_url = (
        f"data:{mime_type};base64,"
        f"{base64.b64encode(image_bytes).decode('ascii')}"
    )
    FakeChats.rows = [
        {
            "role": "user",
            "content": prompt,
            "files": [
                {
                    "content_type": mime_type,
                    "type": "image",
                    "url": data_url,
                }
            ],
        }
    ]
    loaded = runtime["load_messages"]("chat-source-regression", "parent-message")
    loaded_file_count = sum(len(message.get("files", [])) for message in loaded)
    form_data = {"messages": copy.deepcopy(loaded)}
    if runtime["injection"] is not None:
        runtime["injection"](form_data)
    messages = runtime["process_messages"](form_data["messages"])
    body = {"messages": messages, "model": "local-open-webui-source-regression"}
    body_bytes = canonical_json_bytes(body)
    expected = expected_binding(task_id, task_text, mime_type, image_bytes)
    case = baselines.CorpusCase(
        audit_case_id=f"{case_id}-{version}",
        body=body_bytes,
        expected=expected,
        is_control=version == "v0.8.3",
        registry={task_id: expected},
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
        "image_parts_in_emitted_body": count_image_parts(body),
        "loaded_file_count": loaded_file_count,
        "unguarded_dispatch_invocations": 1,
    }


def build_audit() -> dict[str, Any]:
    old = source_runtime("v0.8.2")
    fixed = source_runtime("v0.8.3")
    diff = list(
        difflib.ndiff(old["source"].splitlines(), fixed["source"].splitlines())
    )
    insertions = sum(line.startswith("+ ") for line in diff)
    deletions = sum(line.startswith("- ") for line in diff)
    if (insertions, deletions) != (26, 1):
        raise SourceDriftError("unexpected upstream source diff")

    fixtures = (
        ("open-webui-png", "image/png", PNG_BYTES),
        ("open-webui-jpeg", "image/jpeg", JPEG_BYTES),
    )
    rows = []
    for case_id, mime_type, image_bytes in fixtures:
        rows.append(
            {
                "case_id": case_id,
                "image_sha256": sha256_bytes(image_bytes),
                "mime_type": mime_type,
                "v0.8.2": run_version(
                    old, "v0.8.2", case_id, mime_type, image_bytes
                ),
                "v0.8.3": run_version(
                    fixed, "v0.8.3", case_id, mime_type, image_bytes
                ),
            }
        )

    if any(
        row["v0.8.2"]["loaded_file_count"] != 0
        or row["v0.8.2"]["image_parts_in_emitted_body"] != 0
        or row["v0.8.2"]["contract_decision"] != "blocked"
        or row["v0.8.2"]["contract_reason"] != "missing_image"
        or row["v0.8.2"]["guarded_dispatch_invocations"] != 0
        for row in rows
    ):
        raise SourceDriftError("v0.8.2 did not reproduce the missing-image path")
    if any(
        row["v0.8.3"]["loaded_file_count"] != 1
        or row["v0.8.3"]["image_parts_in_emitted_body"] != 1
        or row["v0.8.3"]["contract_decision"] != "forwarded"
        or row["v0.8.3"]["contract_reason"]
        != "caller_owned_binding_satisfied"
        or row["v0.8.3"]["guarded_dispatch_invocations"] != 1
        for row in rows
    ):
        raise SourceDriftError("v0.8.3 did not reproduce the fixed image path")

    return {
        "audit_schema_version": 1,
        "cases": rows,
        "caveats": [
            "This is a source-level execution of the pinned attachment-serialization path, not a full Open WebUI server deployment.",
            "The fixtures use synthetic one-pixel PNG and JPEG images and make no model or provider call.",
            "The result reproduces one reported client boundary and does not estimate incident frequency, production effectiveness, or clinical safety.",
        ],
        "execution": {
            "credentials_required": False,
            "exact_upstream_ast_executed": True,
            "full_server_started": False,
            "network_calls": 0,
            "provider_or_model_calls": 0,
        },
        "source": {
            "fix_commit": FIX_COMMIT,
            "fix_commit_parent": FIX_PARENT,
            "fix_commit_url": FIX_URL,
            "issue_url": ISSUE_URL,
            "project": "open-webui/open-webui",
            "source_files": SOURCE_PROVENANCE,
            "upstream_diff": {"insertions": insertions, "deletions": deletions},
            "v0.8.3_injection_ast_sha256": fixed["injection_ast_sha256"],
        },
        "totals": {
            "fixed_contract_forwarded": sum(
                row["v0.8.3"]["contract_decision"] == "forwarded" for row in rows
            ),
            "fixed_image_parts": sum(
                row["v0.8.3"]["image_parts_in_emitted_body"] for row in rows
            ),
            "fixtures": len(rows),
            "old_contract_blocked": sum(
                row["v0.8.2"]["contract_decision"] == "blocked" for row in rows
            ),
            "old_guarded_dispatches": sum(
                row["v0.8.2"]["guarded_dispatch_invocations"] for row in rows
            ),
            "old_image_parts": sum(
                row["v0.8.2"]["image_parts_in_emitted_body"] for row in rows
            ),
        },
    }


def render_tex(audit: Mapping[str, Any]) -> str:
    total = audit["totals"]["fixtures"]
    return "\n".join(
        (
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\caption{\textbf{Pinned Open WebUI source regression.} Exact upstream source is executed on one PNG and one JPEG fixture. This is a source-level path test, not a full server deployment.}",
            r"\label{tab:open-webui-source-regression}",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Upstream source & Files kept & Image parts & Contract pass & Guarded sends \\",
            r"\midrule",
            f"Open WebUI v0.8.2 & 0/{total} & 0/{total} & 0/{total} & 0/{total} \\\\",
            f"Open WebUI v0.8.3 & {total}/{total} & {total}/{total} & {total}/{total} & {total}/{total} \\\\",
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
        default=DATA_ROOT / "open_webui_source_regression_audit.json",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=ARTIFACT_ROOT / "data" / "derived" / "tables" / "open_webui_source_regression.tex",
    )
    args = parser.parse_args()
    audit = build_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.tex_out.write_text(render_tex(audit))
    print(
        "open webui source regression passed: "
        f"{audit['totals']['old_contract_blocked']}/"
        f"{audit['totals']['fixtures']} old bodies blocked; "
        f"{audit['totals']['fixed_contract_forwarded']}/"
        f"{audit['totals']['fixtures']} fixed bodies forwarded"
    )


if __name__ == "__main__":
    main()
