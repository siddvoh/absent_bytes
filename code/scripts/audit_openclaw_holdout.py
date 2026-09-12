#!/usr/bin/env python3
"""Verify the post-freeze OpenClaw image-routing holdout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = CODE_ROOT.parent
DATA_ROOT = ARTIFACT_ROOT / "data"
THIRD_PARTY = ARTIFACT_ROOT / "third_party" / "openclaw"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_enforcement_baselines as baselines
import audit_open_webui_source_regression as open_webui


PROJECT = "openclaw/openclaw"
PARENT_COMMIT = "36e2e04a32c2a94eafc83fd7e1e82a89045e2e9f"
FIX_COMMIT = "9c86a9fd23469e7c10b80be3ed6503ee9cd8da0e"
SOURCE_PATH = "src/gateway/openai-http.ts"
TEST_PATH = "src/gateway/openai-http.test.ts"
LOCK_PATH = "pnpm-lock.yaml"
ISSUE_URL = "https://github.com/openclaw/openclaw/issues/18583"
PR_URL = "https://github.com/openclaw/openclaw/pull/34068"
FIX_URL = f"https://github.com/openclaw/openclaw/commit/{FIX_COMMIT}"
NATIVE_COMMAND = (
    "pnpm exec vitest run --config vitest.gateway.config.ts "
    "src/gateway/openai-http.test.ts --pool=forks --maxWorkers=1"
)
SELECTED_AT_UTC = "2026-07-17T09:53:20Z"

SOURCE_PROVENANCE = {
    "parent": {
        "commit": PARENT_COMMIT,
        "file": "third_party/openclaw/openai_http_parent.ts",
        "git_blob_sha1": "10e8d713feed548faba7ef3d0c40a1e70eaab050",
        "sha256": "23eabe7181281edcc518372f6fd068273f6b3ab6354bcd623a2ddf8c911d5a15",
    },
    "fixed": {
        "commit": FIX_COMMIT,
        "file": "third_party/openclaw/openai_http_fix.ts",
        "git_blob_sha1": "1f37dfb1fae6afc1c3235238daa97c5ff165d162",
        "sha256": "d79b2846dd2f05b9d0d19c37bb53992bc360f35e23cacb3e8992fd4939662c10",
    },
}
TEST_PROVENANCE = {
    "parent": {
        "commit": PARENT_COMMIT,
        "file": "third_party/openclaw/openai_http_parent.test.ts",
        "git_blob_sha1": "c9d429521a46dc165416dc5e72b8f63722d0b0c0",
        "sha256": "a01f70ae9848c47e7336d55bc2bcb83091afb827c3382e3f6a21332601df763a",
    },
    "fixed": {
        "commit": FIX_COMMIT,
        "file": "third_party/openclaw/openai_http_fix.test.ts",
        "git_blob_sha1": "f3ab97093ba51618af0cf109c57f297af6647e07",
        "sha256": "bac6ac56f54cb5b7c9b5c81dfd3bed2d35f7c9fe18052da2b2f8ecfc1dd85358",
    },
}
LOCK_PROVENANCE = {
    "git_blob_sha1": "79313de6f9f4b3c49f886e1919e6cd5fd00941e0",
    "path": LOCK_PATH,
    "sha256": "e8ddb3f877d1de7bbdebb0d99e6c399c55a286d09b8d53a4ee7a6039782a5370",
}
LICENSE_PROVENANCE = {
    "file": "third_party/openclaw/LICENSE",
    "git_blob_sha1": "f7b526698bb7ed2d26d96c49f2f32234c88f69bc",
    "sha256": "62316704df7426e5a79d2827ff8aca36e9abb3a73b8e68557030749ebefec667",
}
PATCH_PROVENANCE = {
    "file": "third_party/openclaw/native_regression_test.patch",
    "sha256": "eb152b9bca99358cfa5912d3288cb091b1c28e5898a6d5dabf12a4f2282edf5b",
}
GUARD_LOCKS = {
    "code/scripts/audit_enforcement_baselines.py": "880a7622516635aced5ab3ffc7703a855ee697e171bb269ae7ac395ce61e95ca",
    "code/scripts/audit_evidence_binding_matrix.py": "0156d09e2890e5462ea116746c82e4f3da1378b74fa71a5271977c14b16ac25f",
    "code/scripts/fault_injection_pipeline.py": "7b618efb4589f5aa5f6be891625a8a9176b1d80c5ecbf7326006dc61ba3b01de",
}


class AuditError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def load_verified(provenance: Mapping[str, str]) -> bytes:
    value = (ARTIFACT_ROOT / provenance["file"]).read_bytes()
    if sha256_bytes(value) != provenance["sha256"]:
        raise AuditError(f"SHA-256 drift: {provenance['file']}")
    blob = provenance.get("git_blob_sha1")
    if blob and git_blob_sha1(value) != blob:
        raise AuditError(f"Git blob drift: {provenance['file']}")
    return value


def verify_test_patch() -> None:
    parent = load_verified(TEST_PROVENANCE["parent"])
    fixed = load_verified(TEST_PROVENANCE["fixed"])
    patch = load_verified(PATCH_PROVENANCE)
    with tempfile.TemporaryDirectory(prefix="openclaw-patch-check-") as temp:
        root = Path(temp)
        target = root / TEST_PATH
        target.parent.mkdir(parents=True)
        target.write_bytes(parent)
        applied = subprocess.run(
            ["git", "apply", str(THIRD_PARTY / "native_regression_test.patch")],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if applied.returncode:
            raise AuditError(applied.stderr.decode().strip())
        if target.read_bytes() != fixed:
            raise AuditError("test-only patch does not produce the fixed test blob")
    if b"firstCall?.images" not in patch or b"type: \"image_url\"" not in patch:
        raise AuditError("native regression patch lost its image assertion")


def verify_frozen_guard() -> None:
    for rel, expected in GUARD_LOCKS.items():
        observed = sha256_bytes((ARTIFACT_ROOT / rel).read_bytes())
        if observed != expected:
            raise AuditError(f"post-freeze guard drift: {rel}")


def freeze_record() -> dict[str, Any]:
    return {
        "freeze_schema_version": 1,
        "holdout_selected_at_utc": SELECTED_AT_UTC,
        "selection": {
            "kind": "post-definition non-random external holdout",
            "issue_url": ISSUE_URL,
            "project": PROJECT,
            "pull_request_url": PR_URL,
            "rationale": (
                "public image-omission report with an independently authored upstream "
                "before/fix regression test"
            ),
        },
        "iteration43_commitment": {
            "final_pdf_sha256": "812ec13da358417a30ff2b2e50af1f304b9cd910672dd30a8119b3054b3cc5a4",
            "frozen_packet_manifest_sha256": "2bddd061d0ddaad4aa27243fe1a95c64b5a0c78106eec597687426e671f0a5d4",
            "release_manifest_sha256": "347c820586135ebbcc2cb30af9d0591a29c9c5d6b9cb1804b8679b0a545b37f7",
            "guard_source_sha256": GUARD_LOCKS,
        },
        "openclaw_revisions": {
            "fix": FIX_COMMIT,
            "fix_url": FIX_URL,
            "parent": PARENT_COMMIT,
        },
        "guard_source_changed_for_holdout": False,
        "preregistered_or_random_sample": False,
    }


TOP_LEVEL_FUNCTION = re.compile(r"(?m)^(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\(")


def extract_function_segment(source: str, name: str) -> str:
    matches = list(TOP_LEVEL_FUNCTION.finditer(source))
    for index, match in enumerate(matches):
        if match.group(1) != name:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        return source[match.start():end].rstrip()
    raise AuditError(f"missing upstream function: {name}")


def fixture_rows() -> list[dict[str, Any]]:
    fixtures = (
        ("openclaw-png", "image/png", open_webui.PNG_BYTES),
        ("openclaw-jpeg", "image/jpeg", open_webui.JPEG_BYTES),
    )
    rows = []
    for case_id, mime_type, image_bytes in fixtures:
        task_text = f"Describe the local {mime_type} fixture."
        encoded = base64.b64encode(image_bytes).decode("ascii")
        rows.append(
            {
                "case_id": case_id,
                "mime_type": mime_type,
                "image_base64": encoded,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
                "task_text": task_text,
            }
        )
    return rows


def node_program(label: str, source: str) -> tuple[str, dict[str, str]]:
    names = ["buildAgentCommandInput", "asMessages", "extractTextContent"]
    if label == "fixed":
        names.extend(
            [
                "resolveImageUrlPart",
                "extractImageUrls",
                "parseImageUrlToSource",
                "resolveActiveTurnContext",
            ]
        )
    segments = {name: extract_function_segment(source, name) for name in names}
    body = "\n\n".join(segments.values())
    fixed_logic = """
      const context = resolveActiveTurnContext(fixture.messages);
      images = context.urls.map((url) => {
        const parsed = parseImageUrlToSource(url);
        if (parsed.type !== "base64") throw new Error("unexpected URL fixture");
        return {type: "image", data: parsed.data, mimeType: parsed.mediaType};
      });
""" if label == "fixed" else ""
    program = f"""
{body}

const fixtures = JSON.parse(process.env.OPENCLAW_FIXTURES_JSON || "[]");
const results = fixtures.map((fixture) => {{
  const active = fixture.messages[fixture.messages.length - 1];
  const message = extractTextContent(active.content).trim();
  let images = [];
{fixed_logic}
  const commandInput = buildAgentCommandInput({{
    prompt: {{message, images: images.length > 0 ? images : undefined}},
    sessionKey: "holdout-session",
    runId: fixture.case_id,
    messageChannel: "webchat",
  }});
  return {{case_id: fixture.case_id, commandInput}};
}});
process.stdout.write(JSON.stringify(results));
"""
    hashes = {name: sha256_bytes(value.encode("utf-8")) for name, value in segments.items()}
    return program, hashes


def run_source(label: str, fixtures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    source = load_verified(SOURCE_PROVENANCE[label]).decode("utf-8")
    program, hashes = node_program(label, source)
    node = shutil.which("node")
    if node is None:
        raise AuditError("Node.js is required for exact TypeScript function execution")
    with tempfile.TemporaryDirectory(prefix=f"openclaw-{label}-") as temp:
        path = Path(temp) / "holdout.ts"
        path.write_text(program)
        env = os.environ.copy()
        env["OPENCLAW_FIXTURES_JSON"] = json.dumps(fixtures, separators=(",", ":"))
        completed = subprocess.run(
            [node, "--no-warnings", str(path)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise AuditError(completed.stderr.strip())
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list) or len(rows) != len(fixtures):
        raise AuditError(f"unexpected {label} source execution output")
    return rows, hashes


def expected_binding(case_id: str, task_text: str, mime_type: str, image_bytes: bytes) -> baselines.ExpectedBinding:
    task_id = f"task-{case_id}"
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
        "correlation_id": f"openclaw-holdout/{case_id}",
    }
    return baselines.ExpectedBinding(
        client_response_binding=response_binding,
        expected_response_binding=response_binding,
        images=(image,),
        response_schema_required=False,
        serialized_response_schema=None,
        task_text=task_text,
    )


def command_view(command_input: Mapping[str, Any], expected: baselines.ExpectedBinding) -> baselines.RequestView:
    images = []
    for item in command_input.get("images") or []:
        encoded = str(item.get("data") or "")
        try:
            data = base64.b64decode(encoded, validate=True)
            valid = True
        except ValueError:
            data = b""
            valid = False
        declared = str(item.get("mimeType") or "").lower()
        images.append(
            baselines.InlineImage(
                base64_valid=valid,
                data=data,
                declared_mime_type=declared,
                decoded_mime_type=baselines._decoded_mime_type(data) if valid else None,
            )
        )
    frozen_images = tuple(images)
    return baselines.RequestView(
        active_images=frozen_images,
        active_references=(),
        active_task_id=expected.task_id,
        active_task_text=expected.task_text,
        body=dict(command_input),
        global_images=frozen_images,
        global_references=(),
        global_task_id=expected.task_id,
        out_of_scope_image_count=0,
        response_schema=None,
        shape_reason="openclaw_command_input_adapter",
        shape_valid=True,
    )


def build_source_audit() -> dict[str, Any]:
    fixtures = fixture_rows()
    parent_rows, parent_hashes = run_source("parent", fixtures)
    fixed_rows, fixed_hashes = run_source("fixed", fixtures)
    by_label = {
        "parent": {row["case_id"]: row for row in parent_rows},
        "fixed": {row["case_id"]: row for row in fixed_rows},
    }
    rows = []
    for fixture in fixtures:
        case_id = fixture["case_id"]
        image_bytes = base64.b64decode(fixture["image_base64"], validate=True)
        expected = expected_binding(
            case_id, fixture["task_text"], fixture["mime_type"], image_bytes
        )
        case = baselines.CorpusCase(
            audit_case_id=case_id,
            body=b"{}",
            expected=expected,
            is_control=True,
            registry={expected.task_id: expected},
            sdk_family="openai",
            source_case_id=case_id,
        )
        row = {"case_id": case_id, "mime_type": fixture["mime_type"]}
        for label in ("parent", "fixed"):
            command_input = by_label[label][case_id]["commandInput"]
            allowed, reason = baselines.caller_owned_binding(
                case, command_view(command_input, expected)
            )
            row[label] = {
                "command_input_sha256": sha256_bytes(canonical_json_bytes(command_input)),
                "guard_decision": "forwarded" if allowed else "blocked",
                "guard_reason": reason,
                "image_count_at_agent_command": len(command_input.get("images") or []),
                "message": command_input.get("message"),
            }
        rows.append(row)
    if any(
        row["parent"]["image_count_at_agent_command"] != 0
        or row["parent"]["guard_decision"] != "blocked"
        or row["parent"]["guard_reason"] != "missing_image"
        or row["fixed"]["image_count_at_agent_command"] != 1
        or row["fixed"]["guard_decision"] != "forwarded"
        for row in rows
    ):
        raise AuditError("OpenClaw holdout result drift")
    return {
        "adapter": {
            "authored_after_holdout_selection": True,
            "description": (
                "thin adapter from captured OpenClaw agentCommand image objects to the "
                "frozen generic RequestView"
            ),
            "guard_source_modified": False,
        },
        "cases": rows,
        "exact_function_source_sha256": {
            "fixed": fixed_hashes,
            "parent": parent_hashes,
        },
        "execution": {
            "exact_upstream_typescript_functions_executed": True,
            "network_calls": 0,
            "provider_or_model_calls": 0,
            "valid_data_uri_resolver": "local no-network adapter",
        },
        "totals": {
            "affected_cases": len(rows),
            "fixed_contract_forwarded": sum(
                row["fixed"]["guard_decision"] == "forwarded" for row in rows
            ),
            "fixed_images_at_agent_command": sum(
                row["fixed"]["image_count_at_agent_command"] for row in rows
            ),
            "parent_contract_blocked": sum(
                row["parent"]["guard_decision"] == "blocked" for row in rows
            ),
            "parent_images_at_agent_command": sum(
                row["parent"]["image_count_at_agent_command"] for row in rows
            ),
        },
    }


def git_text(checkout: Path, revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=checkout,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise AuditError(completed.stderr.decode().strip())
    return completed.stdout


def verify_checkout(checkout: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=checkout, capture_output=True, check=False
    )
    if status.returncode or status.stdout:
        raise AuditError("OpenClaw checkout must be clean")
    for sources, path in ((SOURCE_PROVENANCE, SOURCE_PATH), (TEST_PROVENANCE, TEST_PATH)):
        for label, provenance in sources.items():
            value = git_text(checkout, provenance["commit"], path)
            if sha256_bytes(value) != provenance["sha256"]:
                raise AuditError(f"checkout {label} SHA-256 drift: {path}")
            if git_blob_sha1(value) != provenance["git_blob_sha1"]:
                raise AuditError(f"checkout {label} Git blob drift: {path}")
    for revision in (PARENT_COMMIT, FIX_COMMIT):
        lock = git_text(checkout, revision, LOCK_PATH)
        if sha256_bytes(lock) != LOCK_PROVENANCE["sha256"]:
            raise AuditError(f"checkout lock drift: {revision}")
        if git_blob_sha1(lock) != LOCK_PROVENANCE["git_blob_sha1"]:
            raise AuditError(f"checkout lock blob drift: {revision}")


def parse_vitest(output: str, returncode: int) -> dict[str, Any]:
    test_line = next((line for line in output.splitlines() if line.strip().startswith("Tests")), "")
    counts = {kind: int(count) for count, kind in re.findall(r"(\d+) (passed|failed)", test_line)}
    return {
        "exit_status": returncode,
        "failed": counts.get("failed", 0),
        "passed": counts.get("passed", 0),
    }


def run_native(checkout: Path, pnpm: Path) -> None:
    if not pnpm.is_file():
        raise AuditError(f"pnpm executable not found: {pnpm}")
    with tempfile.TemporaryDirectory(prefix="openclaw-native-regression-") as temp:
        root = Path(temp)
        worktrees = {
            "parent": (root / "parent", PARENT_COMMIT),
            "fixed": (root / "fixed", FIX_COMMIT),
        }
        for worktree, revision in worktrees.values():
            added = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), revision],
                cwd=checkout,
                capture_output=True,
                check=False,
            )
            if added.returncode:
                raise AuditError(added.stderr.decode().strip())
        try:
            applied = subprocess.run(
                ["git", "apply", str(THIRD_PARTY / "native_regression_test.patch")],
                cwd=worktrees["parent"][0],
                capture_output=True,
                check=False,
            )
            if applied.returncode:
                raise AuditError(applied.stderr.decode().strip())
            results = {}
            for label, (worktree, _) in worktrees.items():
                installed = subprocess.run(
                    [str(pnpm), "install", "--frozen-lockfile", "--ignore-scripts"],
                    cwd=worktree,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if installed.returncode:
                    raise AuditError(installed.stderr.strip())
                command = [
                    str(pnpm), "exec", "vitest", "run", "--config",
                    "vitest.gateway.config.ts", TEST_PATH, "--pool=forks", "--maxWorkers=1",
                ]
                completed = subprocess.run(
                    command, cwd=worktree, capture_output=True, text=True, check=False
                )
                results[label] = parse_vitest(
                    completed.stdout + completed.stderr, completed.returncode
                )
                if label == "parent" and (
                    "firstCall?.images" not in completed.stdout + completed.stderr
                    or "Received:" not in completed.stdout + completed.stderr
                ):
                    raise AuditError("parent did not fail at the image-forwarding assertion")
            expected = {
                "parent": {"exit_status": 1, "failed": 1, "passed": 3},
                "fixed": {"exit_status": 0, "failed": 0, "passed": 4},
            }
            if results != expected:
                raise AuditError(f"native OpenClaw regression mismatch: {results}")
        finally:
            for worktree, _ in worktrees.values():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=checkout,
                    capture_output=True,
                    check=False,
                )


def native_record() -> dict[str, Any]:
    return {
        "command": NATIVE_COMMAND,
        "execution": {
            "agent_command_mocked": True,
            "credentials_required": False,
            "full_gateway_handler_started": True,
            "localhost_http": True,
            "provider_or_model_calls": 0,
        },
        "fixed": {
            "commit": FIX_COMMIT,
            "exit_status": 0,
            "failed": 0,
            "passed": 4,
            "result": "passed",
        },
        "parent_with_upstream_test_only_patch": {
            "assertion": "agentCommand images expected one PNG; received undefined",
            "commit": PARENT_COMMIT,
            "exit_status": 1,
            "failed": 1,
            "passed": 3,
            "result": "failed",
        },
        "test_patch": PATCH_PROVENANCE,
        "toolchain": {
            "node": "v24.14.1",
            "os": "Darwin 25.5.0 arm64",
            "pnpm": "10.23.0",
            "vitest": "4.0.18",
        },
    }


def build_audit() -> dict[str, Any]:
    for provenance in SOURCE_PROVENANCE.values():
        load_verified(provenance)
    load_verified(LICENSE_PROVENANCE)
    verify_test_patch()
    verify_frozen_guard()
    source_audit = build_source_audit()
    return {
        "audit_schema_version": 1,
        "caveats": [
            "The holdout was selected after the predicate and guard were frozen, but it was deliberately selected from a public report. It was not randomly sampled or preregistered.",
            "The native result runs the exact upstream gateway test with a mocked agent command and no provider or model; it is not a production OpenClaw deployment.",
            "Normal offline verification executes exact upstream TypeScript extraction and command-input functions, then uses a thin author-written adapter to the frozen generic RequestView.",
            "This single external route establishes neither unseen-fault coverage nor incident or deployment frequency.",
        ],
        "execution": {
            "credentials_required": False,
            "normal_verification_network_calls": 0,
            "provider_or_model_calls": 0,
        },
        "freeze": freeze_record(),
        "native_regression": native_record(),
        "source": {
            "fix_commit": FIX_COMMIT,
            "fix_url": FIX_URL,
            "issue_url": ISSUE_URL,
            "license": LICENSE_PROVENANCE,
            "lock": LOCK_PROVENANCE,
            "parent_commit": PARENT_COMMIT,
            "project": PROJECT,
            "pull_request_url": PR_URL,
            "source_file": SOURCE_PATH,
            "source_files": SOURCE_PROVENANCE,
            "test_file": TEST_PATH,
            "test_files": TEST_PROVENANCE,
        },
        "source_execution": source_audit,
    }


def render_tex(audit: Mapping[str, Any]) -> str:
    totals = audit["source_execution"]["totals"]
    return "\n".join(
        (
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\caption{\textbf{Post-freeze OpenClaw holdout.} The exact upstream gateway test fails at the parent because \texttt{agentCommand.images} is absent and passes at the fix. The frozen guard is evaluated through a thin command-input adapter on local PNG and JPEG fixtures.}",
            r"\label{tab:openclaw-holdout}",
            r"\begin{tabular}{@{}p{.22\linewidth}p{.22\linewidth}p{.23\linewidth}p{.23\linewidth}@{}}",
            r"\toprule",
            "Revision & Upstream gateway test & Agent images & Frozen guard \\\\",
            r"\midrule",
            f"Parent + test patch & 3/4 pass; image test fails & 0/{totals['affected_cases']} & {totals['parent_contract_blocked']}/{totals['affected_cases']} blocked \\\\",
            f"Fix commit & 4/4 pass & {totals['fixed_images_at_agent_command']}/{totals['affected_cases']} & {totals['fixed_contract_forwarded']}/{totals['affected_cases']} forwarded \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=DATA_ROOT / "openclaw_holdout_audit.json"
    )
    parser.add_argument(
        "--freeze-out", type=Path, default=DATA_ROOT / "openclaw_holdout_freeze.json"
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=ARTIFACT_ROOT / "data" / "derived" / "tables" / "openclaw_holdout.tex",
    )
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--pnpm", type=Path)
    parser.add_argument("--rerun-native", action="store_true")
    args = parser.parse_args()
    if args.rerun_native and (args.checkout is None or args.pnpm is None):
        parser.error("--rerun-native requires --checkout and --pnpm")
    if args.checkout is not None:
        verify_checkout(args.checkout.resolve())
    if args.rerun_native:
        run_native(args.checkout.resolve(), args.pnpm.resolve())
    audit = build_audit()
    for path in (args.out, args.freeze_out, args.tex_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.freeze_out.write_text(
        json.dumps(audit["freeze"], indent=2, sort_keys=True) + "\n"
    )
    args.tex_out.write_text(render_tex(audit))
    totals = audit["source_execution"]["totals"]
    print(
        "OpenClaw holdout passed: upstream parent test failed and fix passed; "
        f"{totals['parent_contract_blocked']}/{totals['affected_cases']} parent "
        f"commands blocked and {totals['fixed_contract_forwarded']}/"
        f"{totals['affected_cases']} fixed commands forwarded"
    )


if __name__ == "__main__":
    main()
