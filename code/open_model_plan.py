from __future__ import annotations

import base64
import binascii
import hashlib
import struct
import zlib

from control_specs import (
    DEMOGRAPHIC_ABLATIONS,
    DEMOGRAPHIC_TEXT,
    E5_VARIANTS,
    E6_MODES,
    E7_IDENTITIES,
    GUARD_STRESS_PHRASES,
    attachment_zero_prompt,
    base_json_prompt,
    defensive_schema_prompt,
    demographic_ablation_prompt,
    e5_prompt,
    e6_prompt,
    e7_prompt,
    e8_prompt,
    guard_stress_prompt,
)

OPEN_MODEL_KEYS = ["qwen3_vl_32b", "llama4_maverick"]
OPEN_MODEL_PRICES = {
    "qwen3_vl_32b": (0.104, 0.416),
    "llama4_maverick": (0.15, 0.60),
}

DOMAINS = ["xray", "mri", "derm"]
DEMOS = ["D0"] + [d for d in DEMOGRAPHIC_TEXT if d != "D0"]
TOP_CELLS = {
    "xray": "32bm",
    "mri": "32wm",
    "derm": "65wm",
}


def extension_seeds(n: int, *, offset: int = 0) -> list[int]:
    seeds: list[int] = []
    i = offset
    while len(seeds) < n:
        digest = hashlib.sha256(f"open-model-base-rate-precision:{i}".encode()).digest()
        seeds.append(int.from_bytes(digest[:4], "big") % 2_147_483_647)
        i += 1
    return seeds


def _png_data_uri(width: int, height: int, pixels: list[tuple[int, int, int]]) -> str:
    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)
        for x in range(width):
            raw_rows.extend(pixels[y * width + x])

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw_rows))) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def image_conditions() -> dict[str, str]:
    width = height = 64
    blank = [(245, 245, 245)] * (width * height)
    checker = []
    for y in range(height):
        for x in range(width):
            checker.append((32, 32, 32) if ((x // 8) + (y // 8)) % 2 == 0 else (225, 225, 225))
    return {
        "blank_image_bytes": _png_data_uri(width, height, blank),
        "nonmedical_image_bytes": _png_data_uri(width, height, checker),
        "corrupted_image_bytes": "data:image/png;base64," + base64.b64encode(b"not a valid png").decode("ascii"),
    }


def _task(
    provider: str,
    experiment: str,
    domain: str,
    demographic: str,
    prompt: str,
    n: int,
    *,
    output_mode: str = "strict_json",
    prompt_variant: str | None = None,
    control_family: str | None = None,
    source_cell: str | None = None,
    image_data_uri: str | None = None,
    seed_source: str = "master",
) -> dict:
    return {
        "provider": provider,
        "experiment": experiment,
        "domain": domain,
        "demographic": demographic,
        "prompt": prompt,
        "n": n,
        "output_mode": output_mode,
        "prompt_variant": prompt_variant,
        "control_family": control_family or experiment,
        "source_cell": source_cell,
        "image_data_uri": image_data_uri,
        "seed_source": seed_source,
    }


def build_open_model_tasks(n: int, *, e15_n: int = 200, corrupted_n: int = 20) -> list[dict]:
    tasks: list[dict] = []
    images = image_conditions()
    for provider in OPEN_MODEL_KEYS:
        for domain in DOMAINS:
            for demo in DEMOS:
                tasks.append(_task(provider, "E1", domain, demo, e5_prompt(domain, demo, "original"), n,
                                   prompt_variant="original", source_cell=f"{domain}_{demo}"))
        for domain, demo in TOP_CELLS.items():
            for p_suffix in ("P1", "P2", "P3"):
                demographic = f"{demo}_{p_suffix}"
                tasks.append(_task(provider, "E2b", domain, demographic, e5_prompt(domain, demo, "original"), n,
                                   prompt_variant=p_suffix, source_cell=f"{domain}_{demo}"))
        for domain, demo in TOP_CELLS.items():
            for variant in E5_VARIANTS:
                tasks.append(_task(provider, "E5", domain, f"{demo}_{variant}", e5_prompt(domain, demo, variant), n,
                                   prompt_variant=variant, source_cell=f"{domain}_{demo}"))
            for mode in E6_MODES:
                tasks.append(_task(provider, "E6", domain, f"{demo}_{mode}", e6_prompt(domain, demo, mode), n,
                                   output_mode=mode, prompt_variant=mode, source_cell=f"{domain}_{demo}"))
            for guard_key in GUARD_STRESS_PHRASES:
                tasks.append(_task(provider, "E12", domain, f"{demo}_{guard_key}",
                                   guard_stress_prompt(domain, demo, guard_key), n,
                                   prompt_variant=guard_key, source_cell=f"{domain}_{demo}"))
            tasks.append(_task(provider, "E13", domain, f"{demo}_schema_contract",
                               defensive_schema_prompt(domain, demo), n,
                               prompt_variant="schema_contract_prompt", source_cell=f"{domain}_{demo}"))
            tasks.append(_task(provider, "E8", domain, f"{demo}_mitigation", e8_prompt(domain, demo), n,
                               prompt_variant="strict_absent_image_guard", source_cell=f"{domain}_{demo}"))
            tasks.append(_task(provider, "E9", domain, f"{demo}_attachment_zero",
                               attachment_zero_prompt(domain, demo), n,
                               prompt_variant="attachment_zero", source_cell=f"{domain}_{demo}"))
            for condition, data_uri in images.items():
                task_n = corrupted_n if condition == "corrupted_image_bytes" else n
                tasks.append(_task(provider, "E14", domain, f"{demo}_{condition}", e5_prompt(domain, demo, "original"),
                                   task_n, prompt_variant=condition, source_cell=f"{domain}_{demo}",
                                   image_data_uri=data_uri))
        for domain in ("xray", "mri"):
            for identity in E7_IDENTITIES:
                tasks.append(_task(provider, "E7", domain, f"32m_{identity}", e7_prompt(domain, identity), n,
                                   prompt_variant=identity, source_cell=f"{domain}_32rm"))
        for domain in DOMAINS:
            for key in DEMOGRAPHIC_ABLATIONS:
                tasks.append(_task(provider, "E11", domain, key, demographic_ablation_prompt(domain, key), n,
                                   prompt_variant=key, source_cell=f"{domain}_{TOP_CELLS[domain]}"))
        for domain, demo in TOP_CELLS.items():
            tasks.append(_task(provider, "E15", domain, demo, base_json_prompt(domain, demo, attachment_count=True),
                               e15_n, prompt_variant="original_absent_byte", source_cell=f"{domain}_{demo}",
                               control_family="base_rate_precision", seed_source="extension"))
    return tasks
