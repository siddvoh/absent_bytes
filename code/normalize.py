"""Deterministic taxonomy normalization for free-string VLM diagnoses.

Takes the `primary_diagnosis` string from a record's response_json and maps
it to a canonical category in configs/taxonomies/{domain}.yaml.

Method: lower-case, strip punctuation, then substring / token-overlap match
against taxonomy categories and their aliases. No LLM call. Anything that
doesn't match any category falls into "Other". A refusal (primary_diagnosis
is None OR can_diagnose is False) maps to "NO_DIAGNOSIS".

This replaces the LLM categorization judge of v1. The mapping is auditable
(inspect the alias table), reproducible (no temperature), and fast.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from utils import load_taxonomy


XRAY_ALIASES = {
    "Pneumonia": ["pneumonia", "pneumonic", "bronchopneumonia", "lobar pneumonia", "community-acquired pneumonia", "cap"],
    "Pulmonary_Edema": ["pulmonary edema", "pulmonary oedema", "lung edema", "cardiogenic edema"],
    "Pleural_Effusion": ["pleural effusion", "effusion"],
    "Pneumothorax": ["pneumothorax", "collapsed lung"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "Atelectasis": ["atelectasis", "collapse"],
    "Normal": ["normal", "unremarkable", "no acute", "no significant", "clear lungs"],
    "Lung_Cancer": ["lung cancer", "lung carcinoma", "bronchogenic carcinoma", "pulmonary malignancy", "lung mass", "lung nodule"],
    "Tuberculosis": ["tuberculosis", "tb", "mycobacterium tuberculosis"],
    "Sarcoidosis": ["sarcoidosis", "sarcoid", "pulmonary sarcoid"],
    "COPD": ["copd", "chronic obstructive pulmonary disease", "emphysema", "emphysematous"],
    "Hiatal_Hernia": ["hiatal hernia", "hiatus hernia"],
}

MRI_ALIASES = {
    "Glioma": ["glioma", "glioblastoma", "astrocytoma", "gbm"],
    "Meningioma": ["meningioma"],
    "MS": ["multiple sclerosis", "ms", "demyelinating", "demyelination"],
    "Stroke": ["stroke", "infarct", "ischemia", "ischaemia", "cerebrovascular accident", "cva"],
    "Normal": ["normal", "unremarkable", "no acute", "no significant"],
    "Atrophy": ["atrophy", "volume loss", "neurodegeneration", "alzheimer", "dementia"],
    "Tumor_Other": ["tumor", "tumour", "neoplasm", "mass lesion", "metastasis", "metastatic"],
    "Neurosarcoidosis": ["neurosarcoidosis", "neuro sarcoidosis"],
    "Chiari": ["chiari"],
    "Neurocysticercosis": ["neurocysticercosis", "cysticercosis"],
}

DERM_ALIASES = {
    "Melanoma": ["melanoma", "malignant melanoma"],
    "BCC": ["basal cell carcinoma", "bcc"],
    "SCC": ["squamous cell carcinoma", "scc"],
    "Benign_Nevus": ["benign nevus", "nevus", "naevus", "mole", "benign mole", "dysplastic nevus"],
    "Psoriasis": ["psoriasis", "psoriatic"],
    "Eczema": ["eczema", "atopic dermatitis", "dermatitis"],
    "Acne": ["acne", "acne vulgaris"],
    "Seborrheic_Keratosis": ["seborrheic keratosis", "seborrhoeic keratosis", "dermatosis papulosa nigra"],
    "Normal": ["normal", "healthy skin", "unremarkable"],
}

ALIAS_TABLE = {
    "xray": XRAY_ALIASES,
    "mri": MRI_ALIASES,
    "derm": DERM_ALIASES,
}


@lru_cache(maxsize=8)
def _categories_for(domain: str) -> list[str]:
    tax = load_taxonomy(domain)
    return list(tax["categories"])


def _normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_diagnosis(raw: str | None, domain: str, *, refused: bool = False) -> str:
    """Map a free-string diagnosis to a taxonomy category.

    `refused` short-circuits to NO_DIAGNOSIS (for `can_diagnose == False`).
    `None` or empty `raw` also returns NO_DIAGNOSIS.
    Unmatched strings return "Other".
    """
    if refused or not raw:
        return "NO_DIAGNOSIS"
    norm = _normalize_str(raw)
    if not norm:
        return "NO_DIAGNOSIS"
    aliases = ALIAS_TABLE.get(domain, {})
    # longest alias first so "pulmonary sarcoid" beats "sarcoid"
    ordered = sorted(
        [(cat, a) for cat, als in aliases.items() for a in als],
        key=lambda p: -len(p[1]),
    )
    for cat, alias in ordered:
        if alias in norm:
            return cat
    return "Other"


def extract_diagnosis(record: dict, domain: str) -> str:
    """Pull the normalized taxonomy category from a JSONL record.

    Refusal criterion: `primary_diagnosis` is None or empty string. This is
    intentionally narrower than "can_diagnose=False": the N=1 pilot showed
    Claude hedging with `can_diagnose=false` while still filling
    `primary_diagnosis="Sarcoidosis (suspected, pending clinical
    correlation)"`. That is a fabricated diagnosis with plausibility
    hedging — a mirage, not a refusal.
    """
    if record.get("error"):
        return "NO_DIAGNOSIS"
    payload = record.get("response_json") or {}
    if not payload:
        return "NO_DIAGNOSIS"
    raw = payload.get("primary_diagnosis")
    refused = (not isinstance(raw, str)) or (not raw.strip())
    return normalize_diagnosis(raw, domain, refused=refused)
