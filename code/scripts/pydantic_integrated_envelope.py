from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import re
from typing import Any, Literal, Mapping

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator


RESPONSE_FIELDS = (
    "primary_diagnosis",
    "differentials",
    "clinical_assessment",
)
BINDING_FIELDS = (
    "task_id",
    "task_text_sha256",
    "contract_sha256",
    "contract_version",
    "active_field",
    "correlation_id",
)
TASK_ID_PATTERN = re.compile(r"(?m)^evidence_task_id: ([a-z0-9-]+)$")
TASK_METADATA_LINE_PATTERN = re.compile(r"^evidence_(?:task_id|case_id): [a-z0-9-]+$")
MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class _ImageURL(_WireModel):
    url: str


class _OpenAIPart(_WireModel):
    type: str
    text: str | None = None
    image_url: _ImageURL | None = None

    @model_validator(mode="after")
    def _has_supported_shape(self) -> "_OpenAIPart":
        if self.type == "text" and self.text is not None:
            return self
        if self.type == "image_url" and self.image_url is not None:
            return self
        raise ValueError("unsupported OpenAI content part")


class _OpenAITurn(_WireModel):
    role: str
    content: str | list[_OpenAIPart]


class _OpenAIRequest(_WireModel):
    messages: list[_OpenAITurn]


class _GoogleInlineData(_WireModel):
    data: str
    mimeType: str


class _GoogleInlineDataSnake(_WireModel):
    data: str
    mime_type: str


class _GoogleFileData(_WireModel):
    fileUri: str


class _GoogleFileDataSnake(_WireModel):
    file_uri: str


class _GooglePart(_WireModel):
    text: str | None = None
    inlineData: _GoogleInlineData | None = None
    inline_data: _GoogleInlineDataSnake | None = None
    fileData: _GoogleFileData | None = None
    file_data: _GoogleFileDataSnake | None = None

    @model_validator(mode="after")
    def _has_one_variant(self) -> "_GooglePart":
        variants = (
            self.text is not None,
            self.inlineData is not None,
            self.inline_data is not None,
            self.fileData is not None,
            self.file_data is not None,
        )
        if sum(variants) != 1:
            raise ValueError("Google part must contain exactly one supported variant")
        return self


class _GoogleTurn(_WireModel):
    role: str
    parts: list[_GooglePart]


class _GoogleRequest(_WireModel):
    contents: list[_GoogleTurn]


class _ExpectedImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mime_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ClientResponseBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    active_field: Literal[
        "primary_diagnosis", "differentials", "clinical_assessment"
    ]
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_version: int = Field(ge=1)
    correlation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ExpectedBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    client_response_binding: _ClientResponseBinding
    expected_response_binding: _ClientResponseBinding
    images: list[_ExpectedImage] = Field(min_length=1)
    response_schema_required: bool
    serialized_response_schema: dict[str, Any] | None

    @model_validator(mode="after")
    def _sidecar_matches_expectation(self) -> "_ExpectedBinding":
        if self.client_response_binding != self.expected_response_binding:
            raise ValueError("client response binding does not match its expectation")
        return self


class _EvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    base64_valid: bool
    declared_mime: str
    decoded_mime: str
    sha256: str


class _ResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    additional_properties: Any
    forbidden_binding_fields: list[str]
    path: str
    present: bool
    properties: list[str]
    required: list[Any]
    valid: bool


class _RequestEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    active_images: list[_EvidenceRow]
    active_reference_count: int
    global_reference_count: int
    out_of_scope_image_count: int
    parser_active_field: str
    response_contract: _ResponseContract
    sdk_family: Literal["openai", "google_genai"]
    task_id: str
    task_text_sha256: str

    @model_validator(mode="after")
    def _matches_binding(self, info: ValidationInfo) -> "_RequestEnvelope":
        expected = info.context.get("expected") if info.context else None
        if not isinstance(expected, _ExpectedBinding):
            raise ValueError("expected binding is unavailable")
        binding = expected.client_response_binding
        if self.parser_active_field != binding.active_field:
            raise ValueError("parser active field mismatch")
        if self.task_id != binding.task_id:
            raise ValueError("task ID mismatch")
        if self.task_text_sha256 != binding.task_text_sha256:
            raise ValueError("task text mismatch")
        if self.active_reference_count or self.global_reference_count:
            raise ValueError("remote evidence reference present")
        if self.out_of_scope_image_count:
            raise ValueError("evidence is outside the active user turn")
        wanted_images = expected.images
        if len(self.active_images) != len(wanted_images):
            raise ValueError("evidence image count mismatch")
        for observed, wanted in zip(self.active_images, wanted_images, strict=True):
            if not observed.base64_valid or not observed.decoded_mime:
                raise ValueError("evidence image is undecodable")
            if observed.declared_mime != wanted.mime_type:
                raise ValueError("evidence MIME mismatch")
            if observed.decoded_mime != wanted.mime_type:
                raise ValueError("evidence decoded MIME mismatch")
            if observed.sha256 != wanted.sha256:
                raise ValueError("evidence bytes or order mismatch")
        if expected.response_schema_required:
            if (
                expected.serialized_response_schema is None
                or not self.response_contract.valid
                or self.response_contract.model_dump()
                != expected.serialized_response_schema
            ):
                raise ValueError("response contract mismatch")
        return self


def _decode_base64(value: str) -> tuple[bytes, bool]:
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded, altchars=b"-_", validate=True), True
    except (ValueError, binascii.Error):
        return b"", False


def _decoded_mime(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return MIME_BY_FORMAT.get(str(image.format or "").upper(), "")
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError):
        return ""


def _scan_evidence(value: Any) -> tuple[list[_EvidenceRow], list[str]]:
    images: list[_EvidenceRow] = []
    references: list[str] = []

    def add(encoded: str, declared_mime: str) -> None:
        data, base64_valid = _decode_base64(encoded)
        images.append(
            _EvidenceRow(
                base64_valid=base64_valid,
                declared_mime=declared_mime.lower(),
                decoded_mime=_decoded_mime(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    def walk(item: Any) -> None:
        if isinstance(item, BaseModel):
            walk(item.model_dump())
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, Mapping):
            return
        consumed: set[str] = set()
        image_url = item.get("image_url")
        if isinstance(image_url, Mapping):
            consumed.add("image_url")
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                header, encoded = url.split(",", 1)
                add(encoded, header[5:].split(";", 1)[0])
            elif isinstance(url, str) and url:
                references.append(url)
        for key in ("inlineData", "inline_data"):
            inline = item.get(key)
            if isinstance(inline, Mapping):
                consumed.add(key)
                data = inline.get("data")
                mime = inline.get("mimeType", inline.get("mime_type"))
                add(data if isinstance(data, str) else "", mime if isinstance(mime, str) else "")
        for key in ("fileData", "file_data"):
            file_data = item.get(key)
            if isinstance(file_data, Mapping):
                consumed.add(key)
                uri = file_data.get("fileUri", file_data.get("file_uri"))
                if isinstance(uri, str) and uri:
                    references.append(uri)
        for key, child in item.items():
            if key not in consumed:
                walk(child)

    walk(value)
    return images, references


def _text_fields(value: str | list[_OpenAIPart] | list[_GooglePart]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [part.text for part in value if part.text is not None]


def _task_projection(value: str | list[_OpenAIPart] | list[_GooglePart]) -> tuple[str, str]:
    texts = _text_fields(value)
    task_ids = TASK_ID_PATTERN.findall("\n".join(texts))
    task_id = task_ids[0] if len(task_ids) == 1 else ""
    lines = [
        line.rstrip()
        for text in texts
        for line in text.splitlines()
        if not TASK_METADATA_LINE_PATTERN.fullmatch(line.strip())
    ]
    task_text = "\n".join(lines).strip()
    return task_id, hashlib.sha256(task_text.encode("utf-8")).hexdigest()


def _response_contract(body: Mapping[str, Any], sdk_family: str) -> _ResponseContract:
    if sdk_family == "openai":
        response_format = body.get("response_format")
        json_schema = response_format.get("json_schema") if isinstance(response_format, Mapping) else None
        schema = json_schema.get("schema") if isinstance(json_schema, Mapping) else None
    else:
        generation = body.get("generationConfig")
        candidates = [
            generation[key]
            for key in ("responseSchema", "responseJsonSchema")
            if isinstance(generation, Mapping) and key in generation
        ]
        schema = candidates[0] if len(candidates) == 1 else None
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    required = schema.get("required") if isinstance(schema, Mapping) else None
    forbidden = (
        sorted(set(properties).intersection(BINDING_FIELDS))
        if isinstance(properties, Mapping)
        else []
    )
    expected_properties = {
        "primary_diagnosis": {"type": ["string", "null"]},
        "differentials": {"type": "array", "items": {"type": "string"}},
        "clinical_assessment": {"type": "string"},
    }
    return _ResponseContract(
        additional_properties=(
            schema.get("additionalProperties") if isinstance(schema, Mapping) else None
        ),
        forbidden_binding_fields=forbidden,
        path=(
            "$.response_format.json_schema.schema"
            if sdk_family == "openai"
            else "$.generationConfig.responseJsonSchema"
        ),
        present=isinstance(schema, Mapping),
        properties=sorted(properties) if isinstance(properties, Mapping) else [],
        required=sorted(required) if isinstance(required, list) else [],
        valid=bool(
            isinstance(schema, Mapping)
            and schema.get("type") == "object"
            and dict(properties) == expected_properties
            and isinstance(required, list)
            and set(required) == set(RESPONSE_FIELDS)
            and len(required) == len(RESPONSE_FIELDS)
            and schema.get("additionalProperties") is False
            and not forbidden
        ),
    )


def _materialize(body_bytes: bytes, sdk_family: str, parser_active_field: str) -> _RequestEnvelope:
    try:
        raw = json.loads(body_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("request body is not an object")
    if sdk_family == "openai":
        request = _OpenAIRequest.model_validate(raw)
        active_turns = [turn for turn in request.messages if turn.role == "user"]
    elif sdk_family == "google_genai":
        request = _GoogleRequest.model_validate(raw)
        active_turns = [turn for turn in request.contents if turn.role == "user"]
    else:
        raise ValueError(f"unsupported SDK family: {sdk_family}")
    if not active_turns:
        raise ValueError("request has no user turn")
    active = active_turns[-1].content if sdk_family == "openai" else active_turns[-1].parts
    active_images, active_references = _scan_evidence(active)
    global_images, global_references = _scan_evidence(raw)
    task_id, task_text_sha256 = _task_projection(active)
    return _RequestEnvelope.model_construct(
        active_images=active_images,
        active_reference_count=len(active_references),
        global_reference_count=len(global_references),
        out_of_scope_image_count=len(global_images) - len(active_images),
        parser_active_field=parser_active_field,
        response_contract=_response_contract(raw, sdk_family),
        sdk_family=sdk_family,
        task_id=task_id,
        task_text_sha256=task_text_sha256,
    )


def evaluate_request(
    body_bytes: bytes,
    sdk_family: str,
    expected: Mapping[str, Any],
    *,
    parser_active_field: str | None = None,
) -> tuple[bool, str]:
    try:
        binding = _ExpectedBinding.model_validate(expected)
        envelope = _materialize(
            body_bytes,
            sdk_family,
            parser_active_field or binding.client_response_binding.active_field,
        )
        _RequestEnvelope.model_validate(envelope.model_dump(), context={"expected": binding})
    except (TypeError, ValueError, ValidationError) as exc:
        return False, f"pydantic_integrated_envelope:{str(exc).splitlines()[0]}"
    return True, "pydantic_integrated_envelope_satisfied"


def model_schema_size_bytes() -> int:
    return len(json.dumps(_RequestEnvelope.model_json_schema(), separators=(",", ":"), sort_keys=True).encode("utf-8"))
