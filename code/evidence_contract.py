from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


RESPONSE_FIELDS = (
    "primary_diagnosis",
    "differentials",
    "clinical_assessment",
)

CLIENT_RESPONSE_BINDING_FIELDS = (
    "task_id",
    "task_text_sha256",
    "contract_sha256",
    "contract_version",
    "active_field",
    "correlation_id",
)


class ParserContractMismatch(ValueError):
    pass


class EnvelopeContractMismatch(ValueError):
    pass


class DispatchAnchorMismatch(ValueError):
    pass


@dataclass(frozen=True)
class BoundImage:
    data: bytes
    mime_type: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class EvidenceFieldContract:
    task_id: str
    task_text: str
    images: tuple[BoundImage, ...]
    active_field: str
    contract_version: int

    def __post_init__(self) -> None:
        if self.active_field not in RESPONSE_FIELDS:
            raise ValueError(f"unsupported active field: {self.active_field}")
        if (
            not self.task_id
            or not self.task_text
            or self.contract_version < 1
            or not self.images
        ):
            raise ValueError("contract task, version, and evidence are required")

    @property
    def task_text_sha256(self) -> str:
        return hashlib.sha256(self.task_text.encode("utf-8")).hexdigest()

    def commitment(self) -> dict[str, Any]:
        return {
            "active_field": self.active_field,
            "contract_version": self.contract_version,
            "ordered_image_mime_types": [image.mime_type for image in self.images],
            "ordered_image_sha256": [image.sha256 for image in self.images],
            "task_id": self.task_id,
            "task_text_sha256": self.task_text_sha256,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.commitment(), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def decoded_image_mime_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass(frozen=True)
class LocalDispatchEnvelope:
    correlation_id: str
    contract: EvidenceFieldContract
    envelope_version: int = 1

    def __post_init__(self) -> None:
        if not self.correlation_id or self.envelope_version != 1:
            raise ValueError("correlation ID and envelope version 1 are required")

    def payload(self) -> dict[str, Any]:
        images = [
            {
                "byte_length": len(image.data),
                "data_base64": base64.b64encode(image.data).decode("ascii"),
                "mime_type": image.mime_type,
                "sha256": image.sha256,
            }
            for image in self.contract.images
        ]
        return {
            "canonical_task_text": self.contract.task_text,
            "contract": self.contract.commitment(),
            "contract_sha256": self.contract.sha256,
            "correlation_id": self.correlation_id,
            "envelope_version": self.envelope_version,
            "materialized_images": images,
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def queue_message_sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "LocalDispatchEnvelope":
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvelopeContractMismatch("queue envelope is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise EnvelopeContractMismatch("queue envelope must be an object")
        expected_keys = {
            "canonical_task_text",
            "contract",
            "contract_sha256",
            "correlation_id",
            "envelope_version",
            "materialized_images",
        }
        if set(payload) != expected_keys:
            raise EnvelopeContractMismatch("queue envelope keys do not match contract")
        if canonical_json_bytes(payload) != value:
            raise EnvelopeContractMismatch("queue envelope is not canonical JSON")
        if payload.get("envelope_version") != 1:
            raise EnvelopeContractMismatch("queue envelope version mismatch")
        correlation_id = payload.get("correlation_id")
        task_text = payload.get("canonical_task_text")
        commitment = payload.get("contract")
        image_rows = payload.get("materialized_images")
        if (
            not isinstance(correlation_id, str)
            or not correlation_id
            or not isinstance(task_text, str)
            or not task_text
            or not isinstance(commitment, Mapping)
            or not isinstance(image_rows, list)
            or not image_rows
        ):
            raise EnvelopeContractMismatch("queue envelope fields are invalid")
        expected_contract_keys = {
            "active_field",
            "contract_version",
            "ordered_image_mime_types",
            "ordered_image_sha256",
            "task_id",
            "task_text_sha256",
        }
        if set(commitment) != expected_contract_keys:
            raise EnvelopeContractMismatch("contract commitment keys are invalid")
        if commitment.get("task_text_sha256") != hashlib.sha256(
            task_text.encode("utf-8")
        ).hexdigest():
            raise EnvelopeContractMismatch("task text does not match its commitment")

        images: list[BoundImage] = []
        for row in image_rows:
            if not isinstance(row, Mapping) or set(row) != {
                "byte_length",
                "data_base64",
                "mime_type",
                "sha256",
            }:
                raise EnvelopeContractMismatch("materialized image row is invalid")
            encoded = row.get("data_base64")
            if not isinstance(encoded, str):
                raise EnvelopeContractMismatch("image bytes are not base64 text")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise EnvelopeContractMismatch("image bytes are not valid base64") from exc
            mime_type = row.get("mime_type")
            digest = hashlib.sha256(data).hexdigest()
            if row.get("byte_length") != len(data) or row.get("sha256") != digest:
                raise EnvelopeContractMismatch("image length or digest mismatch")
            if not isinstance(mime_type, str) or decoded_image_mime_type(data) != mime_type:
                raise EnvelopeContractMismatch("image MIME does not match decoded bytes")
            images.append(BoundImage(data=data, mime_type=mime_type))

        try:
            contract = EvidenceFieldContract(
                task_id=commitment["task_id"],
                task_text=task_text,
                images=tuple(images),
                active_field=commitment["active_field"],
                contract_version=commitment["contract_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvelopeContractMismatch("contract fields are invalid") from exc
        if contract.commitment() != dict(commitment):
            raise EnvelopeContractMismatch("materialized evidence does not match contract")
        if payload.get("contract_sha256") != contract.sha256:
            raise EnvelopeContractMismatch("contract digest mismatch")
        return cls(
            correlation_id=correlation_id,
            contract=contract,
            envelope_version=1,
        )


@dataclass(frozen=True)
class ProducerDispatchAnchor:
    correlation_id: str
    contract_sha256: str
    queue_message_sha256: str
    key_id: str
    mac_sha256: str
    anchor_version: int = 1

    def __post_init__(self) -> None:
        digest_fields = (self.contract_sha256, self.queue_message_sha256, self.mac_sha256)
        if (
            not self.correlation_id
            or not self.key_id
            or self.anchor_version != 1
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digest_fields
            )
        ):
            raise ValueError("dispatch anchor fields are invalid")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "anchor_version": self.anchor_version,
            "contract_sha256": self.contract_sha256,
            "correlation_id": self.correlation_id,
            "key_id": self.key_id,
            "queue_message_sha256": self.queue_message_sha256,
        }

    def payload(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "mac_sha256": self.mac_sha256}

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @classmethod
    def issue(
        cls,
        envelope: LocalDispatchEnvelope,
        *,
        key_id: str,
        signing_key: bytes,
    ) -> "ProducerDispatchAnchor":
        if not signing_key:
            raise ValueError("dispatch anchor signing key is required")
        unsigned = {
            "anchor_version": 1,
            "contract_sha256": envelope.contract.sha256,
            "correlation_id": envelope.correlation_id,
            "key_id": key_id,
            "queue_message_sha256": envelope.queue_message_sha256,
        }
        mac_sha256 = hmac.new(
            signing_key,
            canonical_json_bytes(unsigned),
            hashlib.sha256,
        ).hexdigest()
        return cls(mac_sha256=mac_sha256, **unsigned)

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "ProducerDispatchAnchor":
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DispatchAnchorMismatch("dispatch anchor is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise DispatchAnchorMismatch("dispatch anchor must be an object")
        expected_keys = {
            "anchor_version",
            "contract_sha256",
            "correlation_id",
            "key_id",
            "mac_sha256",
            "queue_message_sha256",
        }
        if set(payload) != expected_keys:
            raise DispatchAnchorMismatch("dispatch anchor keys do not match contract")
        if canonical_json_bytes(payload) != value:
            raise DispatchAnchorMismatch("dispatch anchor is not canonical JSON")
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise DispatchAnchorMismatch("dispatch anchor fields are invalid") from exc

    def verify(
        self,
        queue_bytes: bytes,
        *,
        verification_key: bytes,
    ) -> LocalDispatchEnvelope:
        if not verification_key:
            raise DispatchAnchorMismatch("dispatch anchor verification key is required")
        expected_mac = hmac.new(
            verification_key,
            canonical_json_bytes(self.unsigned_payload()),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(self.mac_sha256, expected_mac):
            raise DispatchAnchorMismatch("dispatch anchor authentication failed")
        if not hmac.compare_digest(
            self.queue_message_sha256,
            hashlib.sha256(queue_bytes).hexdigest(),
        ):
            raise DispatchAnchorMismatch("queue message does not match trusted anchor")
        try:
            envelope = LocalDispatchEnvelope.from_json_bytes(queue_bytes)
        except EnvelopeContractMismatch as exc:
            raise DispatchAnchorMismatch("queue envelope failed contract verification") from exc
        if envelope.correlation_id != self.correlation_id:
            raise DispatchAnchorMismatch("queue correlation does not match trusted anchor")
        if not hmac.compare_digest(envelope.contract.sha256, self.contract_sha256):
            raise DispatchAnchorMismatch("queue contract does not match trusted anchor")
        return envelope


def verify_detached_dispatch(
    queue_bytes: bytes,
    anchor_bytes: bytes,
    *,
    verification_key: bytes,
) -> tuple[LocalDispatchEnvelope, ProducerDispatchAnchor]:
    anchor = ProducerDispatchAnchor.from_json_bytes(anchor_bytes)
    envelope = anchor.verify(queue_bytes, verification_key=verification_key)
    return envelope, anchor


@dataclass(frozen=True)
class ClientOwnedResponseBinding:
    task_id: str
    task_text_sha256: str
    contract_sha256: str
    contract_version: int
    active_field: str
    correlation_id: str

    def __post_init__(self) -> None:
        digests = (self.task_text_sha256, self.contract_sha256)
        if (
            not self.task_id
            or not self.correlation_id
            or self.contract_version < 1
            or self.active_field not in RESPONSE_FIELDS
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
        ):
            raise ValueError("client-owned response binding is invalid")

    @classmethod
    def from_contract(
        cls, contract: EvidenceFieldContract, *, correlation_id: str
    ) -> "ClientOwnedResponseBinding":
        return cls(
            task_id=contract.task_id,
            task_text_sha256=contract.task_text_sha256,
            contract_sha256=contract.sha256,
            contract_version=contract.contract_version,
            active_field=contract.active_field,
            correlation_id=correlation_id,
        )

    def commitment(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text_sha256": self.task_text_sha256,
            "contract_sha256": self.contract_sha256,
            "contract_version": self.contract_version,
            "active_field": self.active_field,
            "correlation_id": self.correlation_id,
        }


@dataclass(frozen=True)
class TrustedResponseReceipt:
    correlation_id: str
    contract_sha256: str
    dispatch_body_sha256: str
    provider_response_id: str
    provider_payload_sha256: str
    key_id: str
    mac_sha256: str
    receipt_version: int = 1

    def __post_init__(self) -> None:
        digests = (
            self.contract_sha256,
            self.dispatch_body_sha256,
            self.provider_payload_sha256,
            self.mac_sha256,
        )
        if (
            not self.correlation_id
            or not self.provider_response_id
            or not self.key_id
            or self.receipt_version != 1
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
        ):
            raise ValueError("trusted response receipt is invalid")

    @staticmethod
    def payload_sha256(provider_payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(canonical_json_bytes(dict(provider_payload))).hexdigest()

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "correlation_id": self.correlation_id,
            "dispatch_body_sha256": self.dispatch_body_sha256,
            "key_id": self.key_id,
            "provider_payload_sha256": self.provider_payload_sha256,
            "provider_response_id": self.provider_response_id,
            "receipt_version": self.receipt_version,
        }

    def commitment(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "mac_sha256": self.mac_sha256}

    @classmethod
    def issue(
        cls,
        binding: ClientOwnedResponseBinding,
        *,
        dispatch_body_sha256: str,
        provider_response_id: str,
        provider_payload: Mapping[str, Any],
        key_id: str,
        signing_key: bytes,
    ) -> "TrustedResponseReceipt":
        if not signing_key:
            raise ValueError("response receipt signing key is required")
        unsigned = {
            "contract_sha256": binding.contract_sha256,
            "correlation_id": binding.correlation_id,
            "dispatch_body_sha256": dispatch_body_sha256,
            "key_id": key_id,
            "provider_payload_sha256": cls.payload_sha256(provider_payload),
            "provider_response_id": provider_response_id,
            "receipt_version": 1,
        }
        mac_sha256 = hmac.new(
            signing_key,
            canonical_json_bytes(unsigned),
            hashlib.sha256,
        ).hexdigest()
        return cls(mac_sha256=mac_sha256, **unsigned)

    def verify(
        self,
        binding: ClientOwnedResponseBinding,
        *,
        expected_dispatch_body_sha256: str,
        provider_payload: Mapping[str, Any],
        verification_key: bytes,
    ) -> None:
        if not verification_key:
            raise ParserContractMismatch("response receipt verification key is required")
        expected_mac = hmac.new(
            verification_key,
            canonical_json_bytes(self.unsigned_payload()),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(self.mac_sha256, expected_mac):
            raise ParserContractMismatch("response receipt authentication failed")
        if (
            self.contract_sha256 != binding.contract_sha256
            or self.correlation_id != binding.correlation_id
            or self.dispatch_body_sha256 != expected_dispatch_body_sha256
        ):
            raise ParserContractMismatch("response receipt does not match dispatch")
        observed_payload_sha256 = self.payload_sha256(provider_payload)
        if not hmac.compare_digest(
            self.provider_payload_sha256,
            observed_payload_sha256,
        ):
            raise ParserContractMismatch("response payload does not match trusted receipt")


@dataclass(frozen=True)
class ClientOwnedResponseEnvelope:
    binding: ClientOwnedResponseBinding
    provider_payload: Mapping[str, Any]
    receipt: TrustedResponseReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ClientOwnedResponseBinding):
            raise ParserContractMismatch("client-owned response binding is invalid")
        if not isinstance(self.receipt, TrustedResponseReceipt):
            raise ParserContractMismatch("trusted response receipt is invalid")
        payload = dict(self.provider_payload)
        if set(payload) != set(RESPONSE_FIELDS):
            raise ParserContractMismatch(
                "provider response fields do not match the semantic response contract"
            )
        primary = payload["primary_diagnosis"]
        differentials = payload["differentials"]
        assessment = payload["clinical_assessment"]
        if primary is not None and not isinstance(primary, str):
            raise ParserContractMismatch("primary diagnosis has an invalid type")
        if not isinstance(differentials, list) or not all(
            isinstance(value, str) for value in differentials
        ):
            raise ParserContractMismatch("differentials have an invalid type")
        if not isinstance(assessment, str):
            raise ParserContractMismatch("clinical assessment has an invalid type")
        object.__setattr__(self, "provider_payload", MappingProxyType(payload))

    @classmethod
    def from_trusted_completion(
        cls,
        binding: ClientOwnedResponseBinding,
        provider_payload: Mapping[str, Any],
        receipt: TrustedResponseReceipt,
    ) -> "ClientOwnedResponseEnvelope":
        return cls(
            binding=binding,
            provider_payload=provider_payload,
            receipt=receipt,
        )


def response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primary_diagnosis": {"type": ["string", "null"]},
            "differentials": {"type": "array", "items": {"type": "string"}},
            "clinical_assessment": {"type": "string"},
        },
        "required": list(RESPONSE_FIELDS),
        "additionalProperties": False,
    }


def inspect_serialized_response_schema(
    body: Mapping[str, Any], sdk_family: str
) -> dict[str, Any]:
    if sdk_family == "openai":
        path = "$.response_format.json_schema.schema"
        response_format = body.get("response_format")
        schema = (
            response_format.get("json_schema", {}).get("schema")
            if isinstance(response_format, Mapping)
            else None
        )
    elif sdk_family == "google_genai":
        generation = body.get("generationConfig")
        path = "$.generationConfig.responseJsonSchema"
        schema = (
            generation.get("responseJsonSchema")
            if isinstance(generation, Mapping)
            else None
        )
    else:
        raise ValueError(f"unsupported SDK family: {sdk_family}")

    result: dict[str, Any] = {
        "additional_properties": None,
        "forbidden_binding_fields": [],
        "path": path,
        "present": isinstance(schema, Mapping),
        "properties": [],
        "required": [],
        "valid": False,
    }
    if not isinstance(schema, Mapping):
        return result
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return result

    forbidden = sorted(set(properties).intersection(CLIENT_RESPONSE_BINDING_FIELDS))
    expected_properties = response_json_schema()["properties"]
    valid = bool(
        dict(properties) == expected_properties
        and set(required) == set(RESPONSE_FIELDS)
        and len(required) == len(RESPONSE_FIELDS)
        and schema.get("additionalProperties") is False
        and not forbidden
    )
    result.update(
        {
            "additional_properties": schema.get("additionalProperties"),
            "forbidden_binding_fields": forbidden,
            "properties": sorted(properties),
            "required": list(required),
            "valid": valid,
        }
    )
    return result


def mock_structured_response(contract: EvidenceFieldContract) -> dict[str, Any]:
    return {
        "primary_diagnosis": f"bound-primary::{contract.task_id}",
        "differentials": [f"decoy-differential::{contract.task_id}"],
        "clinical_assessment": f"decoy-assessment::{contract.task_id}",
    }


@dataclass(frozen=True)
class ParsedField:
    task_id: str
    contract_version: int
    field_name: str
    value: Any
    correlation_id: str
    dispatch_body_sha256: str
    provider_response_id: str
    provider_payload_sha256: str


@dataclass(frozen=True)
class BoundResponseParser:
    contract: EvidenceFieldContract
    response_binding: ClientOwnedResponseBinding
    selected_field: str
    expected_dispatch_body_sha256: str
    response_receipt_verification_key: bytes

    def __post_init__(self) -> None:
        if self.selected_field != self.contract.active_field:
            raise ParserContractMismatch(
                f"parser selects {self.selected_field}, contract selects "
                f"{self.contract.active_field}"
            )
        expected_binding = ClientOwnedResponseBinding.from_contract(
            self.contract,
            correlation_id=self.response_binding.correlation_id,
        )
        if self.response_binding != expected_binding:
            raise ParserContractMismatch(
                "client-owned response binding does not match parser contract"
            )
        if (
            len(self.expected_dispatch_body_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_dispatch_body_sha256
            )
            or not self.response_receipt_verification_key
        ):
            raise ParserContractMismatch("parser dispatch receipt configuration is invalid")

    @classmethod
    def from_contract(
        cls,
        contract: EvidenceFieldContract,
        *,
        correlation_id: str,
        expected_dispatch_body_sha256: str,
        response_receipt_verification_key: bytes,
    ) -> "BoundResponseParser":
        binding = ClientOwnedResponseBinding.from_contract(
            contract, correlation_id=correlation_id
        )
        return cls(
            contract=contract,
            response_binding=binding,
            selected_field=contract.active_field,
            expected_dispatch_body_sha256=expected_dispatch_body_sha256,
            response_receipt_verification_key=response_receipt_verification_key,
        )

    def parse(self, response: ClientOwnedResponseEnvelope) -> ParsedField:
        if not isinstance(response, ClientOwnedResponseEnvelope):
            raise ParserContractMismatch("parser requires a client-owned response envelope")
        if response.binding != self.response_binding:
            raise ParserContractMismatch(
                "response binding does not match parser contract"
            )
        response.receipt.verify(
            response.binding,
            expected_dispatch_body_sha256=self.expected_dispatch_body_sha256,
            provider_payload=response.provider_payload,
            verification_key=self.response_receipt_verification_key,
        )
        if self.selected_field not in response.provider_payload:
            raise ParserContractMismatch("contract field is absent from provider response")
        return ParsedField(
            task_id=self.contract.task_id,
            contract_version=self.contract.contract_version,
            field_name=self.selected_field,
            value=response.provider_payload[self.selected_field],
            correlation_id=response.receipt.correlation_id,
            dispatch_body_sha256=response.receipt.dispatch_body_sha256,
            provider_response_id=response.receipt.provider_response_id,
            provider_payload_sha256=response.receipt.provider_payload_sha256,
        )
