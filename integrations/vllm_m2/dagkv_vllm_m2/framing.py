"""Canonical byte observation for one complete KV block."""

from dataclasses import dataclass
from hashlib import sha256
from struct import pack
from typing import Any

PAYLOAD_FRAMING = "DAGKV_PAYLOAD_V1"
_MAGIC = b"DAGKV_PAYLOAD_V1\x00"


@dataclass(frozen=True, slots=True)
class PayloadObservation:
    digest: str
    payload_bytes: int


def group_payload_bytes(kv_caches: Any, group_idx: int = 0) -> int:
    refs = kv_caches.group_data_refs[group_idx]
    return sum(int(ref.page_size_bytes) for ref in refs)


def _row_bytes(tensor: Any, block_id: int, size: int) -> bytes:
    if type(block_id) is not int or block_id < 0:
        raise ValueError("physical block ID must be a non-negative integer")
    if type(size) is not int or size <= 0:
        raise ValueError("page_size_bytes must be a positive integer")
    if block_id >= tensor.shape[0]:
        raise ValueError(f"physical block {block_id} is outside tensor allocation")
    row = tensor[block_id].view(-1)
    if row.numel() < size:
        raise ValueError("canonical tensor row is smaller than unpadded page size")
    return row[:size].detach().contiguous().cpu().numpy().tobytes()


def observe_payload(
    tensors: list[Any],
    group_data_refs: list[list[Any]],
    block_id: int,
    *,
    group_idx: int = 0,
) -> PayloadObservation:
    """Hash ordered, unpadded ref bytes with unambiguous canonical framing."""

    if group_idx != 0 or len(group_data_refs) != 1:
        raise ValueError("M2 diagnostics require exactly one KV cache group")
    refs = group_data_refs[group_idx]
    if not refs:
        raise ValueError("KV cache group must contain at least one data reference")

    digest = sha256()
    digest.update(_MAGIC)
    digest.update(pack(">II", group_idx, len(refs)))
    payload_bytes = 0
    for ordinal, ref in enumerate(refs):
        tensor_idx = int(ref.tensor_idx)
        page_size = int(ref.page_size_bytes)
        if tensor_idx < 0 or tensor_idx >= len(tensors):
            raise ValueError("canonical data reference has an invalid tensor index")
        payload = _row_bytes(tensors[tensor_idx], block_id, page_size)
        if len(payload) != page_size:
            raise ValueError("captured payload length differs from page_size_bytes")
        digest.update(pack(">IIQ", ordinal, tensor_idx, page_size))
        digest.update(payload)
        payload_bytes += page_size
    return PayloadObservation(digest.hexdigest(), payload_bytes)
