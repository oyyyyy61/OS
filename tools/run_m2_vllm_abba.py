#!/usr/bin/env python3
"""Run the frozen M2 vLLM KV-cache ABBA calibration protocol.

The default mode only calibrates numerical equivalence. It can never accept M2.
Formal holdouts require a frozen calibration cohort and tolerance before startup.
All vLLM and NumPy imports are delayed so helper tests remain CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

try:
    from tools.aggregate_m2_calibration import (
        _validate_run as _validate_calibration_run,
    )
    from tools.m2_calibration_evidence import (
        CALIBRATION_COHORT_SCHEMA,  # noqa: F401
        CalibrationEvidenceError,
        read_stable_bytes,
        validate_published_calibration_bundle,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from aggregate_m2_calibration import (  # type: ignore[no-redef]
        _validate_run as _validate_calibration_run,
    )
    from m2_calibration_evidence import (  # type: ignore[no-redef]
        CALIBRATION_COHORT_SCHEMA,  # noqa: F401
        CalibrationEvidenceError,
        read_stable_bytes,
        validate_published_calibration_bundle,
    )

PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v2"
DIAGNOSTIC_SCHEMA = "dagkv.vllm_m2.transfer_probe.v1"
TOLERANCE_SCHEMA = "dagkv.m2.frozen_tolerance.v2"
ITEM8_FORMAL_RUN_SCHEMA = "dagkv.m2.item8.formal_run.v1"
PROMPT_TOKEN_IDS = tuple(range(1000, 1017))
FROZEN_SEED = 20260724
FROZEN_QWEN3_8B_VOCAB_SIZE = 151_936
BLOCK_SIZE = 16
EXPECTED_EXTERNAL_TOKENS = 16
MAX_FORMAL_ATOL = 0.125
MIN_CALIBRATION_RUNS = 59
TOLERANCE_DERIVATION = "fixed_binary_cap_from_excluded_pilot"
DEFAULT_MODEL = Path("/home/data/25_oyzx/moqae_runtime_gpu/modelscope/Qwen/Qwen3-8B")
DEFAULT_VLLM_ROOT = Path("/home/data/25_oyzx/Agentrix/vllm")
REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = REPO_ROOT / "integrations" / "vllm_m2"
PROTOCOL_SOURCE = REPO_ROOT / "research" / "protocols" / ("M2_VLLM_REPLAY_PROTOCOL.md")


class M2ValidationError(RuntimeError):
    """Raised when an M2 evidence invariant is absent or violated."""


@dataclass(frozen=True, slots=True)
class FrozenTolerance:
    atol: float
    rtol: float
    frozen_at_utc: str
    calibration_manifest_sha256: str
    reproducibility_fingerprint: str
    calibration_run_count: int
    derivation: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class LogitComparison:
    left: str
    right: str
    token_equal: bool
    allclose: bool
    max_abs_error: float
    max_rel_error: float


@dataclass(slots=True)
class Measurement:
    phase: str
    request_id: str
    trace_id: str
    token_id: int
    num_cached_tokens: int
    elapsed_ms: float
    top1_margin: float
    logits: Any


def require(condition: bool, message: str) -> None:
    """Raise a stable validation error when a paper-facing invariant fails."""

    if not condition:
        raise M2ValidationError(message)


def _regular_file_identity(path: Path, *, label: str) -> tuple[int, int, int, int, int]:
    """Capture the mutation-sensitive identity of one non-symlink input file."""

    try:
        value = path.lstat()
    except OSError as exc:
        raise M2ValidationError(f"cannot inspect {label} at {path}: {exc}") from exc
    require(stat.S_ISREG(value.st_mode), f"{label} must be a regular file: {path}")
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def validate_prompt_tokens(
    token_ids: Sequence[int], *, block_size: int = BLOCK_SIZE
) -> tuple[int, ...]:
    """Validate the one-full-block-plus-one-token M2 prompt."""

    values = tuple(token_ids)
    require(block_size == 16, "M2 requires vLLM block_size=16")
    require(len(values) == block_size + 1, "M2 prompt must contain exactly 17 tokens")
    require(
        all(type(token_id) is int and token_id >= 0 for token_id in values),
        "prompt token IDs must be non-negative integers",
    )
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    """Read JSONL without repairing blank, partial, or non-object rows."""

    require(path.is_file(), f"required trace is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(
                line.endswith("\n"), f"unterminated JSONL row at {path}:{line_number}"
            )
            require(line.strip() != "", f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise M2ValidationError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            require(
                isinstance(row, dict),
                f"JSONL row must be an object at {path}:{line_number}",
            )
            rows.append(row)
    require(rows, f"required trace is empty: {path}")
    return rows


def dense_logits_from_logprobs(logprobs: Any, vocab_size: int) -> Any:
    """Reconstruct one full-vocabulary raw-logit vector from vLLM output."""

    import numpy as np

    require(type(vocab_size) is int and vocab_size > 0, "vocab_size must be positive")
    require(logprobs is not None, "vLLM did not return sample raw logits")
    require(len(logprobs) == 1, "M2 measurement must contain exactly one decode step")

    if all(
        hasattr(logprobs, name)
        for name in ("start_indices", "end_indices", "token_ids", "logprobs")
    ):
        start = logprobs.start_indices[0]
        end = logprobs.end_indices[0]
        token_ids = list(logprobs.token_ids[start:end])
        values = list(logprobs.logprobs[start:end])
    else:
        position = logprobs[0]
        require(isinstance(position, dict), "sample raw logits must be a token map")
        token_ids = list(position)
        values = [position[token_id].logprob for token_id in token_ids]

    require(len(token_ids) == len(values), "raw-logit token/value lengths differ")
    vector = np.full(vocab_size, np.nan, dtype=np.float64)
    seen: set[int] = set()
    for token_id, value in zip(token_ids, values, strict=True):
        require(type(token_id) is int, "raw-logit token ID must be an integer")
        require(
            0 <= token_id < vocab_size, f"raw-logit token ID out of range: {token_id}"
        )
        numeric = float(value)
        require(math.isfinite(numeric), f"non-finite raw logit for token {token_id}")
        if token_id in seen:
            require(
                float(vector[token_id]) == numeric,
                f"conflicting duplicate raw logit for token {token_id}",
            )
        vector[token_id] = numeric
        seen.add(token_id)

    require(
        len(seen) == vocab_size,
        f"raw logits cover {len(seen)} of {vocab_size} vocabulary entries",
    )
    require(
        bool(np.isfinite(vector).all()), "raw-logit vector contains NaN or infinity"
    )
    return vector


def compare_logit_vectors(
    left_name: str,
    left_token: int,
    left: Any,
    right_name: str,
    right_token: int,
    right: Any,
    *,
    atol: float,
    rtol: float,
) -> LogitComparison:
    """Compare exact token output and complete raw-logit vectors."""

    import numpy as np

    require(atol >= 0.0 and math.isfinite(atol), "atol must be finite and non-negative")
    require(rtol >= 0.0 and math.isfinite(rtol), "rtol must be finite and non-negative")
    require(
        left.shape == right.shape,
        f"logit shape mismatch: {left.shape} != {right.shape}",
    )
    require(
        bool(np.isfinite(left).all()), f"{left_name} logits contain non-finite values"
    )
    require(
        bool(np.isfinite(right).all()), f"{right_name} logits contain non-finite values"
    )

    abs_error = np.abs(left - right)
    denominator = np.maximum(np.abs(left), np.abs(right))
    rel_error = np.divide(
        abs_error,
        denominator,
        out=np.zeros_like(abs_error),
        where=denominator != 0,
    )
    return LogitComparison(
        left=left_name,
        right=right_name,
        token_equal=left_token == right_token,
        allclose=bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False)),
        max_abs_error=float(abs_error.max(initial=0.0)),
        max_rel_error=float(rel_error.max(initial=0.0)),
    )


def load_frozen_tolerance(path: Path, *, run_started_ns: int) -> FrozenTolerance:
    """Load a formal tolerance file proven to predate this process run."""

    identity = _regular_file_identity(path, label="formal tolerance file")
    require(
        identity[3] < run_started_ns,
        "formal tolerance file must be frozen before runner startup",
    )
    try:
        raw = read_stable_bytes(path, label="formal tolerance file")
        payload = json.loads(raw.decode("utf-8"))
    except CalibrationEvidenceError as exc:
        raise M2ValidationError(str(exc)) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M2ValidationError(f"invalid formal tolerance file: {exc}") from exc
    require(isinstance(payload, dict), "formal tolerance payload must be an object")
    require(
        set(payload)
        == {
            "schema_version",
            "frozen",
            "frozen_at_utc",
            "atol",
            "rtol",
            "calibration_manifest_sha256",
            "reproducibility_fingerprint",
            "calibration_run_count",
            "derivation",
        },
        "formal tolerance fields differ from the frozen schema",
    )
    require(payload.get("schema_version") == TOLERANCE_SCHEMA, "wrong tolerance schema")
    require(payload.get("frozen") is True, "formal tolerance file must set frozen=true")

    atol = payload.get("atol")
    rtol = payload.get("rtol")
    require(type(atol) in (int, float), "frozen atol must be numeric")
    require(type(rtol) in (int, float), "frozen rtol must be numeric")
    atol = float(atol)
    rtol = float(rtol)
    require(
        0.0 <= atol <= MAX_FORMAL_ATOL and math.isfinite(atol),
        f"frozen atol must be within [0, {MAX_FORMAL_ATOL}]",
    )
    require(rtol == 0.0, "formal rtol is frozen at zero")

    frozen_at = payload.get("frozen_at_utc")
    require(isinstance(frozen_at, str) and frozen_at, "frozen_at_utc is required")
    try:
        timestamp = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise M2ValidationError("frozen_at_utc must be ISO 8601") from exc
    require(timestamp.tzinfo is not None, "frozen_at_utc must include a timezone")
    require(
        timestamp.astimezone(UTC).timestamp() * 1_000_000_000 < run_started_ns,
        "frozen_at_utc must predate runner startup",
    )

    digests = {
        "calibration_manifest_sha256": payload.get("calibration_manifest_sha256"),
        "reproducibility_fingerprint": payload.get("reproducibility_fingerprint"),
    }
    for name, digest in digests.items():
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            f"{name} must be a lowercase SHA-256 digest",
        )
    calibration_run_count = payload.get("calibration_run_count")
    require(
        type(calibration_run_count) is int
        and calibration_run_count == MIN_CALIBRATION_RUNS,
        f"formal tolerance requires exactly {MIN_CALIBRATION_RUNS} calibrations",
    )
    derivation = payload.get("derivation")
    require(
        derivation == TOLERANCE_DERIVATION,
        "formal tolerance uses an unregistered derivation",
    )
    require(
        _regular_file_identity(path, label="formal tolerance file") == identity,
        "formal tolerance file changed during validation",
    )
    return FrozenTolerance(
        atol=atol,
        rtol=rtol,
        frozen_at_utc=frozen_at,
        calibration_manifest_sha256=digests["calibration_manifest_sha256"],
        reproducibility_fingerprint=digests["reproducibility_fingerprint"],
        calibration_run_count=calibration_run_count,
        derivation=derivation,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_calibration_cohort(
    path: Path,
    *,
    frozen_tolerance: FrozenTolerance,
    run_started_ns: int,
    expected_implementation_manifest_sha256: str,
) -> dict[str, Any]:
    """Revalidate the frozen cohort and its complete upstream bundle."""

    identity = _regular_file_identity(path, label="calibration cohort manifest")
    require(
        identity[3] < run_started_ns,
        "calibration cohort manifest must predate runner startup",
    )
    try:
        payload, _, evidence = validate_published_calibration_bundle(
            path,
            run_validator=_validate_calibration_run,
            expected_manifest_sha256=(frozen_tolerance.calibration_manifest_sha256),
            expected_implementation_manifest_sha256=(
                expected_implementation_manifest_sha256
            ),
        )
    except CalibrationEvidenceError as exc:
        raise M2ValidationError(str(exc)) from exc
    run_count = payload.get("run_count")
    require(
        run_count == frozen_tolerance.calibration_run_count
        and type(run_count) is int
        and run_count == MIN_CALIBRATION_RUNS,
        "calibration cohort run count differs from the frozen tolerance",
    )
    require(
        payload.get("formal_atol") == frozen_tolerance.atol,
        "calibration cohort atol differs from the frozen tolerance",
    )
    require(
        payload.get("formal_rtol") == frozen_tolerance.rtol == 0.0,
        "calibration cohort rtol differs from the frozen tolerance",
    )
    observed_max = payload.get("observed_max_abs_error")
    require(
        type(observed_max) in (int, float)
        and math.isfinite(float(observed_max))
        and 0.0 <= float(observed_max) <= frozen_tolerance.atol,
        "calibration cohort exceeds the frozen absolute tolerance",
    )
    require(
        payload.get("reproducibility_fingerprint")
        == frozen_tolerance.reproducibility_fingerprint,
        "calibration cohort reproducibility fingerprint differs",
    )
    require(len(evidence.runs) == run_count, "cohort evidence run count drifted")
    require(
        _regular_file_identity(path, label="calibration cohort manifest") == identity,
        "calibration cohort manifest changed during validation",
    )
    return payload


def validate_prefetch_result(result: dict[str, Any]) -> None:
    """Require one completed external 16-token H2D prefetch."""

    require(isinstance(result, dict), "prefetch result must be an object")
    require(result.get("started") is True, f"prefetch did not start: {result}")
    require(result.get("completed") is True, f"prefetch did not complete: {result}")
    require(result.get("reason") == "completed", f"prefetch terminal reason: {result}")
    require(
        result.get("local_gpu_hit_tokens") == 0, "prefetch must start after GPU reset"
    )
    local_tokens = result.get("local_gpu_hit_tokens")
    external_tokens = result.get("external_hit_tokens")
    if external_tokens is None and result.get("lookup_pending") is True:
        loaded_tokens = result.get("loaded_tokens")
        if type(loaded_tokens) is int and type(local_tokens) is int:
            external_tokens = loaded_tokens - local_tokens
    require(
        external_tokens == EXPECTED_EXTERNAL_TOKENS,
        "prefetch must find exactly one external 16-token block",
    )
    require(
        result.get("loaded_tokens") == EXPECTED_EXTERNAL_TOKENS,
        "prefetch must load exactly 16 tokens",
    )


def _terminal_endpoint(row: dict[str, Any], name: str, tier: str) -> dict[str, Any]:
    endpoint = row.get(name)
    require(isinstance(endpoint, dict), f"diagnostic terminal lacks {name} endpoint")
    require(endpoint.get("tier") == tier, f"{name} tier must be {tier}")
    slot = endpoint.get("physical_slot")
    generation = endpoint.get("allocation_generation")
    digest = endpoint.get("digest")
    require(type(slot) is int and slot >= 0, f"{name} physical_slot is invalid")
    require(
        type(generation) is int and generation > 0,
        f"{name} allocation_generation is invalid",
    )
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        f"{name} digest is not lowercase SHA-256",
    )
    return endpoint


def validate_diagnostic_transfer(
    rows: Sequence[dict[str, Any]],
    *,
    request_id: str,
    direction: str,
    run_id: str,
) -> dict[str, Any]:
    """Validate one submitted/completed diagnostic DMA job."""

    require(direction in {"D2H", "H2D"}, f"unsupported direction: {direction}")
    selected = [
        row
        for row in rows
        if row.get("request_id") == request_id and row.get("direction") == direction
    ]
    require(selected, f"no {direction} diagnostic rows for request {request_id}")
    require(
        all(row.get("schema_version") == DIAGNOSTIC_SCHEMA for row in selected),
        f"wrong diagnostic schema for {request_id}",
    )
    require(
        all(row.get("run_id") == run_id for row in selected),
        f"diagnostic run_id mismatch for {request_id}",
    )
    submitted = [row for row in selected if row.get("event") == "submitted"]
    terminal = [row for row in selected if row.get("event") == "terminal"]
    require(len(submitted) == 1, f"expected one submitted {direction} row")
    require(len(terminal) == 1, f"expected one terminal {direction} row")
    require(
        submitted[0].get("job_id") == terminal[0].get("job_id"),
        f"{direction} submitted/terminal job IDs differ",
    )
    row = terminal[0]
    require(row.get("status") == "completed", f"{direction} diagnostic failed: {row}")
    require(row.get("failure_reason") in (None, ""), f"{direction} has failure reason")
    require(row.get("framing") == "DAGKV_PAYLOAD_V1", "unexpected DMA framing")
    payload_bytes = row.get("payload_bytes")
    require(
        type(payload_bytes) is int and payload_bytes > 0, "payload_bytes is invalid"
    )
    require(row.get("reported_bytes") == payload_bytes, "reported DMA bytes differ")

    source_tier, target_tier = ("GPU", "CPU") if direction == "D2H" else ("CPU", "GPU")
    source = _terminal_endpoint(row, "source", source_tier)
    target = _terminal_endpoint(row, "target", target_tier)
    require(source["digest"] == target["digest"], f"{direction} payload digest changed")
    return row


def validate_diagnostic_trace_closed_set(
    rows: Sequence[dict[str, Any]],
    *,
    run_id: str,
    expected_transfers: Sequence[tuple[str, str]],
) -> None:
    """Reject every unaccounted, duplicate, malformed, or failed DMA row."""

    expected = set(expected_transfers)
    require(expected, "diagnostic closed set must contain at least one transfer")
    require(
        len(expected) == len(expected_transfers),
        "diagnostic expected transfer identities must be unique",
    )
    for request_id, direction in expected:
        require(
            isinstance(request_id, str) and request_id, "empty diagnostic request ID"
        )
        require(direction in {"D2H", "H2D"}, "invalid expected DMA direction")

    require(
        len(rows) == 2 * len(expected),
        "diagnostic trace must contain exactly submitted/terminal pairs",
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    job_to_transfer: dict[int, tuple[str, str]] = {}
    for row in rows:
        require(
            row.get("schema_version") == DIAGNOSTIC_SCHEMA, "wrong diagnostic schema"
        )
        require(row.get("run_id") == run_id, "diagnostic trace contains another run")
        require(row.get("phase") == "ABBA", "diagnostic trace contains another phase")
        request_id = row.get("request_id")
        direction = row.get("direction")
        event = row.get("event")
        job_id = row.get("job_id")
        require(
            isinstance(request_id, str) and request_id, "diagnostic request ID missing"
        )
        require(direction in {"D2H", "H2D"}, "diagnostic DMA direction is invalid")
        require(event in {"submitted", "terminal"}, "diagnostic event is invalid")
        require(type(job_id) is int and job_id >= 0, "diagnostic job ID is invalid")
        require(row.get("framing") == "DAGKV_PAYLOAD_V1", "unexpected DMA framing")
        transfer = (request_id, direction)
        grouped.setdefault(transfer, []).append(row)
        previous = job_to_transfer.setdefault(job_id, transfer)
        require(previous == transfer, "diagnostic job ID was reused across transfers")

    require(set(grouped) == expected, "diagnostic trace contains unexpected transfers")
    require(
        len(job_to_transfer) == len(expected),
        "each diagnostic transfer must use one unique job ID",
    )
    for transfer, pair in grouped.items():
        require(len(pair) == 2, f"diagnostic transfer {transfer} lacks an event pair")
        require(
            len({row["job_id"] for row in pair}) == 1,
            f"diagnostic transfer {transfer} changed job ID",
        )
        by_event = {row["event"]: row for row in pair}
        require(
            set(by_event) == {"submitted", "terminal"},
            f"diagnostic transfer {transfer} has duplicate events",
        )
        require(
            by_event["submitted"].get("status") == "in_flight",
            f"diagnostic transfer {transfer} has invalid submit status",
        )
        require(
            by_event["terminal"].get("status") == "completed",
            f"diagnostic transfer {transfer} did not complete successfully",
        )


def validate_native_transfer(
    rows: Sequence[dict[str, Any]],
    *,
    trace_id: str,
    direction: str,
) -> int:
    """Require native scheduling, completion, and lifecycle evidence."""

    require(direction in {"D2H", "H2D"}, f"unsupported direction: {direction}")
    selected = [row for row in rows if row.get("trace_id") == trace_id]
    require(selected, f"native trace lacks trace_id={trace_id}")
    prefix = "store" if direction == "D2H" else "load"
    action = "save" if direction == "D2H" else "prefetch"
    scheduled = [row for row in selected if row.get("event") == f"{prefix}_scheduled"]
    completed = [row for row in selected if row.get("event") == f"{prefix}_complete"]
    require(len(scheduled) == 1, f"native {prefix}_scheduled count must be one")
    require(len(completed) == 1, f"native {prefix}_complete count must be one")
    byte_key = f"native_{prefix}_bytes"
    declared_bytes = scheduled[0].get(byte_key)
    require(type(declared_bytes) is int and declared_bytes > 0, f"invalid {byte_key}")
    require(
        completed[0].get(byte_key) == declared_bytes, f"native {prefix} bytes differ"
    )

    lifecycle = [
        row
        for row in selected
        if row.get("event") == "kv_lifecycle" and row.get("action") == action
    ]
    lifecycle_scheduled = [row for row in lifecycle if row.get("status") == "scheduled"]
    lifecycle_completed = [row for row in lifecycle if row.get("status") == "completed"]
    require(
        len(lifecycle_scheduled) == 1, f"native {action} lifecycle schedule missing"
    )
    require(
        len(lifecycle_completed) == 1, f"native {action} lifecycle terminal missing"
    )
    require(
        lifecycle_completed[0].get("parent_event_id")
        == lifecycle_scheduled[0].get("event_id"),
        f"native {action} parent_event_id mismatch",
    )
    require(
        lifecycle_completed[0].get("observed_byte_count") == declared_bytes,
        f"native {action} observed bytes differ",
    )
    return declared_bytes


def resolve_native_transfer_request_id(
    rows: Sequence[dict[str, Any]], *, trace_id: str, direction: str
) -> str:
    """Resolve the EngineCore ID from one uniquely traced DMA schedule."""

    require(direction in {"D2H", "H2D"}, "native transfer direction is invalid")
    action = "save" if direction == "D2H" else "prefetch"
    selected = [
        row
        for row in rows
        if row.get("event") == "kv_lifecycle"
        and row.get("trace_id") == trace_id
        and row.get("action") == action
        and row.get("status") == "scheduled"
    ]
    require(
        len(selected) == 1,
        f"expected one scheduled {direction} lifecycle row for {trace_id}",
    )
    request_id = selected[0].get("request_id")
    require(
        isinstance(request_id, str) and request_id,
        f"scheduled {direction} lifecycle row lacks request_id",
    )
    return request_id


def validate_native_trace_closed_set(
    rows: Sequence[dict[str, Any]],
    *,
    run_id: str,
    expected_transfers: Sequence[tuple[str, str]],
) -> None:
    """Require the native trace to contain only the four declared DMA flows."""

    expected = set(expected_transfers)
    require(expected, "native closed set must contain at least one transfer")
    require(
        len(expected) == len(expected_transfers),
        "native expected transfer traces must be unique",
    )
    scheduler_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    lifecycle_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        trace_id = row.get("trace_id")
        require(
            isinstance(trace_id, str) and trace_id.startswith(f"{run_id}:"),
            "native row has an unrelated trace ID",
        )
        if "run_id" in row:
            require(row.get("run_id") == run_id, "native row has an unrelated run ID")
        require(row.get("status") != "failed", "native trace contains a failed event")
        event = row.get("event")
        require(isinstance(event, str) and event, "native row lacks an event")

        scheduler_direction = {
            "store_scheduled": "D2H",
            "store_complete": "D2H",
            "load_scheduled": "H2D",
            "load_complete": "H2D",
        }.get(event)
        if scheduler_direction is not None:
            scheduler_rows.setdefault((trace_id, scheduler_direction), []).append(row)

        if event == "kv_lifecycle" and row.get("action") in {"save", "prefetch"}:
            direction = "D2H" if row["action"] == "save" else "H2D"
            lifecycle_rows.setdefault((trace_id, direction), []).append(row)

    require(
        set(scheduler_rows) == expected,
        "native scheduler trace contains unexpected transfers",
    )
    require(
        set(lifecycle_rows) == expected,
        "native lifecycle trace contains unexpected transfers",
    )
    for transfer in expected:
        scheduled = scheduler_rows[transfer]
        expected_events = (
            {"store_scheduled", "store_complete"}
            if transfer[1] == "D2H"
            else {"load_scheduled", "load_complete"}
        )
        require(
            len(scheduled) == 2
            and {row["event"] for row in scheduled} == expected_events,
            f"native scheduler transfer {transfer} lacks a complete event pair",
        )
        lifecycle = lifecycle_rows[transfer]
        require(
            len(lifecycle) == 2
            and {row.get("status") for row in lifecycle} == {"scheduled", "completed"},
            f"native lifecycle transfer {transfer} lacks a complete event pair",
        )


def validate_lookup_trace(
    rows: Sequence[dict[str, Any]], *, trace_id: str, expected_gpu_hit_tokens: int
) -> None:
    lookups = [
        row
        for row in rows
        if row.get("trace_id") == trace_id and row.get("event") == "lookup"
    ]
    require(lookups, f"native lookup row missing for trace_id={trace_id}")
    row = lookups[-1]
    require(
        row.get("gpu_hit_tokens") == expected_gpu_hit_tokens,
        f"native GPU hit mismatch for {trace_id}: {row}",
    )
    require(
        row.get("native_load_hit_tokens") == 0,
        f"measurement unexpectedly performed an on-demand CPU load: {row}",
    )
    require(
        row.get("skip_reading_prefix_cache") is False,
        f"prefix-cache lookup was bypassed: {row}",
    )


def validate_abba_transfer_chain(
    b1_d2h: dict[str, Any],
    b1_h2d: dict[str, Any],
    b2_d2h: dict[str, Any],
    b2_h2d: dict[str, Any],
) -> None:
    """Validate digest continuity, re-publication, and B-run independence."""

    for label, d2h, h2d in (
        ("B1", b1_d2h, b1_h2d),
        ("B2", b2_d2h, b2_h2d),
    ):
        require(
            d2h["target"]["digest"] == h2d["source"]["digest"],
            f"{label} CPU-retained digest changed before H2D",
        )
        require(
            d2h["payload_bytes"] == h2d["payload_bytes"],
            f"{label} D2H/H2D payload bytes differ",
        )
        require(
            (
                d2h["target"]["physical_slot"],
                d2h["target"]["allocation_generation"],
            )
            == (
                h2d["source"]["physical_slot"],
                h2d["source"]["allocation_generation"],
            ),
            f"{label} H2D did not consume the retained CPU allocation",
        )
        d2h_gpu = d2h["source"]
        h2d_gpu = h2d["target"]
        require(
            (d2h_gpu["physical_slot"], d2h_gpu["allocation_generation"])
            != (h2d_gpu["physical_slot"], h2d_gpu["allocation_generation"]),
            f"{label} H2D reused a stale GPU allocation identity",
        )
        if d2h_gpu["physical_slot"] == h2d_gpu["physical_slot"]:
            require(
                h2d_gpu["allocation_generation"] > d2h_gpu["allocation_generation"],
                f"{label} GPU allocation generation did not advance",
            )

    require(
        b1_d2h["source"]["digest"] == b2_d2h["source"]["digest"],
        "independent B runs produced different canonical KV payloads",
    )
    require(
        (
            b1_d2h["target"]["physical_slot"],
            b1_d2h["target"]["allocation_generation"],
        )
        != (
            b2_d2h["target"]["physical_slot"],
            b2_d2h["target"]["allocation_generation"],
        ),
        "B2 reused B1 CPU allocation identity after connector reset",
    )


def build_llm_kwargs(model: Path, kv_transfer_config: Any) -> dict[str, Any]:
    """Return the frozen single-GPU Qwen3 engine configuration."""

    kwargs = {
        "model": str(model),
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "block_size": BLOCK_SIZE,
        "max_model_len": 64,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 64,
        "gpu_memory_utilization": 0.82,
        "disable_hybrid_kv_cache_manager": True,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "scheduling_policy": "fcfs",
        "seed": FROZEN_SEED,
        "dtype": "bfloat16",
        "attention_config": {
            "backend": "FLASH_ATTN",
            "flash_attn_version": 2,
        },
        "trust_remote_code": False,
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
        "kv_transfer_config": kv_transfer_config,
    }
    require(
        "kv_offloading_size" not in kwargs,
        "kv_offloading_size would overwrite the external connector configuration",
    )
    return kwargs


def _transfer_params(
    *,
    run_id: str,
    phase: str,
    trace_id: str,
    native_trace: Path,
    role: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "offload_trace_file": str(native_trace),
        "offload_trace_id": trace_id,
        "offload_run_id": run_id,
        "offload_phase": phase,
        "offload_workflow_id": f"m2-abba:{run_id}",
        "offload_lifecycle_accounting": True,
        "max_offload_tokens": 0,
    }
    if role == "producer":
        params["max_offload_tokens"] = EXPECTED_EXTERNAL_TOKENS
        params["evict_after_store_complete"] = True
    elif role == "prefetch":
        params["kv_prefetch_request"] = True
        params["dag_preload_event"] = "m2_explicit_prefetch"
        params["dag_preload_sources"] = ["m2_frozen_abba"]
    elif role != "measurement":
        raise ValueError(f"unsupported request role: {role}")
    return params


def _sampling_params(SamplingParams: Any, transfer_params: dict[str, Any]) -> Any:
    return SamplingParams(
        max_tokens=1,
        temperature=0.0,
        seed=FROZEN_SEED,
        ignore_eos=True,
        logprobs=-1,
        prompt_logprobs=None,
        flat_logprobs=True,
        detokenize=False,
        extra_args={"kv_transfer_params": transfer_params},
    )


def _producer_params(SamplingParams: Any, transfer_params: dict[str, Any]) -> Any:
    return SamplingParams(
        max_tokens=1,
        temperature=0.0,
        seed=FROZEN_SEED,
        ignore_eos=True,
        detokenize=False,
        extra_args={"kv_transfer_params": transfer_params},
    )


def _reset(llm: Any, *, reset_connector: bool) -> None:
    for _ in range(100):
        if llm.reset_prefix_cache(
            reset_running_requests=False,
            reset_connector=reset_connector,
        ):
            return
        time.sleep(0.05)
    scope = "GPU and connector" if reset_connector else "GPU"
    raise M2ValidationError(f"timed out resetting {scope} cache")


def _measure(
    *,
    llm: Any,
    TokensPrompt: Any,
    SamplingParams: Any,
    prompt_ids: tuple[int, ...],
    vocab_size: int,
    run_id: str,
    phase: str,
    native_trace: Path,
    expected_cached_tokens: int,
    lifecycle_phase: str | None = None,
) -> Measurement:
    trace_id = f"{run_id}:{phase}:measurement"
    params = _sampling_params(
        SamplingParams,
        _transfer_params(
            run_id=run_id,
            phase=lifecycle_phase or phase,
            trace_id=trace_id,
            native_trace=native_trace,
            role="measurement",
        ),
    )
    started = time.perf_counter()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt_ids))],
        params,
        use_tqdm=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    require(len(outputs) == 1, f"{phase} returned {len(outputs)} request outputs")
    request = outputs[0]
    require(request.finished is True, f"{phase} request did not finish")
    require(
        request.num_cached_tokens == expected_cached_tokens,
        f"{phase} expected {expected_cached_tokens} cached tokens, "
        f"observed {request.num_cached_tokens}",
    )
    require(len(request.outputs) == 1, f"{phase} must return one completion")
    completion = request.outputs[0]
    token_ids = tuple(completion.token_ids)
    require(len(token_ids) == 1, f"{phase} must decode exactly one token")
    logits = dense_logits_from_logprobs(completion.logprobs, vocab_size)

    import numpy as np

    max_value = float(logits.max())
    require(
        float(logits[token_ids[0]]) == max_value,
        f"{phase} greedy token does not have the maximum raw logit",
    )
    require(bool(np.isfinite(logits).all()), f"{phase} raw logits are not finite")
    second_value = float(np.partition(logits, -2)[-2])
    top1_margin = max_value - second_value
    require(top1_margin > 0.0, f"{phase} greedy top-1 margin must be positive")
    return Measurement(
        phase=phase,
        request_id=request.request_id,
        trace_id=trace_id,
        token_id=token_ids[0],
        num_cached_tokens=request.num_cached_tokens,
        elapsed_ms=elapsed_ms,
        top1_margin=top1_margin,
        logits=logits,
    )


def _produce_and_replay(
    *,
    llm: Any,
    TokensPrompt: Any,
    SamplingParams: Any,
    prompt_ids: tuple[int, ...],
    vocab_size: int,
    run_id: str,
    phase: str,
    native_trace: Path,
    timeout_s: float,
) -> tuple[Measurement, dict[str, Any], str, str]:
    _reset(llm, reset_connector=True)
    producer_trace_id = f"{run_id}:{phase}:producer"
    producer_params = _producer_params(
        SamplingParams,
        _transfer_params(
            run_id=run_id,
            phase=phase,
            trace_id=producer_trace_id,
            native_trace=native_trace,
            role="producer",
        ),
    )
    producer_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt_ids))],
        producer_params,
        use_tqdm=False,
    )
    require(len(producer_outputs) == 1, f"{phase} producer output count is invalid")
    producer = producer_outputs[0]
    require(producer.finished is True, f"{phase} producer did not finish")
    producer_request_id = str(producer.request_id)
    require(producer_request_id, f"{phase} producer request ID is empty")

    _reset(llm, reset_connector=False)
    prefetch_trace_id = f"{run_id}:{phase}:prefetch"
    prefetch_request_id = f"m2-prefetch-{phase.lower()}-{uuid.uuid4().hex}"
    prefetch_params = _transfer_params(
        run_id=run_id,
        phase=phase,
        trace_id=prefetch_trace_id,
        native_trace=native_trace,
        role="prefetch",
    )
    prefetch_result = llm.prefetch_kv_cache(
        list(prompt_ids),
        kv_transfer_params=prefetch_params,
        request_id=prefetch_request_id,
        timeout_s=timeout_s,
    )
    validate_prefetch_result(prefetch_result)
    measurement = _measure(
        llm=llm,
        TokensPrompt=TokensPrompt,
        SamplingParams=SamplingParams,
        prompt_ids=prompt_ids,
        vocab_size=vocab_size,
        run_id=run_id,
        phase=phase,
        native_trace=native_trace,
        expected_cached_tokens=EXPECTED_EXTERNAL_TOKENS,
    )
    return measurement, prefetch_result, producer_request_id, producer_trace_id


def _wait_for_diagnostic_rows(
    path: Path, *, run_id: str, terminal_count: int, timeout_s: float
) -> list[dict[str, Any]]:
    require(terminal_count > 0, "diagnostic terminal count must be positive")
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    observed_terminal_ids: set[str] = set()
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                rows = read_jsonl_strict(path)
                observed_terminal_ids = {
                    str(row.get("request_id"))
                    for row in rows
                    if row.get("event") == "terminal"
                    and row.get("status") in {"completed", "failed"}
                    and row.get("run_id") == run_id
                }
                if len(observed_terminal_ids) >= terminal_count:
                    return rows
            except M2ValidationError as exc:
                last_error = exc
        time.sleep(0.05)
    detail = (
        f"last_error={last_error}"
        if last_error is not None
        else f"observed_terminal_ids={sorted(observed_terminal_ids)}"
    )
    raise M2ValidationError(
        f"timed out waiting for {terminal_count} diagnostic terminals: {detail}"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _git_capture(root: Path, *, output_dir: Path, label: str) -> dict[str, Any]:
    """Freeze a reconstructable tracked patch and untracked-file archive."""

    def run(*args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise M2ValidationError(
                f"failed to capture {label} Git state: {exc}"
            ) from exc
        return result.stdout

    root = root.resolve()
    head = run("rev-parse", "HEAD").decode("ascii").strip()
    require(len(head) == 40, f"{label} Git HEAD is invalid")
    status = sorted(
        run("status", "--short", "--untracked-files=all").decode("utf-8").splitlines()
    )
    tracked_diff = run("diff", "--binary", "HEAD", "--")
    untracked_raw = run("ls-files", "--others", "--exclude-standard", "-z")
    untracked_names = [os.fsdecode(item) for item in untracked_raw.split(b"\0") if item]

    state_dir = output_dir / "source_state"
    patch_path = state_dir / f"{label}.tracked.patch"
    archive_path = state_dir / f"{label}.untracked.tar"
    _atomic_write_bytes(patch_path, tracked_diff)

    untracked: list[dict[str, Any]] = []
    validated: list[tuple[str, Path]] = []
    for name in untracked_names:
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe {label} untracked path: {name}",
        )
        source = root / relative
        require(
            source.exists() or source.is_symlink(), f"untracked file vanished: {name}"
        )
        if source.is_symlink():
            target = os.readlink(source)
            payload = os.fsencode(target)
            entry = {
                "path": name,
                "type": "symlink",
                "target": target,
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        else:
            require(source.is_file(), f"untracked path is not a file: {name}")
            entry = {
                "path": name,
                "type": "file",
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        untracked.append(entry)
        validated.append((name, source))

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(
        f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tarfile.open(
            temporary_archive, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for name, source in validated:
                archive.add(source, arcname=name, recursive=False)
        with temporary_archive.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_archive, archive_path)
        _fsync_parent(archive_path)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()

    snapshot = {
        "head": head,
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked": untracked,
    }
    return {
        "root": str(root),
        "head": head,
        "dirty": bool(status),
        "status_short": status,
        "tracked_patch": patch_path.relative_to(output_dir).as_posix(),
        "tracked_patch_sha256": sha256_file(patch_path),
        "untracked_archive": archive_path.relative_to(output_dir).as_posix(),
        "untracked_archive_sha256": sha256_file(archive_path),
        "untracked_files": untracked,
        "snapshot_sha256": _canonical_digest(snapshot),
    }


def _implementation_capture() -> dict[str, Any]:
    """Hash the exact code and claim-boundary files that define this protocol."""

    paths: set[Path] = {
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
        PROTOCOL_SOURCE,
        REPO_ROOT / "research" / "REFERENCES.md",
        REPO_ROOT / "research" / "imported" / "RELATED_WORK_MATRIX.md",
        Path(__file__).resolve(),
        REPO_ROOT / "tools" / "aggregate_m2_calibration.py",
        REPO_ROOT / "tools" / "freeze_m2_tolerance.py",
        REPO_ROOT / "tools" / "m2_calibration_evidence.py",
        REPO_ROOT / "tools" / "m2_raw_replay.py",
        REPO_ROOT / "tools" / "run_m2_calibration_campaign.py",
    }
    for root in (REPO_ROOT / "src", INTEGRATION_ROOT / "dagkv_vllm_m2"):
        paths.update(path for path in root.rglob("*.py") if path.is_file())
    entries = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    return {
        "files": entries,
        "manifest_sha256": _canonical_digest(entries),
    }


def _model_capture(model: Path, *, full_hashes: bool) -> dict[str, Any]:
    """Fingerprint model metadata and, for formal evidence, every weight shard."""

    model = model.resolve()
    files = sorted(
        path
        for path in model.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(model).parts
    )
    require(files, "model directory contains no files")
    for path in files:
        require(not path.is_symlink(), f"model file cannot be a symlink: {path}")

    index_path = model / "model.safetensors.index.json"
    require(index_path.is_file(), "model safetensors index is missing")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M2ValidationError(f"invalid model safetensors index: {exc}") from exc
    require(isinstance(index, dict), "model safetensors index must be an object")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "model weight_map is missing")
    referenced_weights: set[str] = set()
    for value in weight_map.values():
        require(isinstance(value, str) and value, "model weight_map path is invalid")
        relative = Path(value)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe model shard path: {value}",
        )
        require(relative.suffix == ".safetensors", f"invalid model shard: {value}")
        referenced_weights.add(relative.as_posix())

    entries: list[dict[str, Any]] = []
    content_entries: list[dict[str, Any]] = []
    observed_weights: set[str] = set()
    for path in files:
        relative = path.relative_to(model).as_posix()
        is_weight = path.suffix == ".safetensors"
        if is_weight:
            observed_weights.add(relative)
        stat = path.stat()
        content = {
            "path": relative,
            "size": stat.st_size,
            "kind": "weight" if is_weight else "metadata",
            "sha256": sha256_file(path) if full_hashes or not is_weight else None,
        }
        content_entries.append(content)
        entries.append({**content, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino})
    require(
        observed_weights == referenced_weights,
        "model safetensors files differ from the index closed set",
    )
    return {
        "root": str(model),
        "full_hashes": full_hashes,
        "files": entries,
        "manifest_sha256": _canonical_digest(content_entries),
    }


def _runtime_binary_capture(vllm_root: Path, *, full_hashes: bool) -> dict[str, Any]:
    """Fingerprint loaded-tree native extensions and the Python executable."""

    vllm_root = vllm_root.resolve()
    binaries = sorted((vllm_root / "vllm").rglob("*.so"))
    require(binaries, "vLLM tree contains no native extension binaries")
    entries: list[dict[str, Any]] = []
    content_entries: list[dict[str, Any]] = []
    for path in binaries:
        stat = path.stat()
        content = {
            "path": path.relative_to(vllm_root).as_posix(),
            "size": stat.st_size,
            "sha256": sha256_file(path) if full_hashes else None,
        }
        content_entries.append(content)
        entries.append({**content, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino})
    executable = Path(sys.executable).resolve()
    executable_stat = executable.stat()
    python_entry = {
        "path": str(executable),
        "size": executable_stat.st_size,
        "sha256": sha256_file(executable) if full_hashes else None,
        "mtime_ns": executable_stat.st_mtime_ns,
        "inode": executable_stat.st_ino,
    }
    python_content = {key: python_entry[key] for key in ("path", "size", "sha256")}
    return {
        "root": str(vllm_root),
        "full_hashes": full_hashes,
        "vllm_extensions": entries,
        "python_executable": python_entry,
        "manifest_sha256": _canonical_digest(
            {
                "vllm_extensions": content_entries,
                "python_executable": python_content,
            }
        ),
    }


def _verify_git_capture(capture: dict[str, Any], *, label: str) -> str:
    """Re-hash a Git worktree after execution and reject any content drift."""

    root = Path(capture["root"])

    def run(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise M2ValidationError(
                f"failed to verify {label} Git state: {exc}"
            ) from exc

    head = run("rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = run("diff", "--binary", "HEAD", "--")
    names = [
        os.fsdecode(item)
        for item in run("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if item
    ]
    untracked: list[dict[str, Any]] = []
    for name in names:
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe {label} untracked path during verification: {name}",
        )
        source = root / relative
        if source.is_symlink():
            target = os.readlink(source)
            payload = os.fsencode(target)
            untracked.append(
                {
                    "path": name,
                    "type": "symlink",
                    "target": target,
                    "size": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
        else:
            require(source.is_file(), f"{label} untracked file vanished: {name}")
            untracked.append(
                {
                    "path": name,
                    "type": "file",
                    "size": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
    snapshot = {
        "head": head,
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked": untracked,
    }
    digest = _canonical_digest(snapshot)
    require(
        digest == capture["snapshot_sha256"], f"{label} Git state changed during run"
    )
    return digest


def _verify_file_stats(capture: dict[str, Any], *, kind: str) -> None:
    """Reject model or native-binary replacement while the engine was running."""

    if kind == "model":
        root = Path(capture["root"])
        entries = [(root / item["path"], item) for item in capture["files"]]
    elif kind == "runtime_binaries":
        root = Path(capture["root"])
        entries = [(root / item["path"], item) for item in capture["vllm_extensions"]]
        python_entry = capture["python_executable"]
        entries.append((Path(python_entry["path"]), python_entry))
    else:
        raise ValueError(f"unsupported stat capture kind: {kind}")

    for path, expected in entries:
        require(path.is_file(), f"{kind} file vanished during run: {path}")
        observed = path.stat()
        require(
            (
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ino,
            )
            == (
                expected["size"],
                expected["mtime_ns"],
                expected["inode"],
            ),
            f"{kind} file changed during run: {path}",
        )


def _dependency_capture() -> dict[str, Any]:
    packages = sorted(
        {
            (
                distribution.metadata.get("Name", "<unnamed>"),
                distribution.version,
            )
            for distribution in importlib_metadata.distributions()
        }
    )
    entries = [{"name": name, "version": version} for name, version in packages]
    return {"packages": entries, "manifest_sha256": _canonical_digest(entries)}


def _system_capture(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,pci.bus_id,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M2ValidationError(
            f"failed to capture NVIDIA runtime identity: {exc}"
        ) from exc
    require(
        smi and len(smi.splitlines()) == 1, "expected exactly one visible NVIDIA GPU"
    )
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        os_release = {}
    relevant_environment = {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key == "CUDA_VISIBLE_DEVICES"
        or key
        in {
            "HF_HUB_OFFLINE",
            "LD_LIBRARY_PATH",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_OFFLINE",
        }
        or key.startswith("VLLM_")
    }
    return {
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "libc": list(platform.libc_ver()),
        "os_release": os_release,
        "gpu": {
            "nvidia_smi": smi,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "build_config": torch.__config__.show(),
        },
        "environment": relevant_environment,
    }


def _write_sha256sums(output_dir: Path) -> Path:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not path.name.endswith(".tmp")
        and path.name
        not in {
            "M2_ITEM8_FORMAL_RUN_MANIFEST.json",
            "M2_ITEM8_ACCEPTANCE_MANIFEST.json",
            "M2_ACCEPTANCE_MANIFEST.json",
        }
    )
    require(paths, "cannot write SHA256SUMS for an empty artifact directory")
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in paths
    ]
    checksum_path = output_dir / "SHA256SUMS"
    _atomic_write_bytes(checksum_path, ("\n".join(lines) + "\n").encode("ascii"))
    return checksum_path


def _prepare_import_path() -> None:
    require(
        INTEGRATION_ROOT.is_dir(), f"vLLM M2 integration is missing: {INTEGRATION_ROOT}"
    )
    integration = str(INTEGRATION_ROOT)
    if integration not in sys.path:
        sys.path.insert(0, integration)
    current = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        integration if not current else f"{integration}:{current}"
    )


def _capability_preflight(
    *,
    LLM: Any,
    KVTransferConfig: Any,
    connector_module: Any,
    spec_module: Any,
    model: Path,
    vllm_root: Path,
) -> dict[str, Any]:
    import torch
    import vllm

    require(model.is_dir(), f"model directory is missing: {model}")
    require((model / "config.json").is_file(), "model config.json is missing")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(
        torch.cuda.device_count() == 1, "M2 runner requires exactly one visible GPU"
    )

    reset_signature = inspect.signature(LLM.reset_prefix_cache)
    require(
        "reset_connector" in reset_signature.parameters, "connector reset API missing"
    )
    prefetch_signature = inspect.signature(LLM.prefetch_kv_cache)
    for parameter in (
        "prompt_token_ids",
        "kv_transfer_params",
        "request_id",
        "timeout_s",
    ):
        require(
            parameter in prefetch_signature.parameters,
            f"prefetch API lacks {parameter}",
        )
    require(hasattr(LLM, "get_kv_cache_snapshot"), "KV snapshot API missing")
    require(
        hasattr(connector_module, "DAGKVDiagnosticOffloadingConnector"),
        "diagnostic connector class missing",
    )
    require(
        hasattr(spec_module, "DAGKVDiagnosticCPUOffloadingSpec"),
        "diagnostic offloading spec class missing",
    )
    for module in (connector_module, spec_module):
        module_path = Path(module.__file__).resolve()
        require(
            module_path.is_relative_to(INTEGRATION_ROOT.resolve()),
            f"external module resolved outside the DAGKV integration: {module_path}",
        )

    module_root = Path(vllm.__file__).resolve()
    require(
        module_root.is_relative_to(vllm_root.resolve()),
        f"vLLM imported from unexpected tree: {module_root}",
    )
    model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    architectures = model_config.get("architectures", [])
    require(
        any("Qwen3" in str(item) for item in architectures),
        f"M2 requires Qwen3, observed architectures={architectures}",
    )
    vocab_size = model_config.get("vocab_size")
    require(
        vocab_size == FROZEN_QWEN3_8B_VOCAB_SIZE,
        "model vocabulary differs from the frozen Qwen3-8B vocabulary",
    )
    return {
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_module": str(module_root),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "model_architectures": architectures,
        "model_config_sha256": sha256_file(model / "config.json"),
    }


def _validate_engine_config(llm: Any) -> int:
    config = llm.llm_engine.vllm_config
    require(config.cache_config.block_size == BLOCK_SIZE, "engine block size drifted")
    require(
        config.cache_config.kv_offloading_size is None,
        "kv_offloading_size overwrote the custom connector configuration",
    )
    transfer = config.kv_transfer_config
    require(transfer is not None, "engine lost KVTransferConfig")
    require(
        transfer.kv_connector == "DAGKVDiagnosticOffloadingConnector",
        "engine loaded an unexpected connector",
    )
    require(config.model_config.max_logprobs == -1, "max_logprobs must remain -1")
    require(
        config.model_config.logprobs_mode == "raw_logits", "raw_logits mode drifted"
    )
    require(
        config.parallel_config.tensor_parallel_size == 1, "tensor parallelism drifted"
    )
    require(
        config.parallel_config.pipeline_parallel_size == 1,
        "pipeline parallelism drifted",
    )
    require(
        config.scheduler_config.enable_chunked_prefill is True,
        "chunked-prefill configuration drifted",
    )
    attention = config.attention_config
    require(
        attention.backend is not None and attention.backend.name == "FLASH_ATTN",
        "attention backend drifted",
    )
    require(attention.flash_attn_version == 2, "FlashAttention version drifted")
    snapshot = llm.get_kv_cache_snapshot()
    for key in ("total_blocks", "used_blocks", "free_blocks", "usage_ratio"):
        require(key in snapshot, f"KV snapshot lacks {key}")
    vocab_size = config.model_config.get_vocab_size()
    require(
        vocab_size == FROZEN_QWEN3_8B_VOCAB_SIZE,
        "engine vocabulary differs from the frozen Qwen3-8B vocabulary",
    )
    return vocab_size


def _run(args: argparse.Namespace, output_dir: Path, run_started_ns: int) -> None:
    import numpy as np

    prompt_ids = validate_prompt_tokens(PROMPT_TOKEN_IDS)
    require(
        args.full_provenance,
        "M2 v2 runs require --full-provenance",
    )
    implementation = _implementation_capture()
    frozen_tolerance: FrozenTolerance | None = None
    calibration_cohort: dict[str, Any] | None = None
    if args.mode == "formal":
        require(
            args.atol == 0.0 and args.rtol == 0.0, "formal mode forbids CLI tolerance"
        )
        require(
            args.tolerance_file is not None, "formal mode requires --tolerance-file"
        )
        require(
            args.calibration_manifest is not None,
            "formal mode requires --calibration-manifest",
        )
        frozen_tolerance = load_frozen_tolerance(
            args.tolerance_file.expanduser().absolute(),
            run_started_ns=run_started_ns,
        )
        calibration_cohort = load_calibration_cohort(
            args.calibration_manifest.expanduser().absolute(),
            frozen_tolerance=frozen_tolerance,
            run_started_ns=run_started_ns,
            expected_implementation_manifest_sha256=(implementation["manifest_sha256"]),
        )
        atol, rtol = frozen_tolerance.atol, frozen_tolerance.rtol
    else:
        require(
            args.tolerance_file is None,
            "calibration mode does not consume tolerance files",
        )
        require(
            args.calibration_manifest is None,
            "calibration mode does not consume calibration manifests",
        )
        atol, rtol = args.atol, args.rtol
        require(atol >= 0.0 and math.isfinite(atol), "atol must be non-negative")
        require(rtol >= 0.0 and math.isfinite(rtol), "rtol must be non-negative")

    _prepare_import_path()
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    connector_module = importlib.import_module("dagkv_vllm_m2.connector")
    spec_module = importlib.import_module("dagkv_vllm_m2.spec")
    preflight = _capability_preflight(
        LLM=LLM,
        KVTransferConfig=KVTransferConfig,
        connector_module=connector_module,
        spec_module=spec_module,
        model=args.model,
        vllm_root=args.vllm_root,
    )

    native_trace = output_dir / "native_lifecycle.jsonl"
    diagnostic_trace = output_dir / "diagnostic_transfers.jsonl"
    run_id = f"m2-{uuid.uuid4().hex}"
    connector_extra = {
        "cpu_bytes_to_use": args.cpu_bytes,
        "spec_name": "DAGKVDiagnosticCPUOffloadingSpec",
        "spec_module_path": "dagkv_vllm_m2.spec",
        "dagkv_diagnostic_trace_file": str(diagnostic_trace),
        "dagkv_diagnostic_run_id": run_id,
        "dagkv_diagnostic_phase": "ABBA",
        "fanout_layerwise_load": False,
        "lifecycle_accounting_enabled": True,
    }
    transfer_config = KVTransferConfig(
        kv_connector="DAGKVDiagnosticOffloadingConnector",
        kv_connector_module_path="dagkv_vllm_m2.connector",
        kv_role="kv_both",
        kv_buffer_device="cpu",
        kv_load_failure_policy="fail",
        kv_connector_extra_config=connector_extra,
    )
    llm_kwargs = build_llm_kwargs(args.model, transfer_config)

    dagkv_git = _git_capture(REPO_ROOT, output_dir=output_dir, label="dagkv")
    vllm_git = _git_capture(args.vllm_root, output_dir=output_dir, label="vllm")
    if args.mode == "formal":
        require(
            not dagkv_git["dirty"],
            "formal mode requires a clean DAGKV Git worktree",
        )
    model_capture = _model_capture(
        args.model,
        full_hashes=args.full_provenance,
    )
    runtime_binaries = _runtime_binary_capture(
        args.vllm_root,
        full_hashes=args.full_provenance,
    )
    dependencies = _dependency_capture()
    system = _system_capture(torch)
    static_connector_config = {
        key: value
        for key, value in connector_extra.items()
        if key
        not in {
            "dagkv_diagnostic_trace_file",
            "dagkv_diagnostic_run_id",
        }
    }
    reproducibility_components = {
        "implementation_manifest_sha256": implementation["manifest_sha256"],
        "vllm_snapshot_sha256": vllm_git["snapshot_sha256"],
        "model_manifest_sha256": model_capture["manifest_sha256"],
        "runtime_binary_manifest_sha256": runtime_binaries["manifest_sha256"],
        "dependency_manifest_sha256": dependencies["manifest_sha256"],
        "system": system,
        "prompt_token_ids": list(prompt_ids),
        "block_size": BLOCK_SIZE,
        "cpu_bytes": args.cpu_bytes,
        "engine_config": {
            key: value
            for key, value in llm_kwargs.items()
            if key not in {"kv_transfer_config"}
        },
        "connector_config": static_connector_config,
    }
    reproducibility_fingerprint = _canonical_digest(reproducibility_components)
    if frozen_tolerance is not None:
        require(
            reproducibility_fingerprint == frozen_tolerance.reproducibility_fingerprint,
            "current runtime differs from the frozen calibration fingerprint",
        )

    provenance = {
        "schema_version": PROTOCOL_SCHEMA,
        "run_id": run_id,
        "mode": args.mode,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "prompt_token_ids": list(prompt_ids),
        "block_size": BLOCK_SIZE,
        "cpu_bytes": args.cpu_bytes,
        "tolerance": {"atol": atol, "rtol": rtol},
        "frozen_tolerance": asdict(frozen_tolerance) if frozen_tolerance else None,
        "calibration_cohort": (
            {
                "path": str(args.calibration_manifest.resolve()),
                "sha256": frozen_tolerance.calibration_manifest_sha256,
                "run_count": calibration_cohort["run_count"],
            }
            if calibration_cohort is not None and frozen_tolerance is not None
            else None
        ),
        "full_provenance": args.full_provenance,
        "preflight": preflight,
        "implementation": implementation,
        "dagkv_git": dagkv_git,
        "vllm_git": vllm_git,
        "model": model_capture,
        "runtime_binaries": runtime_binaries,
        "dependencies": dependencies,
        "system": system,
        "reproducibility_components": reproducibility_components,
        "reproducibility_fingerprint": reproducibility_fingerprint,
        "engine_config": {
            key: value
            for key, value in llm_kwargs.items()
            if key not in {"kv_transfer_config"}
        },
        "connector_config": connector_extra,
    }
    _write_json(output_dir / "provenance.json", provenance)

    llm = LLM(**llm_kwargs)
    try:
        vocab_size = _validate_engine_config(llm)
        require(
            max(prompt_ids) < vocab_size, "prompt token ID exceeds runtime vocabulary"
        )

        _reset(llm, reset_connector=True)
        a1 = _measure(
            llm=llm,
            TokensPrompt=TokensPrompt,
            SamplingParams=SamplingParams,
            prompt_ids=prompt_ids,
            vocab_size=vocab_size,
            run_id=run_id,
            phase="A1",
            native_trace=native_trace,
            expected_cached_tokens=0,
            lifecycle_phase="A1_G",
        )
        g = _measure(
            llm=llm,
            TokensPrompt=TokensPrompt,
            SamplingParams=SamplingParams,
            prompt_ids=prompt_ids,
            vocab_size=vocab_size,
            run_id=run_id,
            phase="G",
            native_trace=native_trace,
            expected_cached_tokens=EXPECTED_EXTERNAL_TOKENS,
            lifecycle_phase="A1_G",
        )
        b1, b1_prefetch, b1_producer_id, b1_producer_trace = _produce_and_replay(
            llm=llm,
            TokensPrompt=TokensPrompt,
            SamplingParams=SamplingParams,
            prompt_ids=prompt_ids,
            vocab_size=vocab_size,
            run_id=run_id,
            phase="B1",
            native_trace=native_trace,
            timeout_s=args.timeout_s,
        )
        b2, b2_prefetch, b2_producer_id, b2_producer_trace = _produce_and_replay(
            llm=llm,
            TokensPrompt=TokensPrompt,
            SamplingParams=SamplingParams,
            prompt_ids=prompt_ids,
            vocab_size=vocab_size,
            run_id=run_id,
            phase="B2",
            native_trace=native_trace,
            timeout_s=args.timeout_s,
        )
        _reset(llm, reset_connector=True)
        a2 = _measure(
            llm=llm,
            TokensPrompt=TokensPrompt,
            SamplingParams=SamplingParams,
            prompt_ids=prompt_ids,
            vocab_size=vocab_size,
            run_id=run_id,
            phase="A2",
            native_trace=native_trace,
            expected_cached_tokens=0,
        )
        _write_json(
            output_dir / "execution_ids.json",
            {
                "schema_version": PROTOCOL_SCHEMA,
                "run_id": run_id,
                "measurements": {
                    item.phase: {
                        "request_id": item.request_id,
                        "trace_id": item.trace_id,
                    }
                    for item in (a1, g, b1, b2, a2)
                },
                "transfers": {
                    "B1_D2H": {
                        "request_id": b1_producer_id,
                        "trace_id": b1_producer_trace,
                    },
                    "B1_H2D": {
                        "request_id": b1_prefetch["request_id"],
                        "trace_id": f"{run_id}:B1:prefetch",
                    },
                    "B2_D2H": {
                        "request_id": b2_producer_id,
                        "trace_id": b2_producer_trace,
                    },
                    "B2_H2D": {
                        "request_id": b2_prefetch["request_id"],
                        "trace_id": f"{run_id}:B2:prefetch",
                    },
                },
            },
        )
    finally:
        del llm

    measurements = {item.phase: item for item in (a1, g, b1, b2, a2)}
    for phase, measurement in measurements.items():
        np.save(
            output_dir / f"logits_{phase}.npy", measurement.logits, allow_pickle=False
        )

    diagnostic_rows = _wait_for_diagnostic_rows(
        diagnostic_trace,
        run_id=run_id,
        terminal_count=4,
        timeout_s=args.timeout_s,
    )
    native_rows = read_jsonl_strict(native_trace)
    b1_producer_engine_id = resolve_native_transfer_request_id(
        native_rows, trace_id=b1_producer_trace, direction="D2H"
    )
    b2_producer_engine_id = resolve_native_transfer_request_id(
        native_rows, trace_id=b2_producer_trace, direction="D2H"
    )
    require(
        b1_producer_engine_id.startswith(f"{b1_producer_id}-"),
        "B1 API and EngineCore producer request IDs are unrelated",
    )
    require(
        b2_producer_engine_id.startswith(f"{b2_producer_id}-"),
        "B2 API and EngineCore producer request IDs are unrelated",
    )
    require(
        resolve_native_transfer_request_id(
            native_rows,
            trace_id=f"{run_id}:B1:prefetch",
            direction="H2D",
        )
        == b1_prefetch["request_id"],
        "B1 prefetch request ID differs across API and EngineCore",
    )
    require(
        resolve_native_transfer_request_id(
            native_rows,
            trace_id=f"{run_id}:B2:prefetch",
            direction="H2D",
        )
        == b2_prefetch["request_id"],
        "B2 prefetch request ID differs across API and EngineCore",
    )

    execution_ids_path = output_dir / "execution_ids.json"
    execution_ids = json.loads(execution_ids_path.read_text(encoding="utf-8"))
    execution_ids["transfers"]["B1_D2H"]["engine_request_id"] = b1_producer_engine_id
    execution_ids["transfers"]["B2_D2H"]["engine_request_id"] = b2_producer_engine_id
    execution_ids["transfers"]["B1_H2D"]["engine_request_id"] = b1_prefetch[
        "request_id"
    ]
    execution_ids["transfers"]["B2_H2D"]["engine_request_id"] = b2_prefetch[
        "request_id"
    ]
    _write_json(execution_ids_path, execution_ids)
    validate_native_trace_closed_set(
        native_rows,
        run_id=run_id,
        expected_transfers=(
            (b1_producer_trace, "D2H"),
            (f"{run_id}:B1:prefetch", "H2D"),
            (b2_producer_trace, "D2H"),
            (f"{run_id}:B2:prefetch", "H2D"),
        ),
    )
    validate_diagnostic_trace_closed_set(
        diagnostic_rows,
        run_id=run_id,
        expected_transfers=(
            (b1_producer_engine_id, "D2H"),
            (b1_prefetch["request_id"], "H2D"),
            (b2_producer_engine_id, "D2H"),
            (b2_prefetch["request_id"], "H2D"),
        ),
    )

    b1_d2h = validate_diagnostic_transfer(
        diagnostic_rows,
        request_id=b1_producer_engine_id,
        direction="D2H",
        run_id=run_id,
    )
    b1_h2d = validate_diagnostic_transfer(
        diagnostic_rows,
        request_id=b1_prefetch["request_id"],
        direction="H2D",
        run_id=run_id,
    )
    b2_d2h = validate_diagnostic_transfer(
        diagnostic_rows,
        request_id=b2_producer_engine_id,
        direction="D2H",
        run_id=run_id,
    )
    b2_h2d = validate_diagnostic_transfer(
        diagnostic_rows,
        request_id=b2_prefetch["request_id"],
        direction="H2D",
        run_id=run_id,
    )
    validate_abba_transfer_chain(b1_d2h, b1_h2d, b2_d2h, b2_h2d)

    native_bytes = {
        "B1_D2H": validate_native_transfer(
            native_rows, trace_id=b1_producer_trace, direction="D2H"
        ),
        "B1_H2D": validate_native_transfer(
            native_rows,
            trace_id=f"{run_id}:B1:prefetch",
            direction="H2D",
        ),
        "B2_D2H": validate_native_transfer(
            native_rows, trace_id=b2_producer_trace, direction="D2H"
        ),
        "B2_H2D": validate_native_transfer(
            native_rows,
            trace_id=f"{run_id}:B2:prefetch",
            direction="H2D",
        ),
    }
    diagnostic_bytes = {
        "B1_D2H": b1_d2h["payload_bytes"],
        "B1_H2D": b1_h2d["payload_bytes"],
        "B2_D2H": b2_d2h["payload_bytes"],
        "B2_H2D": b2_h2d["payload_bytes"],
    }
    require(
        native_bytes == diagnostic_bytes, "native and diagnostic byte counts differ"
    )

    validate_lookup_trace(native_rows, trace_id=a1.trace_id, expected_gpu_hit_tokens=0)
    validate_lookup_trace(
        native_rows,
        trace_id=g.trace_id,
        expected_gpu_hit_tokens=EXPECTED_EXTERNAL_TOKENS,
    )
    validate_lookup_trace(
        native_rows,
        trace_id=b1.trace_id,
        expected_gpu_hit_tokens=EXPECTED_EXTERNAL_TOKENS,
    )
    validate_lookup_trace(
        native_rows,
        trace_id=b2.trace_id,
        expected_gpu_hit_tokens=EXPECTED_EXTERNAL_TOKENS,
    )
    validate_lookup_trace(native_rows, trace_id=a2.trace_id, expected_gpu_hit_tokens=0)

    tolerant_pairs = (("A1", "G"), ("A1", "B1"), ("A1", "B2"))
    exact_pairs = (("A1", "A2"), ("G", "B1"), ("G", "B2"), ("B1", "B2"))
    pairs = (*tolerant_pairs, *exact_pairs)
    comparisons = [
        compare_logit_vectors(
            left_name,
            measurements[left_name].token_id,
            measurements[left_name].logits,
            right_name,
            measurements[right_name].token_id,
            measurements[right_name].logits,
            atol=atol,
            rtol=rtol,
        )
        for left_name, right_name in pairs
    ]
    for comparison in comparisons:
        require(
            comparison.token_equal,
            f"token mismatch: {comparison.left}/{comparison.right}",
        )
        if (comparison.left, comparison.right) in exact_pairs:
            require(
                comparison.max_abs_error == 0.0,
                f"path-internal logits drifted: {comparison.left}/{comparison.right}",
            )
    minimum_top1_margin = min(item.top1_margin for item in measurements.values())
    require(
        minimum_top1_margin > 2 * MAX_FORMAL_ATOL,
        "greedy top-1 margin is too small for the preregistered tolerance cap",
    )
    within_requested_tolerance = all(item.allclose for item in comparisons)
    if args.mode == "formal":
        require(
            within_requested_tolerance,
            "formal replay logits exceed the frozen tolerance",
        )

    post_implementation = _implementation_capture()
    require(
        post_implementation["manifest_sha256"] == implementation["manifest_sha256"],
        "DAGKV implementation changed during run",
    )
    _verify_file_stats(model_capture, kind="model")
    _verify_file_stats(runtime_binaries, kind="runtime_binaries")
    provenance["postflight"] = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "dagkv_git_snapshot_sha256": _verify_git_capture(dagkv_git, label="dagkv"),
        "vllm_git_snapshot_sha256": _verify_git_capture(vllm_git, label="vllm"),
        "implementation_manifest_sha256": post_implementation["manifest_sha256"],
        "model_file_stats_unchanged": True,
        "runtime_binary_stats_unchanged": True,
    }
    _write_json(output_dir / "provenance.json", provenance)

    result = {
        "schema_version": PROTOCOL_SCHEMA,
        "run_id": run_id,
        "mode": args.mode,
        "gate_status": (
            "M2_ITEM8_FORMAL_HOLDOUT_PASSED"
            if args.mode == "formal"
            else "CALIBRATED_NOT_ACCEPTED"
        ),
        "m2_accepted": False,
        "m2_item8_accepted": False,
        "formal_run_passed": args.mode == "formal",
        "within_requested_tolerance": within_requested_tolerance,
        "minimum_top1_margin": minimum_top1_margin,
        "reproducibility_fingerprint": reproducibility_fingerprint,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "tolerance": {"atol": atol, "rtol": rtol},
        "measurements": {
            phase: {
                "request_id": measurement.request_id,
                "trace_id": measurement.trace_id,
                "token_id": measurement.token_id,
                "num_cached_tokens": measurement.num_cached_tokens,
                "elapsed_ms": measurement.elapsed_ms,
                "top1_margin": measurement.top1_margin,
                "logits_file": f"logits_{phase}.npy",
                "logits_sha256": sha256_file(output_dir / f"logits_{phase}.npy"),
            }
            for phase, measurement in measurements.items()
        },
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "prefetch": {"B1": b1_prefetch, "B2": b2_prefetch},
        "native_bytes": native_bytes,
        "diagnostic_bytes": diagnostic_bytes,
        "transfer_digests": {
            "B1": b1_d2h["source"]["digest"],
            "B2": b2_d2h["source"]["digest"],
        },
        "artifacts": {
            "native_trace": native_trace.name,
            "diagnostic_trace": diagnostic_trace.name,
            "protocol": "protocol.md",
            "provenance": "provenance.json",
        },
    }
    _write_json(output_dir / "result.json", result)

    checksum_path = _write_sha256sums(output_dir)
    if args.mode == "formal":
        assert frozen_tolerance is not None
        formal_run = {
            "schema_version": ITEM8_FORMAL_RUN_SCHEMA,
            "run_id": run_id,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "result_sha256": sha256_file(output_dir / "result.json"),
            "provenance_sha256": sha256_file(output_dir / "provenance.json"),
            "sha256sums_sha256": sha256_file(checksum_path),
            "frozen_tolerance_sha256": frozen_tolerance.file_sha256,
            "calibration_manifest_sha256": (
                frozen_tolerance.calibration_manifest_sha256
            ),
            "reproducibility_fingerprint": reproducibility_fingerprint,
            "statement": (
                "One M2 item 8 formal holdout run passed; cohort aggregation "
                "and the aggregate M2 gate remain open."
            ),
        }
        _write_json(output_dir / "M2_ITEM8_FORMAL_RUN_MANIFEST.json", formal_run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("calibration", "formal"), default="calibration"
    )
    parser.add_argument("--tolerance-file", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--cpu-bytes", type=int, default=1 << 30)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--full-provenance",
        action="store_true",
        help=(
            "hash all model weights and vLLM native binaries "
            "(required for every M2 v2 run)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_started_ns = time.time_ns()
    args = _parser().parse_args(argv)
    if args.cpu_bytes <= 0:
        raise SystemExit("--cpu-bytes must be positive")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.cuda_device < 0:
        raise SystemExit("--cuda-device must be non-negative")

    output_dir = args.output_dir.resolve()
    for protected_root in (
        REPO_ROOT.resolve(),
        args.vllm_root.resolve(),
        args.model.resolve(),
    ):
        require(
            output_dir != protected_root
            and not output_dir.is_relative_to(protected_root),
            f"output directory must be outside protected input root: {protected_root}",
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL_SOURCE, output_dir / "protocol.md")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        _run(args, output_dir, run_started_ns)
    except Exception as exc:
        for name in (
            "M2_ITEM8_FORMAL_RUN_MANIFEST.json",
            "M2_ITEM8_ACCEPTANCE_MANIFEST.json",
            "M2_ACCEPTANCE_MANIFEST.json",
            "SHA256SUMS",
        ):
            artifact = output_dir / name
            if artifact.exists():
                artifact.unlink()
        failure = {
            "schema_version": PROTOCOL_SCHEMA,
            "mode": args.mode,
            "gate_status": "FAILED",
            "m2_accepted": False,
            "m2_item8_accepted": False,
            "formal_run_passed": False,
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "result.json", failure)
        print(f"M2 ABBA failed: {exc}", file=sys.stderr)
        return 1

    status = "item 8 formal holdout" if args.mode == "formal" else "calibration"
    print(f"M2 {status} completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
