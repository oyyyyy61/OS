#!/usr/bin/env python3
"""Close M2 item 8 from exactly twenty frozen formal holdout attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.aggregate_m2_calibration import (
        _validate_run as _validate_calibration_run,
    )
    from tools.freeze_m2_tolerance import TOLERANCE_FIELDS, TOLERANCE_SCHEMA
    from tools.m2_calibration_evidence import (
        CalibrationEvidenceError,
        read_stable_bytes,
        validate_published_calibration_bundle,
    )
    from tools.m2_raw_replay import M2RawReplayError, validate_raw_run
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from aggregate_m2_calibration import (  # type: ignore[no-redef]
        _validate_run as _validate_calibration_run,
    )
    from freeze_m2_tolerance import (  # type: ignore[no-redef]
        TOLERANCE_FIELDS,
        TOLERANCE_SCHEMA,
    )
    from m2_calibration_evidence import (  # type: ignore[no-redef]
        CalibrationEvidenceError,
        read_stable_bytes,
        validate_published_calibration_bundle,
    )
    from m2_raw_replay import (  # type: ignore[no-redef]
        M2RawReplayError,
        validate_raw_run,
    )

PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v3"
FORMAL_RUN_SCHEMA = "dagkv.m2.item8.formal_run.v1"
ITEM8_ACCEPTANCE_SCHEMA = "dagkv.m2.item8.acceptance.v2"
FORMAL_RUN_COUNT = 20
CALIBRATION_RUN_COUNT = 59
MAX_FORMAL_ATOL = 0.125
FORMAL_RTOL = 0.0
TOLERANCE_DERIVATION = "fixed_binary_cap_from_excluded_pilot"

FORMAL_RUN_MANIFEST = "M2_ITEM8_FORMAL_RUN_MANIFEST.json"
ACCEPTANCE_MANIFEST = "M2_ITEM8_ACCEPTANCE_MANIFEST.json"
FORMAL_RUN_STATEMENT = (
    "One M2 item 8 formal holdout run passed; cohort aggregation "
    "and the aggregate M2 gate remain open."
)
ACCEPTANCE_STATEMENT = (
    "Exactly twenty frozen M2 item 8 holdouts passed. This closes item 8 only; "
    "the aggregate M2 gate remains open and this evidence supports no latency, "
    "throughput, hit-rate, scheduling-policy, or paper-performance claim."
)

EXPECTED_MEASUREMENTS = ("A1", "G", "B1", "B2", "A2")
TOLERANT_PAIRS = (("A1", "G"), ("A1", "B1"), ("A1", "B2"))
EXACT_PAIRS = (("A1", "A2"), ("G", "B1"), ("G", "B2"), ("B1", "B2"))
EXPECTED_PAIRS = frozenset((*TOLERANT_PAIRS, *EXACT_PAIRS))
TRANSFER_NAMES = frozenset({"B1_D2H", "B1_H2D", "B2_D2H", "B2_H2D"})

REQUIRED_INTERNAL_ARTIFACTS = frozenset(
    {
        "diagnostic_transfers.jsonl",
        "execution_ids.json",
        "source_state/dagkv.tracked.patch",
        "source_state/dagkv.untracked.tar",
        "source_state/vllm.tracked.patch",
        "source_state/vllm.untracked.tar",
        "logits_A1.npy",
        "logits_A2.npy",
        "logits_B1.npy",
        "logits_B2.npy",
        "logits_G.npy",
        "native_lifecycle.jsonl",
        "protocol.md",
        "provenance.json",
        "result.json",
    }
)
FORBIDDEN_RUN_ARTIFACTS = frozenset(
    {ACCEPTANCE_MANIFEST, "M2_ACCEPTANCE_MANIFEST.json"}
)

RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "gate_status",
        "m2_accepted",
        "m2_item8_accepted",
        "formal_run_passed",
        "within_requested_tolerance",
        "minimum_top1_margin",
        "reproducibility_fingerprint",
        "completed_at_utc",
        "tolerance",
        "measurements",
        "comparisons",
        "prefetch",
        "native_bytes",
        "diagnostic_bytes",
        "transfer_digests",
        "artifacts",
    }
)
MEASUREMENT_FIELDS = frozenset(
    {
        "request_id",
        "trace_id",
        "token_id",
        "num_cached_tokens",
        "elapsed_ms",
        "top1_margin",
        "logits_file",
        "logits_sha256",
    }
)
COMPARISON_FIELDS = frozenset(
    {
        "left",
        "right",
        "token_equal",
        "allclose",
        "max_abs_error",
        "max_rel_error",
    }
)
PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "started_at_utc",
        "argv",
        "python",
        "executable",
        "prompt_token_ids",
        "block_size",
        "cpu_bytes",
        "tolerance",
        "frozen_tolerance",
        "calibration_cohort",
        "full_provenance",
        "preflight",
        "implementation",
        "dagkv_git",
        "vllm_git",
        "model",
        "runtime_binaries",
        "dependencies",
        "system",
        "nvidia_driver_userspace",
        "reproducibility_components",
        "reproducibility_fingerprint",
        "engine_config",
        "connector_config",
        "postflight",
    }
)
REPRODUCIBILITY_COMPONENT_FIELDS = frozenset(
    {
        "implementation_manifest_sha256",
        "vllm_snapshot_sha256",
        "model_manifest_sha256",
        "runtime_binary_manifest_sha256",
        "dependency_manifest_sha256",
        "nvidia_driver_userspace_content_digest",
        "system",
        "prompt_token_ids",
        "block_size",
        "cpu_bytes",
        "engine_config",
        "connector_config",
    }
)
ENGINE_CONFIG_FIELDS = frozenset(
    {
        "model",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "enforce_eager",
        "enable_prefix_caching",
        "block_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "disable_hybrid_kv_cache_manager",
        "enable_chunked_prefill",
        "async_scheduling",
        "scheduling_policy",
        "seed",
        "dtype",
        "attention_config",
        "trust_remote_code",
        "max_logprobs",
        "logprobs_mode",
    }
)
CONNECTOR_CONFIG_FIELDS = frozenset(
    {
        "cpu_bytes_to_use",
        "spec_name",
        "spec_module_path",
        "dagkv_diagnostic_trace_file",
        "dagkv_diagnostic_run_id",
        "dagkv_diagnostic_phase",
        "fanout_layerwise_load",
        "lifecycle_accounting_enabled",
    }
)
FROZEN_TOLERANCE_FIELDS = frozenset(
    {
        "atol",
        "rtol",
        "frozen_at_utc",
        "calibration_manifest_sha256",
        "reproducibility_fingerprint",
        "calibration_run_count",
        "derivation",
        "file_sha256",
    }
)
FORMAL_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "completed_at_utc",
        "result_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
        "frozen_tolerance_sha256",
        "calibration_manifest_sha256",
        "reproducibility_fingerprint",
        "statement",
    }
)


class FormalAggregationError(RuntimeError):
    """Raised when any formal attempt violates the frozen closed-set contract."""


@dataclass(frozen=True, slots=True)
class ValidatedFormalRun:
    run_dir: Path
    run_id: str
    started_at_utc: datetime
    frozen_at_utc: datetime
    result_sha256: str
    provenance_sha256: str
    sha256sums_sha256: str
    formal_run_manifest_sha256: str
    frozen_tolerance_sha256: str
    calibration_manifest_sha256: str
    reproducibility_fingerprint: str
    protocol_sha256: str
    dagkv_snapshot_sha256: str
    nvidia_userspace_bundle_root: str
    nvidia_userspace_bundle_manifest_sha256: str
    nvidia_userspace_bundle_content_digest: str
    nvidia_driver_version: str
    input_inventory: tuple[tuple[str, str, tuple[int, int, int, int, int]], ...]


@dataclass(frozen=True, slots=True)
class ParentEvidence:
    calibration_manifest_sha256: str
    frozen_tolerance_sha256: str
    reproducibility_fingerprint: str
    frozen_at_utc: datetime
    calibration_run_ids: frozenset[str]
    calibration_inventory: tuple[tuple[str, str, tuple[int, int, int, int, int]], ...]
    tolerance_identity: tuple[int, int, int, int, int]
    nvidia_userspace_bundle_root: str
    nvidia_userspace_bundle_manifest_sha256: str
    nvidia_userspace_bundle_content_digest: str
    nvidia_driver_version: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalAggregationError(message)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_bytes(
    path: Path, *, label: str
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    try:
        before = path.lstat()
        raw = read_stable_bytes(path, label=label)
        after = path.lstat()
    except CalibrationEvidenceError as exc:
        raise FormalAggregationError(str(exc)) from exc
    except OSError as exc:
        raise FormalAggregationError(
            f"cannot inspect {label} at {path}: {exc}"
        ) from exc
    require(
        _stat_identity(before) == _stat_identity(after),
        f"{label} changed while read: {path}",
    )
    require(
        before.st_nlink == after.st_nlink == 1,
        f"{label} cannot be hard-linked: {path}",
    )
    return raw, _stat_identity(after)


def sha256_file(path: Path) -> str:
    raw, _ = _stable_bytes(path, label="file")
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalAggregationError(f"cannot canonicalize provenance: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalAggregationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise FormalAggregationError(f"non-finite JSON constant: {value}")


def _decode_json_object(raw: bytes, *, path: Path, label: str) -> dict[str, Any]:
    require(raw.endswith(b"\n"), f"unterminated {label}: {path}")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except FormalAggregationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalAggregationError(f"invalid {label} at {path}: {exc}") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _read_json_object(
    path: Path, *, label: str
) -> tuple[dict[str, Any], bytes, tuple[int, int, int, int, int]]:
    raw, identity = _stable_bytes(path, label=label)
    return _decode_json_object(raw, path=path, label=label), raw, identity


def _exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    require(
        set(payload) == expected,
        f"{label} fields differ: missing={sorted(expected - set(payload))}, "
        f"extra={sorted(set(payload) - expected)}",
    )


def _lower_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _finite_number(value: Any, *, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def _timezone_timestamp(value: Any, *, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} must be non-empty")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FormalAggregationError(f"{label} must be ISO 8601") from exc
    require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        f"{label} must include a timezone",
    )
    return parsed.astimezone(UTC)


def _safe_checksum_name(value: str, *, checksum_path: Path) -> str:
    require(value != "", f"empty checksum path in {checksum_path}")
    require("\\" not in value, f"non-POSIX checksum path in {checksum_path}: {value}")
    pure = PurePosixPath(value)
    require(
        not pure.is_absolute()
        and pure.as_posix() == value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"unsafe checksum path in {checksum_path}: {value}",
    )
    require(value != "SHA256SUMS", "SHA256SUMS cannot list itself")
    require(
        value != FORMAL_RUN_MANIFEST,
        f"{FORMAL_RUN_MANIFEST} must be validated outside SHA256SUMS",
    )
    return value


def _validate_sha256sums(
    run_dir: Path,
) -> tuple[
    dict[str, str],
    str,
    dict[str, bytes],
    dict[str, tuple[int, int, int, int, int]],
]:
    checksum_path = run_dir / "SHA256SUMS"
    try:
        raw, checksum_identity = _stable_bytes(checksum_path, label="SHA256SUMS")
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FormalAggregationError(
            f"invalid SHA256SUMS at {checksum_path}: {exc}"
        ) from exc
    require(raw.endswith(b"\n"), f"unterminated SHA256SUMS: {checksum_path}")
    require(b"\r" not in raw, f"SHA256SUMS must use LF newlines: {checksum_path}")
    lines = text[:-1].split("\n")
    require(lines and all(lines), f"SHA256SUMS contains a blank row: {checksum_path}")

    entries: dict[str, str] = {}
    ordered_names: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        require(
            len(line) > 66 and line[64:66] == "  ",
            f"malformed SHA256SUMS row {checksum_path}:{line_number}",
        )
        digest = _lower_sha256(
            line[:64], label=f"SHA256SUMS digest at {checksum_path}:{line_number}"
        )
        name = _safe_checksum_name(line[66:], checksum_path=checksum_path)
        require(name not in entries, f"duplicate SHA256SUMS path: {name}")
        entries[name] = digest
        ordered_names.append(name)
    require(
        ordered_names == sorted(ordered_names),
        f"SHA256SUMS paths are not sorted: {checksum_path}",
    )

    actual_names: set[str] = set()
    for path in run_dir.rglob("*"):
        require(not path.is_symlink(), f"run artifact cannot be a symlink: {path}")
        if not path.is_file():
            continue
        name = path.relative_to(run_dir).as_posix()
        require(not name.endswith(".tmp"), f"temporary run artifact remains: {path}")
        if name not in {"SHA256SUMS", FORMAL_RUN_MANIFEST}:
            actual_names.add(name)
    require(
        set(entries) == actual_names,
        f"SHA256SUMS coverage mismatch in {run_dir}: "
        f"missing={sorted(actual_names - set(entries))}, "
        f"extra={sorted(set(entries) - actual_names)}",
    )
    require(
        REQUIRED_INTERNAL_ARTIFACTS.issubset(entries),
        f"formal run artifact set is incomplete: {run_dir}; "
        f"missing={sorted(REQUIRED_INTERNAL_ARTIFACTS - set(entries))}",
    )
    for forbidden in FORBIDDEN_RUN_ARTIFACTS:
        require(
            not (run_dir / forbidden).exists(),
            "formal attempt contains aggregate acceptance artifact: "
            f"{run_dir / forbidden}",
        )
    key_payloads: dict[str, bytes] = {}
    identities = {"SHA256SUMS": checksum_identity}
    for name, expected in entries.items():
        path = run_dir / PurePosixPath(name)
        raw_artifact, identity = _stable_bytes(
            path, label=f"checksummed artifact {name}"
        )
        observed = hashlib.sha256(raw_artifact).hexdigest()
        require(
            observed == expected,
            f"checksum mismatch for {path}: expected {expected}, observed {observed}",
        )
        identities[name] = identity
        if name in {"result.json", "provenance.json"}:
            key_payloads[name] = raw_artifact
    return entries, hashlib.sha256(raw).hexdigest(), key_payloads, identities


def _capture_tree_inventory(
    root: Path, *, label: str
) -> tuple[tuple[str, str, tuple[int, int, int, int, int]], ...]:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise FormalAggregationError(
            f"cannot inspect {label} root {root}: {exc}"
        ) from exc
    require(
        stat.S_ISDIR(root_stat.st_mode),
        f"{label} root must be a directory: {root}",
    )
    inventory: list[tuple[str, str, tuple[int, int, int, int, int]]] = [
        (".", "directory", _stat_identity(root_stat))
    ]
    try:
        paths = sorted(
            root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
        )
    except OSError as exc:
        raise FormalAggregationError(f"cannot scan {label} root {root}: {exc}") from exc
    for path in paths:
        try:
            observed = path.lstat()
        except OSError as exc:
            raise FormalAggregationError(
                f"cannot inspect {label} entry {path}: {exc}"
            ) from exc
        relative = path.relative_to(root).as_posix()
        if stat.S_ISREG(observed.st_mode):
            require(
                observed.st_nlink == 1,
                f"{label} file cannot be hard-linked: {path}",
            )
            kind = "file"
        elif stat.S_ISDIR(observed.st_mode):
            kind = "directory"
        else:
            raise FormalAggregationError(
                f"{label} entry must be a regular file or directory: {path}"
            )
        inventory.append((relative, kind, _stat_identity(observed)))
    return tuple(inventory)


def _seal_run_inputs(
    run_dir: Path,
    *,
    hashed_identities: Mapping[str, tuple[int, int, int, int, int]],
    formal_identity: tuple[int, int, int, int, int],
) -> tuple[tuple[str, str, tuple[int, int, int, int, int]], ...]:
    expected = dict(hashed_identities)
    expected[FORMAL_RUN_MANIFEST] = formal_identity
    inventory = _capture_tree_inventory(run_dir, label="formal run")
    observed_files = {
        name: identity for name, kind, identity in inventory if kind == "file"
    }
    require(
        set(observed_files) == set(expected),
        f"formal run file set changed during validation: {run_dir}",
    )
    require(
        observed_files == expected,
        f"formal run input changed during validation: {run_dir}",
    )
    expected_directories = {"."}
    for name in expected:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_directories = {name for name, kind, _ in inventory if kind == "directory"}
    require(
        observed_directories == expected_directories,
        f"formal run directory set differs: {run_dir}",
    )
    return inventory


def _validate_tolerance(value: Any, *, label: str) -> tuple[float, float]:
    require(isinstance(value, dict), f"{label} must be an object")
    _exact_fields(value, frozenset({"atol", "rtol"}), label=label)
    atol = _finite_number(value.get("atol"), label=f"{label}.atol")
    rtol = _finite_number(value.get("rtol"), label=f"{label}.rtol")
    require(
        atol == MAX_FORMAL_ATOL,
        f"{label}.atol must equal the preregistered frozen cap",
    )
    require(rtol == FORMAL_RTOL, f"{label}.rtol must be zero")
    return atol, rtol


def _validate_result(result: dict[str, Any], *, run_dir: Path) -> str:
    _exact_fields(result, RESULT_FIELDS, label=f"result.json in {run_dir}")
    require(
        result.get("schema_version") == PROTOCOL_SCHEMA,
        f"formal result uses a non-v3 protocol: {run_dir}",
    )
    require(result.get("mode") == "formal", f"run is not formal: {run_dir}")
    require(
        result.get("gate_status") == "M2_ITEM8_FORMAL_HOLDOUT_PASSED",
        f"formal holdout gate did not pass: {run_dir}",
    )
    require(result.get("formal_run_passed") is True, f"formal pass is false: {run_dir}")
    require(
        result.get("within_requested_tolerance") is True,
        f"formal tolerance gate did not pass: {run_dir}",
    )
    require(result.get("m2_accepted") is False, f"run claims M2 acceptance: {run_dir}")
    require(
        result.get("m2_item8_accepted") is False,
        f"single run claims item 8 acceptance: {run_dir}",
    )
    run_id = result.get("run_id")
    require(isinstance(run_id, str) and run_id, f"invalid run_id: {run_dir}")
    _timezone_timestamp(
        result.get("completed_at_utc"), label=f"completion in {run_dir}"
    )
    fingerprint = _lower_sha256(
        result.get("reproducibility_fingerprint"),
        label=f"result reproducibility fingerprint in {run_dir}",
    )
    atol, _ = _validate_tolerance(
        result.get("tolerance"), label=f"result tolerance in {run_dir}"
    )

    measurements = result.get("measurements")
    require(isinstance(measurements, dict), f"measurements are missing: {run_dir}")
    require(
        set(measurements) == set(EXPECTED_MEASUREMENTS),
        f"measurement phases differ from the frozen five-stage protocol: {run_dir}",
    )
    token_ids: list[int] = []
    margins: list[float] = []
    for phase in EXPECTED_MEASUREMENTS:
        measurement = measurements[phase]
        require(
            isinstance(measurement, dict), f"invalid {phase} measurement: {run_dir}"
        )
        _exact_fields(
            measurement,
            MEASUREMENT_FIELDS,
            label=f"{phase} measurement in {run_dir}",
        )
        for name in ("request_id", "trace_id"):
            require(
                isinstance(measurement.get(name), str) and measurement[name],
                f"invalid {phase} {name}: {run_dir}",
            )
        token_id = measurement.get("token_id")
        require(
            type(token_id) is int and token_id >= 0, f"invalid {phase} token: {run_dir}"
        )
        expected_cached_tokens = 0 if phase in {"A1", "A2"} else 16
        require(
            measurement.get("num_cached_tokens") == expected_cached_tokens,
            f"{phase} cached-token count differs from the protocol: {run_dir}",
        )
        elapsed_ms = _finite_number(
            measurement.get("elapsed_ms"), label=f"{phase} elapsed_ms in {run_dir}"
        )
        require(elapsed_ms >= 0.0, f"{phase} elapsed_ms is negative: {run_dir}")
        margin = _finite_number(
            measurement.get("top1_margin"),
            label=f"{phase} top1 margin in {run_dir}",
        )
        require(margin > 2 * atol, f"{phase} top1 margin is too small: {run_dir}")
        expected_logits = f"logits_{phase}.npy"
        require(
            measurement.get("logits_file") == expected_logits,
            f"{phase} logits filename differs from the protocol: {run_dir}",
        )
        _lower_sha256(
            measurement.get("logits_sha256"),
            label=f"{phase} logits hash in {run_dir}",
        )
        token_ids.append(token_id)
        margins.append(margin)
    require(len(set(token_ids)) == 1, f"measurement tokens differ: {run_dir}")
    minimum_margin = _finite_number(
        result.get("minimum_top1_margin"),
        label=f"minimum_top1_margin in {run_dir}",
    )
    require(
        minimum_margin == min(margins), f"minimum margin is inconsistent: {run_dir}"
    )

    comparisons = result.get("comparisons")
    require(isinstance(comparisons, list), f"comparisons are missing: {run_dir}")
    seen: set[tuple[str, str]] = set()
    for comparison in comparisons:
        require(isinstance(comparison, dict), f"invalid comparison row: {run_dir}")
        _exact_fields(
            comparison,
            COMPARISON_FIELDS,
            label=f"comparison in {run_dir}",
        )
        pair = (comparison.get("left"), comparison.get("right"))
        require(pair in EXPECTED_PAIRS, f"unexpected comparison {pair}: {run_dir}")
        require(pair not in seen, f"duplicate comparison {pair}: {run_dir}")
        require(
            comparison.get("token_equal") is True,
            f"token mismatch for {pair}: {run_dir}",
        )
        require(
            comparison.get("allclose") is True,
            f"tolerance failure for {pair}: {run_dir}",
        )
        max_abs_error = _finite_number(
            comparison.get("max_abs_error"),
            label=f"max_abs_error for {pair} in {run_dir}",
        )
        max_rel_error = _finite_number(
            comparison.get("max_rel_error"),
            label=f"max_rel_error for {pair} in {run_dir}",
        )
        require(
            max_abs_error >= 0.0 and max_rel_error >= 0.0,
            f"negative comparison error for {pair}: {run_dir}",
        )
        if pair in EXACT_PAIRS:
            require(
                max_abs_error == 0.0 and max_rel_error == 0.0,
                f"exact comparison drifted for {pair}: {run_dir}",
            )
        else:
            require(
                max_abs_error <= atol,
                f"cold/prefix comparison exceeds tolerance for {pair}: {run_dir}",
            )
        seen.add(pair)
    require(seen == EXPECTED_PAIRS, f"comparison pairs are incomplete: {run_dir}")

    prefetch = result.get("prefetch")
    require(
        isinstance(prefetch, dict)
        and set(prefetch) == {"B1", "B2"}
        and all(isinstance(value, dict) for value in prefetch.values()),
        f"prefetch evidence is incomplete: {run_dir}",
    )
    native_bytes = result.get("native_bytes")
    diagnostic_bytes = result.get("diagnostic_bytes")
    require(
        isinstance(native_bytes, dict)
        and isinstance(diagnostic_bytes, dict)
        and set(native_bytes) == TRANSFER_NAMES
        and native_bytes == diagnostic_bytes,
        f"native/diagnostic transfer bytes differ: {run_dir}",
    )
    require(
        all(type(value) is int and value > 0 for value in native_bytes.values()),
        f"transfer byte counts are invalid: {run_dir}",
    )
    transfer_digests = result.get("transfer_digests")
    require(
        isinstance(transfer_digests, dict) and set(transfer_digests) == {"B1", "B2"},
        f"transfer digests are incomplete: {run_dir}",
    )
    b1_digest = _lower_sha256(
        transfer_digests.get("B1"), label=f"B1 digest in {run_dir}"
    )
    b2_digest = _lower_sha256(
        transfer_digests.get("B2"), label=f"B2 digest in {run_dir}"
    )
    require(b1_digest == b2_digest, f"B1/B2 canonical digests differ: {run_dir}")
    require(
        result.get("artifacts")
        == {
            "native_trace": "native_lifecycle.jsonl",
            "diagnostic_trace": "diagnostic_transfers.jsonl",
            "protocol": "protocol.md",
            "provenance": "provenance.json",
        },
        f"result artifact references drifted: {run_dir}",
    )
    return fingerprint


def _validate_content_manifests(provenance: dict[str, Any], *, run_dir: Path) -> None:
    implementation = provenance["implementation"]
    model = provenance["model"]
    runtime = provenance["runtime_binaries"]
    dependencies = provenance["dependencies"]
    require(
        isinstance(implementation, dict),
        f"implementation capture is missing: {run_dir}",
    )
    require(isinstance(model, dict), f"model capture is missing: {run_dir}")
    require(isinstance(runtime, dict), f"runtime capture is missing: {run_dir}")
    require(isinstance(dependencies, dict), f"dependency capture is missing: {run_dir}")

    implementation_files = implementation.get("files")
    require(
        isinstance(implementation_files, list) and implementation_files,
        f"implementation file inventory is missing: {run_dir}",
    )
    for entry in implementation_files:
        require(
            isinstance(entry, dict)
            and set(entry) == {"path", "size", "sha256"}
            and isinstance(entry.get("path"), str)
            and entry["path"]
            and type(entry.get("size")) is int
            and entry["size"] >= 0,
            f"invalid implementation file entry: {run_dir}",
        )
        _lower_sha256(
            entry.get("sha256"), label=f"implementation file hash in {run_dir}"
        )
    require(
        _lower_sha256(
            implementation.get("manifest_sha256"),
            label=f"implementation manifest in {run_dir}",
        )
        == _canonical_digest(implementation_files),
        f"implementation manifest is inconsistent: {run_dir}",
    )

    require(model.get("full_hashes") is True, f"model hashes are incomplete: {run_dir}")
    model_files = model.get("files")
    require(
        isinstance(model_files, list) and model_files,
        f"model files are missing: {run_dir}",
    )
    model_content: list[dict[str, Any]] = []
    for entry in model_files:
        require(isinstance(entry, dict), f"invalid model file entry: {run_dir}")
        _lower_sha256(entry.get("sha256"), label=f"model file hash in {run_dir}")
        model_content.append(
            {key: entry.get(key) for key in ("path", "size", "kind", "sha256")}
        )
    require(
        _lower_sha256(
            model.get("manifest_sha256"), label=f"model manifest in {run_dir}"
        )
        == _canonical_digest(model_content),
        f"model manifest is inconsistent: {run_dir}",
    )

    require(
        runtime.get("full_hashes") is True, f"runtime hashes are incomplete: {run_dir}"
    )
    extensions = runtime.get("vllm_extensions")
    python_entry = runtime.get("python_executable")
    require(
        isinstance(extensions, list) and extensions and isinstance(python_entry, dict),
        f"runtime binary inventory is incomplete: {run_dir}",
    )
    extension_content: list[dict[str, Any]] = []
    for entry in extensions:
        require(isinstance(entry, dict), f"invalid runtime extension entry: {run_dir}")
        _lower_sha256(entry.get("sha256"), label=f"runtime extension hash in {run_dir}")
        extension_content.append(
            {key: entry.get(key) for key in ("path", "size", "sha256")}
        )
    _lower_sha256(python_entry.get("sha256"), label=f"Python binary hash in {run_dir}")
    python_content = {key: python_entry.get(key) for key in ("path", "size", "sha256")}
    runtime_content = {
        "vllm_extensions": extension_content,
        "python_executable": python_content,
    }
    require(
        _lower_sha256(
            runtime.get("manifest_sha256"), label=f"runtime manifest in {run_dir}"
        )
        == _canonical_digest(runtime_content),
        f"runtime manifest is inconsistent: {run_dir}",
    )

    packages = dependencies.get("packages")
    require(
        isinstance(packages, list) and packages, f"dependencies are missing: {run_dir}"
    )
    require(
        _lower_sha256(
            dependencies.get("manifest_sha256"),
            label=f"dependency manifest in {run_dir}",
        )
        == _canonical_digest(packages),
        f"dependency manifest is inconsistent: {run_dir}",
    )


def _validate_frozen_profile(
    provenance: dict[str, Any], *, run_id: str, run_dir: Path
) -> None:
    engine = provenance["engine_config"]
    connector = provenance["connector_config"]
    assert isinstance(engine, dict)
    assert isinstance(connector, dict)
    _exact_fields(engine, ENGINE_CONFIG_FIELDS, label=f"engine config in {run_dir}")
    _exact_fields(
        connector,
        CONNECTOR_CONFIG_FIELDS,
        label=f"connector config in {run_dir}",
    )
    require(
        isinstance(engine.get("model"), str) and engine["model"],
        f"engine model path is invalid: {run_dir}",
    )
    expected_engine = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "block_size": 16,
        "max_model_len": 64,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 64,
        "gpu_memory_utilization": 0.82,
        "disable_hybrid_kv_cache_manager": True,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "scheduling_policy": "fcfs",
        "seed": 20260724,
        "dtype": "bfloat16",
        "attention_config": {
            "backend": "FLASH_ATTN",
            "flash_attn_version": 2,
        },
        "trust_remote_code": False,
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
    }
    for name, expected in expected_engine.items():
        observed = engine.get(name)
        matches = (
            observed is expected if type(expected) is bool else observed == expected
        )
        require(matches, f"frozen engine setting {name} drifted: {run_dir}")

    expected_connector = {
        "cpu_bytes_to_use": provenance["cpu_bytes"],
        "spec_name": "DAGKVDiagnosticCPUOffloadingSpec",
        "spec_module_path": "dagkv_vllm_m2.spec",
        "dagkv_diagnostic_trace_file": str(run_dir / "diagnostic_transfers.jsonl"),
        "dagkv_diagnostic_run_id": run_id,
        "dagkv_diagnostic_phase": "ABBA",
        "fanout_layerwise_load": False,
        "lifecycle_accounting_enabled": True,
    }
    for name, expected in expected_connector.items():
        observed = connector.get(name)
        matches = (
            observed is expected if type(expected) is bool else observed == expected
        )
        require(matches, f"frozen connector setting {name} drifted: {run_dir}")


def _validate_provenance(
    provenance: dict[str, Any],
    *,
    run_id: str,
    result: dict[str, Any],
    run_dir: Path,
) -> tuple[
    str,
    str,
    str,
    str,
    datetime,
    datetime,
    str,
    str,
    str,
    str,
]:
    _exact_fields(provenance, PROVENANCE_FIELDS, label=f"provenance.json in {run_dir}")
    require(
        provenance.get("schema_version") == PROTOCOL_SCHEMA,
        f"formal provenance uses a non-v3 protocol: {run_dir}",
    )
    require(provenance.get("mode") == "formal", f"provenance is not formal: {run_dir}")
    require(provenance.get("run_id") == run_id, f"provenance run_id differs: {run_dir}")
    require(
        provenance.get("full_provenance") is True,
        f"full provenance is absent: {run_dir}",
    )
    started_at = _timezone_timestamp(
        provenance.get("started_at_utc"), label=f"start time in {run_dir}"
    )
    require(
        isinstance(provenance.get("argv"), list)
        and all(isinstance(value, str) for value in provenance["argv"]),
        f"argv capture is invalid: {run_dir}",
    )
    require(
        isinstance(provenance.get("python"), str)
        and isinstance(provenance.get("executable"), str),
        f"Python provenance is invalid: {run_dir}",
    )
    require(
        provenance.get("prompt_token_ids") == list(range(1000, 1017))
        and provenance.get("block_size") == 16,
        f"provenance prompt profile drifted: {run_dir}",
    )
    require(
        type(provenance.get("cpu_bytes")) is int and provenance["cpu_bytes"] > 0,
        f"cpu_bytes is invalid: {run_dir}",
    )
    require(
        provenance.get("tolerance") == result.get("tolerance"),
        f"result/provenance tolerance differs: {run_dir}",
    )

    frozen = provenance.get("frozen_tolerance")
    require(isinstance(frozen, dict), f"frozen tolerance is missing: {run_dir}")
    _exact_fields(
        frozen, FROZEN_TOLERANCE_FIELDS, label=f"frozen tolerance in {run_dir}"
    )
    require(
        {"atol": frozen.get("atol"), "rtol": frozen.get("rtol")}
        == result.get("tolerance"),
        f"frozen/result tolerance differs: {run_dir}",
    )
    _validate_tolerance(result.get("tolerance"), label=f"formal tolerance in {run_dir}")
    frozen_at = _timezone_timestamp(
        frozen.get("frozen_at_utc"), label=f"frozen_at_utc in {run_dir}"
    )
    tolerance_sha = _lower_sha256(
        frozen.get("file_sha256"), label=f"frozen tolerance hash in {run_dir}"
    )
    calibration_sha = _lower_sha256(
        frozen.get("calibration_manifest_sha256"),
        label=f"calibration manifest hash in {run_dir}",
    )
    require(
        frozen.get("calibration_run_count") == CALIBRATION_RUN_COUNT
        and type(frozen.get("calibration_run_count")) is int,
        f"frozen calibration count differs: {run_dir}",
    )
    require(
        frozen.get("derivation") == TOLERANCE_DERIVATION,
        f"frozen tolerance derivation differs: {run_dir}",
    )

    cohort = provenance.get("calibration_cohort")
    require(
        isinstance(cohort, dict), f"calibration cohort binding is missing: {run_dir}"
    )
    _exact_fields(
        cohort,
        frozenset({"path", "sha256", "run_count"}),
        label=f"calibration cohort binding in {run_dir}",
    )
    require(
        isinstance(cohort.get("path"), str)
        and cohort["path"]
        and cohort.get("sha256") == calibration_sha
        and cohort.get("run_count") == CALIBRATION_RUN_COUNT,
        f"calibration cohort binding differs: {run_dir}",
    )

    components = provenance.get("reproducibility_components")
    require(
        isinstance(components, dict),
        f"reproducibility components are missing: {run_dir}",
    )
    _exact_fields(
        components,
        REPRODUCIBILITY_COMPONENT_FIELDS,
        label=f"reproducibility components in {run_dir}",
    )
    fingerprint = _lower_sha256(
        provenance.get("reproducibility_fingerprint"),
        label=f"provenance reproducibility fingerprint in {run_dir}",
    )
    require(
        fingerprint == _canonical_digest(components),
        f"reproducibility fingerprint is inconsistent: {run_dir}",
    )
    require(
        fingerprint
        == result.get("reproducibility_fingerprint")
        == frozen.get("reproducibility_fingerprint"),
        f"result/provenance/frozen fingerprint differs: {run_dir}",
    )

    implementation = provenance.get("implementation")
    dagkv_git = provenance.get("dagkv_git")
    vllm_git = provenance.get("vllm_git")
    nvidia_userspace = provenance.get("nvidia_driver_userspace")
    postflight = provenance.get("postflight")
    for label, value in (
        ("implementation", implementation),
        ("dagkv_git", dagkv_git),
        ("vllm_git", vllm_git),
        ("nvidia_driver_userspace", nvidia_userspace),
        ("postflight", postflight),
        ("preflight", provenance.get("preflight")),
        ("system", provenance.get("system")),
        ("engine_config", provenance.get("engine_config")),
        ("connector_config", provenance.get("connector_config")),
    ):
        require(
            isinstance(value, dict), f"{label} is missing from provenance: {run_dir}"
        )
    assert isinstance(implementation, dict)
    assert isinstance(dagkv_git, dict)
    assert isinstance(vllm_git, dict)
    assert isinstance(nvidia_userspace, dict)
    assert isinstance(postflight, dict)
    _validate_frozen_profile(provenance, run_id=run_id, run_dir=run_dir)
    implementation_sha = _lower_sha256(
        implementation.get("manifest_sha256"),
        label=f"implementation manifest in {run_dir}",
    )
    dagkv_snapshot = _lower_sha256(
        dagkv_git.get("snapshot_sha256"), label=f"DAGKV snapshot in {run_dir}"
    )
    vllm_snapshot = _lower_sha256(
        vllm_git.get("snapshot_sha256"), label=f"vLLM snapshot in {run_dir}"
    )
    nvidia_content_digest = _lower_sha256(
        nvidia_userspace.get("content_digest"),
        label=f"NVIDIA userspace content digest in {run_dir}",
    )
    nvidia_manifest_sha = _lower_sha256(
        nvidia_userspace.get("manifest_sha256"),
        label=f"NVIDIA userspace manifest in {run_dir}",
    )
    nvidia_root = nvidia_userspace.get("root")
    require(
        isinstance(nvidia_root, str) and Path(nvidia_root).is_absolute(),
        f"NVIDIA userspace bundle root is invalid: {run_dir}",
    )
    nvidia_driver_version = nvidia_userspace.get("kernel_module_version")
    require(
        isinstance(nvidia_driver_version, str) and nvidia_driver_version,
        f"NVIDIA driver version is invalid: {run_dir}",
    )
    require(
        dagkv_git.get("dirty") is False, f"formal DAGKV worktree was dirty: {run_dir}"
    )
    require(
        postflight.get("implementation_manifest_sha256") == implementation_sha
        and postflight.get("dagkv_git_snapshot_sha256") == dagkv_snapshot
        and postflight.get("vllm_git_snapshot_sha256") == vllm_snapshot
        and postflight.get("model_file_stats_unchanged") is True
        and postflight.get("runtime_binary_stats_unchanged") is True
        and postflight.get("nvidia_driver_userspace_content_digest")
        == nvidia_content_digest
        and postflight.get("nvidia_driver_userspace_manifest_sha256")
        == nvidia_manifest_sha
        and postflight.get("nvidia_driver_userspace_unchanged") is True
        and postflight.get("libcuda_mapping_unchanged") is True,
        f"postflight provenance differs: {run_dir}",
    )
    _validate_content_manifests(provenance, run_dir=run_dir)

    model_sha = _lower_sha256(
        provenance["model"].get("manifest_sha256"), label=f"model manifest in {run_dir}"
    )
    runtime_sha = _lower_sha256(
        provenance["runtime_binaries"].get("manifest_sha256"),
        label=f"runtime manifest in {run_dir}",
    )
    dependency_sha = _lower_sha256(
        provenance["dependencies"].get("manifest_sha256"),
        label=f"dependency manifest in {run_dir}",
    )
    require(
        components["implementation_manifest_sha256"] == implementation_sha
        and components["vllm_snapshot_sha256"] == vllm_snapshot
        and components["model_manifest_sha256"] == model_sha
        and components["runtime_binary_manifest_sha256"] == runtime_sha
        and components["dependency_manifest_sha256"] == dependency_sha,
        f"reproducibility component hashes differ from provenance: {run_dir}",
    )
    require(
        components["nvidia_driver_userspace_content_digest"] == nvidia_content_digest,
        f"NVIDIA reproducibility component differs from provenance: {run_dir}",
    )
    require(
        components["system"] == provenance["system"]
        and components["prompt_token_ids"] == provenance["prompt_token_ids"]
        and components["block_size"] == provenance["block_size"]
        and components["cpu_bytes"] == provenance["cpu_bytes"]
        and components["engine_config"] == provenance["engine_config"],
        f"reproducibility components differ from runtime settings: {run_dir}",
    )
    dynamic_connector_keys = {
        "dagkv_diagnostic_trace_file",
        "dagkv_diagnostic_run_id",
    }
    static_connector = {
        key: value
        for key, value in provenance["connector_config"].items()
        if key not in dynamic_connector_keys
    }
    require(
        components["connector_config"] == static_connector,
        f"connector reproducibility components differ: {run_dir}",
    )
    return (
        fingerprint,
        tolerance_sha,
        calibration_sha,
        dagkv_snapshot,
        started_at,
        frozen_at,
        nvidia_root,
        nvidia_manifest_sha,
        nvidia_content_digest,
        nvidia_driver_version,
    )


def _validate_capture_artifacts(
    capture: Any,
    *,
    label: str,
    entries: dict[str, str],
    run_dir: Path,
) -> None:
    require(isinstance(capture, dict), f"{label} Git capture is missing: {run_dir}")
    artifact_names: dict[str, str] = {}
    for path_key, digest_key in (
        ("tracked_patch", "tracked_patch_sha256"),
        ("untracked_archive", "untracked_archive_sha256"),
    ):
        name = capture.get(path_key)
        require(
            isinstance(name, str) and name, f"{label} {path_key} is invalid: {run_dir}"
        )
        digest = _lower_sha256(
            capture.get(digest_key), label=f"{label} {digest_key} in {run_dir}"
        )
        require(
            entries.get(name) == digest,
            f"{label} {path_key} hash differs from SHA256SUMS: {run_dir}",
        )
        artifact_names[path_key] = name

    head = capture.get("head")
    untracked = capture.get("untracked_files")
    require(
        isinstance(head, str)
        and len(head) == 40
        and all(character in "0123456789abcdef" for character in head),
        f"{label} Git HEAD is invalid: {run_dir}",
    )
    require(
        isinstance(untracked, list),
        f"{label} untracked inventory is invalid: {run_dir}",
    )
    status = capture.get("status_short")
    require(
        isinstance(status, list)
        and all(isinstance(row, str) and row for row in status),
        f"{label} Git status inventory is invalid: {run_dir}",
    )
    require(
        type(capture.get("dirty")) is bool and capture["dirty"] is bool(status),
        f"{label} Git dirty flag differs from status: {run_dir}",
    )
    tracked_name = artifact_names["tracked_patch"]
    snapshot = {
        "head": head,
        "tracked_diff_sha256": entries[tracked_name],
        "tracked_diff_bytes": (run_dir / PurePosixPath(tracked_name)).stat().st_size,
        "untracked": untracked,
    }
    require(
        _lower_sha256(
            capture.get("snapshot_sha256"),
            label=f"{label} Git snapshot in {run_dir}",
        )
        == _canonical_digest(snapshot),
        f"{label} Git snapshot is inconsistent: {run_dir}",
    )
    if label == "DAGKV":
        require(
            status == [] and untracked == [] and snapshot["tracked_diff_bytes"] == 0,
            f"formal DAGKV source capture is not clean: {run_dir}",
        )


def _validate_formal_manifest(
    manifest: dict[str, Any],
    *,
    run_id: str,
    result_sha256: str,
    provenance_sha256: str,
    sha256sums_sha256: str,
    fingerprint: str,
    tolerance_sha256: str,
    calibration_sha256: str,
    run_dir: Path,
) -> None:
    _exact_fields(
        manifest, FORMAL_RUN_FIELDS, label=f"formal run manifest in {run_dir}"
    )
    require(
        manifest.get("schema_version") == FORMAL_RUN_SCHEMA,
        f"formal run manifest schema differs: {run_dir}",
    )
    require(
        manifest.get("run_id") == run_id, f"formal manifest run_id differs: {run_dir}"
    )
    _timezone_timestamp(
        manifest.get("completed_at_utc"), label=f"formal completion in {run_dir}"
    )
    expected = {
        "result_sha256": result_sha256,
        "provenance_sha256": provenance_sha256,
        "sha256sums_sha256": sha256sums_sha256,
        "frozen_tolerance_sha256": tolerance_sha256,
        "calibration_manifest_sha256": calibration_sha256,
        "reproducibility_fingerprint": fingerprint,
    }
    for name, digest in expected.items():
        _lower_sha256(manifest.get(name), label=f"formal manifest {name} in {run_dir}")
        require(
            manifest.get(name) == digest,
            f"formal manifest {name} does not bind the actual run artifact: {run_dir}",
        )
    require(
        manifest.get("statement") == FORMAL_RUN_STATEMENT,
        f"formal run statement differs from the runner contract: {run_dir}",
    )


def _validate_run(run_dir: Path) -> ValidatedFormalRun:
    result_path = run_dir / "result.json"
    provenance_path = run_dir / "provenance.json"
    formal_path = run_dir / FORMAL_RUN_MANIFEST
    require(
        result_path.is_file() and not result_path.is_symlink(),
        f"missing result.json: {run_dir}",
    )
    require(
        provenance_path.is_file() and not provenance_path.is_symlink(),
        f"missing provenance.json: {run_dir}",
    )
    require(
        formal_path.is_file() and not formal_path.is_symlink(),
        f"missing {FORMAL_RUN_MANIFEST}: {run_dir}",
    )
    entries, checksum_sha, key_payloads, hashed_identities = _validate_sha256sums(
        run_dir
    )
    result_raw = key_payloads["result.json"]
    result = _decode_json_object(result_raw, path=result_path, label="result.json")
    fingerprint = _validate_result(result, run_dir=run_dir)
    for phase in EXPECTED_MEASUREMENTS:
        name = f"logits_{phase}.npy"
        require(
            result["measurements"][phase]["logits_sha256"] == entries[name],
            f"{phase} logits hash differs from SHA256SUMS: {run_dir}",
        )

    provenance_raw = key_payloads["provenance.json"]
    provenance = _decode_json_object(
        provenance_raw, path=provenance_path, label="provenance.json"
    )
    run_id = result["run_id"]
    (
        provenance_fingerprint,
        tolerance_sha,
        calibration_sha,
        dagkv_snapshot,
        started_at,
        frozen_at,
        nvidia_root,
        nvidia_manifest_sha,
        nvidia_content_digest,
        nvidia_driver_version,
    ) = _validate_provenance(
        provenance,
        run_id=run_id,
        result=result,
        run_dir=run_dir,
    )
    require(
        fingerprint == provenance_fingerprint, f"run fingerprint differs: {run_dir}"
    )
    implementation_sha = provenance["implementation"]["manifest_sha256"]
    try:
        raw = validate_raw_run(run_dir)
    except M2RawReplayError as exc:
        raise FormalAggregationError(
            f"raw artifact replay failed for {run_dir}: {exc}"
        ) from exc
    require(raw.mode == "formal", f"raw replay mode differs: {run_dir}")
    require(raw.run_id == run_id, f"raw replay run_id differs: {run_dir}")
    require(
        raw.reproducibility_fingerprint == fingerprint,
        f"raw replay fingerprint differs: {run_dir}",
    )
    require(
        raw.implementation_manifest_sha256 == implementation_sha,
        f"raw replay implementation differs: {run_dir}",
    )
    _validate_capture_artifacts(
        provenance["dagkv_git"], label="DAGKV", entries=entries, run_dir=run_dir
    )
    _validate_capture_artifacts(
        provenance["vllm_git"], label="vLLM", entries=entries, run_dir=run_dir
    )

    result_sha = entries["result.json"]
    provenance_sha = entries["provenance.json"]
    formal, formal_raw, formal_identity = _read_json_object(
        formal_path, label=FORMAL_RUN_MANIFEST
    )
    _validate_formal_manifest(
        formal,
        run_id=run_id,
        result_sha256=result_sha,
        provenance_sha256=provenance_sha,
        sha256sums_sha256=checksum_sha,
        fingerprint=fingerprint,
        tolerance_sha256=tolerance_sha,
        calibration_sha256=calibration_sha,
        run_dir=run_dir,
    )
    input_inventory = _seal_run_inputs(
        run_dir,
        hashed_identities=hashed_identities,
        formal_identity=formal_identity,
    )
    return ValidatedFormalRun(
        run_dir=run_dir,
        run_id=run_id,
        started_at_utc=started_at,
        frozen_at_utc=frozen_at,
        result_sha256=result_sha,
        provenance_sha256=provenance_sha,
        sha256sums_sha256=checksum_sha,
        formal_run_manifest_sha256=hashlib.sha256(formal_raw).hexdigest(),
        frozen_tolerance_sha256=tolerance_sha,
        calibration_manifest_sha256=calibration_sha,
        reproducibility_fingerprint=fingerprint,
        protocol_sha256=entries["protocol.md"],
        dagkv_snapshot_sha256=dagkv_snapshot,
        nvidia_userspace_bundle_root=nvidia_root,
        nvidia_userspace_bundle_manifest_sha256=nvidia_manifest_sha,
        nvidia_userspace_bundle_content_digest=nvidia_content_digest,
        nvidia_driver_version=nvidia_driver_version,
        input_inventory=input_inventory,
    )


def _discover_run_dirs(campaign_dir: Path) -> list[Path]:
    for path in campaign_dir.rglob("*"):
        require(
            not path.is_symlink(), f"formal campaign cannot contain a symlink: {path}"
        )
    run_dirs = sorted(path for path in campaign_dir.iterdir() if path.is_dir())
    require(
        len(run_dirs) == FORMAL_RUN_COUNT,
        f"formal campaign requires exactly {FORMAL_RUN_COUNT} direct attempt "
        f"directories; observed {len(run_dirs)}",
    )
    return run_dirs


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish the fully synced inode without permitting replacement.
        os.link(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_parent_evidence(
    calibration_manifest: Path, frozen_tolerance: Path
) -> ParentEvidence:
    calibration_manifest = calibration_manifest.expanduser().absolute()
    frozen_tolerance = frozen_tolerance.expanduser().absolute()
    tolerance, tolerance_raw, tolerance_identity = _read_json_object(
        frozen_tolerance, label="frozen tolerance"
    )
    _exact_fields(tolerance, TOLERANCE_FIELDS, label="frozen tolerance")
    require(
        tolerance.get("schema_version") == TOLERANCE_SCHEMA,
        "frozen tolerance schema drifted",
    )
    require(tolerance.get("frozen") is True, "tolerance is not frozen")
    require(
        _finite_number(tolerance.get("atol"), label="frozen tolerance atol")
        == MAX_FORMAL_ATOL,
        "frozen tolerance atol differs from the preregistered cap",
    )
    require(
        _finite_number(tolerance.get("rtol"), label="frozen tolerance rtol")
        == FORMAL_RTOL,
        "frozen tolerance rtol must be zero",
    )
    require(
        type(tolerance.get("calibration_run_count")) is int
        and tolerance["calibration_run_count"] == CALIBRATION_RUN_COUNT,
        "frozen tolerance must bind exactly 59 calibration runs",
    )
    require(
        tolerance.get("derivation") == TOLERANCE_DERIVATION,
        "frozen tolerance derivation drifted",
    )
    calibration_sha = _lower_sha256(
        tolerance.get("calibration_manifest_sha256"),
        label="frozen tolerance calibration manifest hash",
    )
    fingerprint = _lower_sha256(
        tolerance.get("reproducibility_fingerprint"),
        label="frozen tolerance reproducibility fingerprint",
    )
    frozen_at = _timezone_timestamp(
        tolerance.get("frozen_at_utc"), label="frozen tolerance timestamp"
    )
    tolerance_sha = hashlib.sha256(tolerance_raw).hexdigest()

    try:
        manifest, observed_calibration_sha, evidence = (
            validate_published_calibration_bundle(
                calibration_manifest,
                run_validator=_validate_calibration_run,
                expected_manifest_sha256=calibration_sha,
            )
        )
    except CalibrationEvidenceError as exc:
        raise FormalAggregationError(
            f"invalid published calibration evidence: {exc}"
        ) from exc
    require(
        observed_calibration_sha == calibration_sha,
        "frozen tolerance references a different calibration manifest",
    )
    require(
        manifest.get("run_count") == CALIBRATION_RUN_COUNT
        and len(evidence.runs) == CALIBRATION_RUN_COUNT,
        "published calibration evidence must contain exactly 59 runs",
    )
    require(
        manifest.get("formal_atol") == MAX_FORMAL_ATOL
        and manifest.get("formal_rtol") == FORMAL_RTOL,
        "published calibration formal tolerance drifted",
    )
    require(
        manifest.get("reproducibility_fingerprint")
        == evidence.reproducibility_fingerprint
        == fingerprint,
        "calibration/tolerance reproducibility fingerprint differs",
    )
    manifest_raw, _ = _stable_bytes(calibration_manifest, label="calibration manifest")
    require(
        hashlib.sha256(manifest_raw).hexdigest() == calibration_sha,
        "calibration manifest changed after parent validation",
    )
    calibration_inventory = _capture_tree_inventory(
        calibration_manifest.parent, label="calibration campaign"
    )
    return ParentEvidence(
        calibration_manifest_sha256=calibration_sha,
        frozen_tolerance_sha256=tolerance_sha,
        reproducibility_fingerprint=fingerprint,
        frozen_at_utc=frozen_at,
        calibration_run_ids=frozenset(run.run_id for run in evidence.runs),
        calibration_inventory=calibration_inventory,
        tolerance_identity=tolerance_identity,
        nvidia_userspace_bundle_root=evidence.nvidia_userspace_bundle_root,
        nvidia_userspace_bundle_manifest_sha256=(
            evidence.nvidia_userspace_bundle_manifest_sha256
        ),
        nvidia_userspace_bundle_content_digest=(
            evidence.nvidia_userspace_bundle_content_digest
        ),
        nvidia_driver_version=evidence.nvidia_driver_version,
    )


def _verify_prepublication_inputs(
    *,
    campaign_dir: Path,
    validated: Sequence[ValidatedFormalRun],
    calibration_manifest: Path,
    frozen_tolerance: Path,
    parent: ParentEvidence,
) -> None:
    rescanned = _discover_run_dirs(campaign_dir)
    require(
        rescanned == [run.run_dir for run in validated],
        "formal attempt directory set changed during aggregation",
    )
    for run in validated:
        require(
            _capture_tree_inventory(run.run_dir, label="formal run")
            == run.input_inventory,
            f"formal run input changed before publication: {run.run_dir}",
        )
    require(
        _capture_tree_inventory(
            calibration_manifest.parent, label="calibration campaign"
        )
        == parent.calibration_inventory,
        "calibration campaign changed before formal publication",
    )
    tolerance_raw, tolerance_identity = _stable_bytes(
        frozen_tolerance, label="frozen tolerance"
    )
    require(
        tolerance_identity == parent.tolerance_identity
        and hashlib.sha256(tolerance_raw).hexdigest() == parent.frozen_tolerance_sha256,
        "frozen tolerance changed before formal publication",
    )
    manifest_raw, _ = _stable_bytes(calibration_manifest, label="calibration manifest")
    require(
        hashlib.sha256(manifest_raw).hexdigest() == parent.calibration_manifest_sha256,
        "calibration manifest changed before formal publication",
    )


def aggregate_campaign(
    campaign_dir: Path,
    *,
    calibration_manifest: Path,
    frozen_tolerance: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the complete formal closed set and atomically accept item 8."""

    require(
        campaign_dir.is_dir() and not campaign_dir.is_symlink(),
        f"formal campaign directory is missing or unsafe: {campaign_dir}",
    )
    campaign_dir = campaign_dir.resolve()
    destination_candidate = (
        output_path if output_path is not None else campaign_dir / ACCEPTANCE_MANIFEST
    )
    require(
        not destination_candidate.is_symlink(),
        f"formal output cannot be a symlink: {destination_candidate}",
    )
    destination = destination_candidate.resolve()
    require(
        destination.name == ACCEPTANCE_MANIFEST,
        f"formal output must be named {ACCEPTANCE_MANIFEST}",
    )
    require(
        not destination.exists(),
        f"formal acceptance manifest already exists: {destination}",
    )
    calibration_manifest = calibration_manifest.expanduser().absolute()
    frozen_tolerance = frozen_tolerance.expanduser().absolute()
    require(
        not destination.is_relative_to(calibration_manifest.parent.resolve()),
        "formal acceptance output must be outside the calibration campaign",
    )
    require(
        destination != frozen_tolerance.resolve(),
        "formal acceptance output cannot replace the frozen tolerance",
    )
    run_dirs = _discover_run_dirs(campaign_dir)
    require(
        all(not destination.is_relative_to(run_dir) for run_dir in run_dirs),
        "formal acceptance output must be outside every attempt directory",
    )

    parent = _validate_parent_evidence(calibration_manifest, frozen_tolerance)
    validated = [_validate_run(run_dir) for run_dir in run_dirs]
    run_ids = {run.run_id for run in validated}
    result_hashes = {run.result_sha256 for run in validated}
    formal_hashes = {run.formal_run_manifest_sha256 for run in validated}
    require(len(run_ids) == FORMAL_RUN_COUNT, "formal run IDs must be unique")
    require(
        len(result_hashes) == FORMAL_RUN_COUNT, "formal result hashes must be unique"
    )
    require(
        len(formal_hashes) == FORMAL_RUN_COUNT,
        "formal per-run manifest hashes must be unique",
    )

    consistent_fields = {
        "reproducibility fingerprints": {
            run.reproducibility_fingerprint for run in validated
        },
        "frozen tolerance hashes": {run.frozen_tolerance_sha256 for run in validated},
        "calibration manifest hashes": {
            run.calibration_manifest_sha256 for run in validated
        },
        "protocol hashes": {run.protocol_sha256 for run in validated},
        "DAGKV snapshots": {run.dagkv_snapshot_sha256 for run in validated},
        "NVIDIA userspace bundle roots": {
            run.nvidia_userspace_bundle_root for run in validated
        },
        "NVIDIA userspace manifest hashes": {
            run.nvidia_userspace_bundle_manifest_sha256 for run in validated
        },
        "NVIDIA userspace content digests": {
            run.nvidia_userspace_bundle_content_digest for run in validated
        },
        "NVIDIA driver versions": {run.nvidia_driver_version for run in validated},
    }
    for label, values in consistent_fields.items():
        require(len(values) == 1, f"formal {label} differ")

    require(
        all(
            run.frozen_tolerance_sha256 == parent.frozen_tolerance_sha256
            for run in validated
        ),
        "formal runs do not bind the supplied frozen tolerance",
    )
    require(
        all(
            run.calibration_manifest_sha256 == parent.calibration_manifest_sha256
            for run in validated
        ),
        "formal runs do not bind the supplied calibration manifest",
    )
    require(
        all(
            run.reproducibility_fingerprint == parent.reproducibility_fingerprint
            for run in validated
        ),
        "formal runs do not match the parent reproducibility fingerprint",
    )
    require(
        all(
            run.nvidia_userspace_bundle_root == parent.nvidia_userspace_bundle_root
            and run.nvidia_userspace_bundle_manifest_sha256
            == parent.nvidia_userspace_bundle_manifest_sha256
            and run.nvidia_userspace_bundle_content_digest
            == parent.nvidia_userspace_bundle_content_digest
            and run.nvidia_driver_version == parent.nvidia_driver_version
            for run in validated
        ),
        "formal runs do not bind the calibration NVIDIA userspace bundle",
    )
    require(
        all(run.frozen_at_utc == parent.frozen_at_utc for run in validated),
        "formal runs do not record the supplied tolerance freeze timestamp",
    )
    require(
        all(parent.frozen_at_utc < run.started_at_utc for run in validated),
        "frozen tolerance must predate every formal holdout",
    )
    require(
        run_ids.isdisjoint(parent.calibration_run_ids),
        "calibration and formal run IDs must be disjoint",
    )

    ordered = sorted(validated, key=lambda run: run.run_id)
    manifest = {
        "schema_version": ITEM8_ACCEPTANCE_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "gate_status": "M2_ITEM8_ACCEPTED",
        "run_count": FORMAL_RUN_COUNT,
        "passed_run_count": FORMAL_RUN_COUNT,
        "m2_item8_accepted": True,
        "m2_accepted": False,
        "performance_claims_supported": False,
        "frozen_tolerance_sha256": ordered[0].frozen_tolerance_sha256,
        "calibration_manifest_sha256": ordered[0].calibration_manifest_sha256,
        "reproducibility_fingerprint": ordered[0].reproducibility_fingerprint,
        "protocol_sha256": ordered[0].protocol_sha256,
        "nvidia_userspace_bundle_root": ordered[0].nvidia_userspace_bundle_root,
        "nvidia_userspace_bundle_manifest_sha256": (
            ordered[0].nvidia_userspace_bundle_manifest_sha256
        ),
        "nvidia_userspace_bundle_content_digest": (
            ordered[0].nvidia_userspace_bundle_content_digest
        ),
        "nvidia_driver_version": ordered[0].nvidia_driver_version,
        "runs": [
            {
                "run_id": run.run_id,
                "formal_run_manifest_sha256": run.formal_run_manifest_sha256,
                "result_sha256": run.result_sha256,
                "provenance_sha256": run.provenance_sha256,
                "sha256sums_sha256": run.sha256sums_sha256,
            }
            for run in ordered
        ],
        "statement": ACCEPTANCE_STATEMENT,
    }
    _verify_prepublication_inputs(
        campaign_dir=campaign_dir,
        validated=validated,
        calibration_manifest=calibration_manifest,
        frozen_tolerance=frozen_tolerance,
        parent=parent,
    )
    _write_json_atomic(destination, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--frozen-tolerance", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or args.campaign_dir / ACCEPTANCE_MANIFEST
    try:
        manifest = aggregate_campaign(
            args.campaign_dir,
            calibration_manifest=args.calibration_manifest,
            frozen_tolerance=args.frozen_tolerance,
            output_path=output,
        )
    except (FormalAggregationError, OSError) as exc:
        print(f"M2 formal aggregation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"M2 item 8 accepted: {output.resolve()} "
        f"({manifest['passed_run_count']}/{manifest['run_count']} holdouts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
