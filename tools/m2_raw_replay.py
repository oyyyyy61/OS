#!/usr/bin/env python3
"""Replay every raw artifact in one M2 ABBA run and fail closed on drift."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.nvidia_driver_userspace_bundle import (
        CONTENT_DIGEST_DOMAIN,
        RUNTIME_DERIVATION,
    )
    from tools.nvidia_driver_userspace_bundle import (
        MANIFEST_FIELDS as NVIDIA_MANIFEST_FIELDS,
    )
    from tools.nvidia_driver_userspace_bundle import (
        MANIFEST_SCHEMA as NVIDIA_MANIFEST_SCHEMA,
    )
    from tools.nvidia_driver_userspace_bundle import (
        PACKAGE_FIELDS as NVIDIA_PACKAGE_FIELDS,
    )
    from tools.nvidia_driver_userspace_bundle import (
        RUNTIME_DIRECTORY_FIELDS as NVIDIA_RUNTIME_DIRECTORY_FIELDS,
    )
    from tools.nvidia_driver_userspace_bundle import (
        RUNTIME_FILE_FIELDS as NVIDIA_RUNTIME_FILE_FIELDS,
    )
    from tools.nvidia_driver_userspace_bundle import (
        RUNTIME_SYMLINK_FIELDS as NVIDIA_RUNTIME_SYMLINK_FIELDS,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from nvidia_driver_userspace_bundle import (  # type: ignore[no-redef]
        CONTENT_DIGEST_DOMAIN,
        RUNTIME_DERIVATION,
    )
    from nvidia_driver_userspace_bundle import (
        MANIFEST_FIELDS as NVIDIA_MANIFEST_FIELDS,
    )
    from nvidia_driver_userspace_bundle import (
        MANIFEST_SCHEMA as NVIDIA_MANIFEST_SCHEMA,
    )
    from nvidia_driver_userspace_bundle import (
        PACKAGE_FIELDS as NVIDIA_PACKAGE_FIELDS,
    )
    from nvidia_driver_userspace_bundle import (
        RUNTIME_DIRECTORY_FIELDS as NVIDIA_RUNTIME_DIRECTORY_FIELDS,
    )
    from nvidia_driver_userspace_bundle import (
        RUNTIME_FILE_FIELDS as NVIDIA_RUNTIME_FILE_FIELDS,
    )
    from nvidia_driver_userspace_bundle import (
        RUNTIME_SYMLINK_FIELDS as NVIDIA_RUNTIME_SYMLINK_FIELDS,
    )

PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v3"
DIAGNOSTIC_SCHEMA = "dagkv.vllm_m2.transfer_probe.v1"
LIFECYCLE_SCHEMA = "kv_lifecycle_event_v2"
NVIDIA_PACKAGE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.-]*")
FROZEN_QWEN3_8B_VOCAB_SIZE = 151_936
EXPECTED_PHASES = ("A1", "G", "B1", "B2", "A2")
TOLERANT_PAIRS = (("A1", "G"), ("A1", "B1"), ("A1", "B2"))
EXACT_PAIRS = (("A1", "A2"), ("G", "B1"), ("G", "B2"), ("B1", "B2"))
EXPECTED_PAIRS = (*TOLERANT_PAIRS, *EXACT_PAIRS)
TRANSFER_SPECS = {
    "B1_D2H": ("B1", "D2H", "producer"),
    "B1_H2D": ("B1", "H2D", "prefetch"),
    "B2_D2H": ("B2", "D2H", "producer"),
    "B2_H2D": ("B2", "H2D", "prefetch"),
}
EXPECTED_ARTIFACTS = frozenset(
    {
        "diagnostic_transfers.jsonl",
        "execution_ids.json",
        "logits_A1.npy",
        "logits_A2.npy",
        "logits_B1.npy",
        "logits_B2.npy",
        "logits_G.npy",
        "native_lifecycle.jsonl",
        "protocol.md",
        "provenance.json",
        "result.json",
        "source_state/dagkv.tracked.patch",
        "source_state/dagkv.untracked.tar",
        "source_state/vllm.tracked.patch",
        "source_state/vllm.untracked.tar",
    }
)
POST_CHECKSUM_FORMAL_ARTIFACTS = frozenset(
    {
        "M2_ITEM8_FORMAL_RUN_MANIFEST.json",
        "M2_ITEM8_ACCEPTANCE_MANIFEST.json",
        "M2_ACCEPTANCE_MANIFEST.json",
    }
)
RESULT_FIELDS = frozenset(
    {
        "artifacts",
        "comparisons",
        "completed_at_utc",
        "diagnostic_bytes",
        "formal_run_passed",
        "gate_status",
        "m2_accepted",
        "m2_item8_accepted",
        "measurements",
        "minimum_top1_margin",
        "mode",
        "native_bytes",
        "prefetch",
        "reproducibility_fingerprint",
        "run_id",
        "schema_version",
        "tolerance",
        "transfer_digests",
        "within_requested_tolerance",
    }
)
PROVENANCE_FIELDS = frozenset(
    {
        "argv",
        "block_size",
        "calibration_cohort",
        "connector_config",
        "cpu_bytes",
        "dagkv_git",
        "dependencies",
        "engine_config",
        "executable",
        "frozen_tolerance",
        "full_provenance",
        "implementation",
        "mode",
        "model",
        "nvidia_driver_userspace",
        "postflight",
        "preflight",
        "prompt_token_ids",
        "python",
        "reproducibility_components",
        "reproducibility_fingerprint",
        "run_id",
        "runtime_binaries",
        "schema_version",
        "started_at_utc",
        "system",
        "tolerance",
        "vllm_git",
    }
)
ENGINE_CONFIG_FIELDS = frozenset(
    {
        "async_scheduling",
        "attention_config",
        "block_size",
        "data_parallel_size",
        "disable_hybrid_kv_cache_manager",
        "dtype",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "enforce_eager",
        "gpu_memory_utilization",
        "logprobs_mode",
        "max_logprobs",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "model",
        "pipeline_parallel_size",
        "scheduling_policy",
        "seed",
        "tensor_parallel_size",
        "trust_remote_code",
    }
)
COMPONENT_FIELDS = frozenset(
    {
        "block_size",
        "connector_config",
        "cpu_bytes",
        "dependency_manifest_sha256",
        "engine_config",
        "implementation_manifest_sha256",
        "model_manifest_sha256",
        "nvidia_driver_userspace_content_digest",
        "prompt_token_ids",
        "runtime_binary_manifest_sha256",
        "system",
        "vllm_snapshot_sha256",
    }
)
NVIDIA_CAPTURE_FIELDS = frozenset(
    {
        "root",
        "expected_manifest_sha256",
        "expected_content_digest",
        "expected_driver_version",
        "manifest",
        "manifest_sha256",
        "content_digest",
        "kernel_module_version",
        "runtime",
        "libcuda_mapping",
    }
)
NVIDIA_RUNTIME_FIELDS = frozenset(
    {"rootfs", "library_directory", "nvidia_smi", "libcuda", "libnvidia_ml"}
)
LIBCUDA_MAPPING_FIELDS = frozenset(
    {
        "path",
        "resolved_path",
        "rootfs_relative_path",
        "device",
        "inode",
        "size",
        "sha256",
        "mapping_count",
    }
)


class M2RawReplayError(ValueError):
    """Raised when raw evidence cannot reproduce a reported M2 result."""


@dataclass(frozen=True, slots=True)
class RawReplayValidation:
    run_id: str
    mode: str
    observed_max_abs_error: float
    minimum_top1_margin: float
    reproducibility_fingerprint: str
    implementation_manifest_sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M2RawReplayError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise M2RawReplayError(f"failed to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M2RawReplayError(f"cannot canonicalize evidence: {exc}") from exc
    return _sha256_bytes(encoded)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise M2RawReplayError(f"non-finite JSON constant: {value}")


def _parse_json(raw: str, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except M2RawReplayError:
        raise
    except json.JSONDecodeError as exc:
        raise M2RawReplayError(f"invalid {label}: {exc}") from exc


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise M2RawReplayError(f"invalid {label} at {path}: {exc}") from exc
    payload = _parse_json(raw, label=f"{label} at {path}")
    require(isinstance(payload, dict), f"{label} must be an object: {path}")
    return payload


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise M2RawReplayError(f"invalid {label} at {path}: {exc}") from exc
    require(raw.endswith(b"\n"), f"unterminated {label}: {path}")
    require(b"\r" not in raw, f"{label} must use LF newlines: {path}")
    lines = text[:-1].split("\n")
    require(lines and all(lines), f"{label} contains a blank row: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        row = _parse_json(line, label=f"{label} row {path}:{line_number}")
        require(
            isinstance(row, dict),
            f"{label} row must be an object at {path}:{line_number}",
        )
        rows.append(row)
    return rows


def _lower_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    require(isinstance(value, str) and value != "", f"{label} must be non-empty")
    return value


def _driver_version(value: Any, *, label: str) -> str:
    text = _nonempty_string(value, label=label)
    require(
        re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,}", text) is not None,
        f"{label} must be a dotted numeric NVIDIA driver version",
    )
    return text


def _finite_number(value: Any, *, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    lower_bound = 0 if allow_zero else 1
    require(
        type(value) is int and value >= lower_bound,
        f"{label} must be an integer >= {lower_bound}",
    )
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    text = _nonempty_string(value, label=label)
    require("\\" not in text, f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    require(
        not path.is_absolute()
        and path.as_posix() == text
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is unsafe: {text}",
    )
    return text


def _resolved_absolute_path(value: Any, *, label: str) -> Path:
    path = Path(_nonempty_string(value, label=label))
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise M2RawReplayError(f"cannot resolve {label}: {exc}") from exc


def _timestamp(value: Any, *, label: str) -> datetime:
    text = _nonempty_string(value, label=label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise M2RawReplayError(f"{label} must be ISO 8601") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _argv_option(argv: list[str], name: str) -> str:
    values: list[str] = []
    for index, value in enumerate(argv):
        if value == name:
            require(index + 1 < len(argv), f"argv option lacks a value: {name}")
            values.append(argv[index + 1])
        elif value.startswith(f"{name}="):
            values.append(value.split("=", maxsplit=1)[1])
    require(len(values) == 1, f"argv must contain exactly one {name}")
    return _nonempty_string(values[0], label=f"argv {name}")


def _manifest_file_digest(payload: dict[str, Any]) -> str:
    try:
        encoded = (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M2RawReplayError(f"cannot canonicalize NVIDIA manifest: {exc}") from exc
    return _sha256_bytes(encoded)


def _runtime_symlink_target(path: str, target: str) -> str:
    target_path = PurePosixPath(target)
    require(not target_path.is_absolute(), "absolute NVIDIA runtime symlink target")
    parts = list(PurePosixPath(path).parent.parts)
    for part in target_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            require(parts, "NVIDIA runtime symlink target escapes rootfs")
            parts.pop()
            continue
        parts.append(part)
    require(parts, "NVIDIA runtime symlink resolves to rootfs")
    return PurePosixPath(*parts).as_posix()


def _validate_runtime_topology(records: dict[str, dict[str, Any]]) -> None:
    for path, record in records.items():
        if path == ".":
            continue
        parent = PurePosixPath(path).parent.as_posix()
        if parent == ".":
            parent = "."
        require(
            parent in records and records[parent]["type"] == "directory",
            f"NVIDIA runtime parent directory is missing: {path}",
        )
        if record["type"] != "symlink":
            continue
        current = path
        visited: set[str] = set()
        while records[current]["type"] == "symlink":
            require(current not in visited, "cyclic NVIDIA runtime symlink")
            visited.add(current)
            current = _runtime_symlink_target(
                current,
                records[current]["target"],
            )
            require(current in records, "dangling NVIDIA runtime symlink")
        require(
            records[current]["type"] == "file",
            "NVIDIA runtime symlink does not resolve to a regular file",
        )


def _validate_nvidia_manifest(
    manifest: Any,
) -> tuple[str, str, dict[str, dict[str, Any]], datetime]:
    require(isinstance(manifest, dict), "NVIDIA bundle manifest must be an object")
    require(
        set(manifest) == NVIDIA_MANIFEST_FIELDS,
        "NVIDIA bundle manifest fields differ",
    )
    require(
        manifest["schema_version"] == NVIDIA_MANIFEST_SCHEMA,
        "NVIDIA bundle manifest schema differs",
    )
    require(
        manifest["bundle_type"] == "nvidia_debian_userspace_exact",
        "NVIDIA bundle type differs",
    )
    created = _timestamp(
        manifest["created_at_utc"], label="NVIDIA bundle created_at_utc"
    )
    kernel_version = _driver_version(
        manifest["kernel_module_version"],
        label="NVIDIA bundle kernel module version",
    )

    packages = manifest["packages"]
    require(isinstance(packages, list) and packages, "NVIDIA package list is empty")
    package_paths: list[str] = []
    package_names: set[str] = set()
    for item in packages:
        require(
            isinstance(item, dict) and set(item) == NVIDIA_PACKAGE_FIELDS,
            "NVIDIA package record fields differ",
        )
        path = _safe_relative(item["path"], label="NVIDIA package path")
        require(Path(path).name == path and path.endswith(".deb"), "unsafe .deb path")
        package_paths.append(path)
        name = _nonempty_string(item["package"], label="NVIDIA package name")
        require(
            NVIDIA_PACKAGE_TOKEN_RE.fullmatch(name) is not None,
            "invalid NVIDIA package name",
        )
        require(name not in package_names, "duplicate NVIDIA package name")
        package_names.add(name)
        version = _nonempty_string(item["version"], label="NVIDIA package version")
        require(
            version.isprintable()
            and not any(character.isspace() for character in version),
            "invalid NVIDIA package version",
        )
        driver_version = version.split(":", maxsplit=1)[-1].split("-", maxsplit=1)[0]
        require(
            driver_version == kernel_version,
            "NVIDIA package and kernel module versions differ",
        )
        architecture = _nonempty_string(
            item["architecture"], label="NVIDIA package architecture"
        )
        require(
            NVIDIA_PACKAGE_TOKEN_RE.fullmatch(architecture) is not None,
            "invalid NVIDIA package architecture",
        )
        mode = _positive_int(item["mode"], label="NVIDIA package mode", allow_zero=True)
        require(
            mode <= 0o7777 and mode & 0o222 == 0,
            "writable or invalid NVIDIA package mode",
        )
        _positive_int(item["size"], label="NVIDIA package size")
        _lower_sha256(item["sha256"], label="NVIDIA package SHA")
    require(
        package_paths == sorted(set(package_paths)),
        "NVIDIA packages are not strictly sorted",
    )
    require(
        _positive_int(manifest["package_count"], label="NVIDIA package count")
        == len(package_paths),
        "NVIDIA package count differs",
    )
    require(
        any(name.startswith("libnvidia-compute-") for name in package_names)
        and any(name.startswith("nvidia-utils-") for name in package_names),
        "NVIDIA bundle package roles are incomplete",
    )

    runtime_tree = manifest["runtime_tree"]
    require(
        isinstance(runtime_tree, list) and runtime_tree,
        "NVIDIA runtime tree is empty",
    )
    runtime_paths: list[str] = []
    runtime_records: dict[str, dict[str, Any]] = {}
    for item in runtime_tree:
        require(isinstance(item, dict), "NVIDIA runtime entry must be an object")
        kind = item.get("type")
        fields = {
            "directory": NVIDIA_RUNTIME_DIRECTORY_FIELDS,
            "file": NVIDIA_RUNTIME_FILE_FIELDS,
            "symlink": NVIDIA_RUNTIME_SYMLINK_FIELDS,
        }.get(kind)
        require(
            fields is not None and set(item) == fields,
            "NVIDIA runtime fields differ",
        )
        path = item.get("path")
        require(
            path == "." or _safe_relative(path, label="NVIDIA runtime path") == path,
            "NVIDIA runtime path differs",
        )
        require(path not in runtime_records, "duplicate NVIDIA runtime path")
        runtime_paths.append(path)
        runtime_records[path] = item
        if kind in {"directory", "file"}:
            mode = _positive_int(
                item["mode"], label="NVIDIA runtime mode", allow_zero=True
            )
            require(mode <= 0o7777 and mode & 0o222 == 0, "writable NVIDIA runtime")
        if kind == "file":
            _positive_int(
                item["size"],
                label="NVIDIA runtime file size",
                allow_zero=True,
            )
            _lower_sha256(item["sha256"], label="NVIDIA runtime file SHA")
        elif kind == "symlink":
            _nonempty_string(item["target"], label="NVIDIA runtime symlink target")
    require(
        runtime_paths[0] == "." and runtime_records["."]["type"] == "directory",
        "NVIDIA runtime root entry is missing",
    )
    _validate_runtime_topology(runtime_records)
    ordering = [
        () if path == "." else PurePosixPath(path).parts for path in runtime_paths
    ]
    require(ordering == sorted(ordering), "NVIDIA runtime tree is not sorted")
    require(
        _positive_int(
            manifest["runtime_entry_count"], label="NVIDIA runtime entry count"
        )
        == len(runtime_paths),
        "NVIDIA runtime entry count differs",
    )
    require(
        manifest["content_digest_algorithm"] == "sha256",
        "NVIDIA content digest algorithm differs",
    )
    require(
        manifest["runtime_derivation"] == RUNTIME_DERIVATION,
        "NVIDIA runtime derivation differs",
    )
    content_digest = _lower_sha256(
        manifest["content_digest"], label="NVIDIA content digest"
    )
    reconstructed = _canonical_digest(
        {
            "domain": CONTENT_DIGEST_DOMAIN,
            "kernel_module_version": kernel_version,
            "packages": packages,
            "runtime_derivation": manifest["runtime_derivation"],
            "runtime_tree": runtime_tree,
        }
    )
    require(content_digest == reconstructed, "NVIDIA content digest differs")
    return content_digest, kernel_version, runtime_records, created


def _validate_nvidia_driver_userspace(
    capture: Any,
    *,
    argv: list[str],
    system: dict[str, Any],
    started: datetime,
) -> str:
    require(
        isinstance(capture, dict) and set(capture) == NVIDIA_CAPTURE_FIELDS,
        "NVIDIA userspace capture fields differ",
    )
    root = _resolved_absolute_path(capture["root"], label="NVIDIA bundle root")
    content_digest, kernel_version, runtime_records, created = (
        _validate_nvidia_manifest(capture["manifest"])
    )
    require(created <= started, "NVIDIA bundle was created after runner startup")
    manifest_sha256 = _lower_sha256(
        capture["manifest_sha256"], label="NVIDIA manifest SHA"
    )
    expected_manifest_sha256 = _lower_sha256(
        capture["expected_manifest_sha256"],
        label="expected NVIDIA manifest SHA",
    )
    require(
        expected_manifest_sha256
        == manifest_sha256
        == _manifest_file_digest(capture["manifest"]),
        "NVIDIA manifest SHA differs",
    )
    expected_digest = _lower_sha256(
        capture["expected_content_digest"], label="expected NVIDIA content digest"
    )
    require(
        expected_digest == content_digest == capture["content_digest"],
        "NVIDIA capture content digests differ",
    )
    require(
        _driver_version(
            capture["expected_driver_version"],
            label="expected NVIDIA driver version",
        )
        == kernel_version,
        "NVIDIA expected driver version differs",
    )
    require(
        capture["kernel_module_version"] == kernel_version,
        "NVIDIA capture kernel module version differs",
    )

    runtime = capture["runtime"]
    require(
        isinstance(runtime, dict) and set(runtime) == NVIDIA_RUNTIME_FIELDS,
        "NVIDIA runtime mapping fields differ",
    )
    rootfs = root / "rootfs"
    library = rootfs / "usr/lib/x86_64-linux-gnu"
    nvidia_smi = rootfs / "usr/bin/nvidia-smi"
    libcuda = library / f"libcuda.so.{kernel_version}"
    libnvidia_ml = library / f"libnvidia-ml.so.{kernel_version}"
    require(
        runtime
        == {
            "rootfs": str(rootfs),
            "library_directory": str(library),
            "nvidia_smi": str(nvidia_smi),
            "libcuda": str(libcuda),
            "libnvidia_ml": str(libnvidia_ml),
        },
        "NVIDIA runtime absolute paths differ",
    )
    required_runtime = {
        "usr/bin/nvidia-smi": "file",
        f"usr/lib/x86_64-linux-gnu/libcuda.so.{kernel_version}": "file",
        "usr/lib/x86_64-linux-gnu/libcuda.so.1": "symlink",
        "usr/lib/x86_64-linux-gnu/libcuda.so": "symlink",
        f"usr/lib/x86_64-linux-gnu/libnvidia-ml.so.{kernel_version}": "file",
        "usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1": "symlink",
    }
    for path, kind in required_runtime.items():
        require(
            path in runtime_records and runtime_records[path]["type"] == kind,
            f"NVIDIA runtime binding is missing: {path}",
        )
    require(
        runtime_records["usr/lib/x86_64-linux-gnu/libcuda.so.1"]["target"]
        == libcuda.name,
        "libcuda.so.1 target differs",
    )
    require(
        runtime_records["usr/lib/x86_64-linux-gnu/libcuda.so"]["target"]
        == "libcuda.so.1",
        "unversioned libcuda target differs",
    )
    require(
        runtime_records["usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"]["target"]
        == libnvidia_ml.name,
        "libnvidia-ml.so.1 target differs",
    )

    mapping = capture["libcuda_mapping"]
    require(
        isinstance(mapping, dict) and set(mapping) == LIBCUDA_MAPPING_FIELDS,
        "libcuda mapping fields differ",
    )
    require(
        mapping["path"] == str(libcuda)
        and mapping["resolved_path"] == str(libcuda)
        and mapping["rootfs_relative_path"] == libcuda.relative_to(rootfs).as_posix(),
        "mapped libcuda path differs from the bundle",
    )
    libcuda_record = runtime_records[mapping["rootfs_relative_path"]]
    for field in ("device", "inode", "mapping_count"):
        _positive_int(mapping[field], label=f"libcuda mapping {field}")
    require(
        _positive_int(mapping["size"], label="libcuda mapping size")
        == libcuda_record["size"],
        "mapped libcuda size differs",
    )
    require(
        _lower_sha256(mapping["sha256"], label="mapped libcuda SHA")
        == libcuda_record["sha256"],
        "mapped libcuda SHA differs",
    )

    require(
        _resolved_absolute_path(
            _argv_option(argv, "--nvidia-userspace-bundle-root"),
            label="argv NVIDIA bundle root",
        )
        == root,
        "argv NVIDIA bundle root differs",
    )
    require(
        _argv_option(argv, "--expected-nvidia-userspace-bundle-manifest-sha256")
        == expected_manifest_sha256,
        "argv NVIDIA manifest SHA differs",
    )
    require(
        _argv_option(argv, "--expected-nvidia-userspace-bundle-content-digest")
        == content_digest,
        "argv NVIDIA content digest differs",
    )
    require(
        _argv_option(argv, "--expected-nvidia-driver-version") == kernel_version,
        "argv NVIDIA driver version differs",
    )
    gpu = system.get("gpu")
    require(isinstance(gpu, dict), "system GPU provenance is missing")
    require(
        gpu.get("nvidia_smi_executable") == str(nvidia_smi),
        "system nvidia-smi executable differs",
    )
    smi = _nonempty_string(gpu.get("nvidia_smi"), label="system nvidia-smi output")
    smi_fields = [field.strip() for field in smi.split(",")]
    require(len(smi_fields) == 5, "system nvidia-smi identity fields differ")
    require(
        _driver_version(smi_fields[1], label="system nvidia-smi driver version")
        == kernel_version,
        "system nvidia-smi driver version differs",
    )
    environment = system.get("environment")
    require(isinstance(environment, dict), "system environment provenance is missing")
    require(
        environment.get("LD_LIBRARY_PATH", "").split(":", maxsplit=1)[0]
        == str(library),
        "captured LD_LIBRARY_PATH does not begin with the NVIDIA bundle",
    )
    require(
        all(name not in environment for name in ("LD_AUDIT", "LD_PRELOAD")),
        "captured loader injection environment is forbidden",
    )
    return content_digest


def _validate_run_tree(run_dir: Path) -> None:
    allowed_directories = {"source_state"}
    observed_directories: set[str] = set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise M2RawReplayError(
                f"cannot inspect run artifact {path}: {exc}"
            ) from exc
        require(
            not stat.S_ISLNK(metadata.st_mode),
            f"run artifact cannot be a symlink: {path}",
        )
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
            require(
                relative in allowed_directories,
                f"run directory layout contains an unexpected directory: {relative}",
            )
            continue
        require(
            stat.S_ISREG(metadata.st_mode),
            f"run artifact must be a regular file: {path}",
        )
        require(
            metadata.st_nlink == 1,
            f"run artifact must have exactly one hard link: {path}",
        )
        parent = PurePosixPath(relative).parent.as_posix()
        require(
            parent in {".", "source_state"},
            f"run artifact is outside the frozen directory layout: {relative}",
        )
    require(
        observed_directories == allowed_directories,
        "run directory layout must contain exactly source_state/",
    )


def _validate_sha256sums(run_dir: Path) -> dict[str, str]:
    _validate_run_tree(run_dir)
    checksum_path = run_dir / "SHA256SUMS"
    require(
        checksum_path.is_file() and not checksum_path.is_symlink(),
        f"missing SHA256SUMS: {run_dir}",
    )
    try:
        raw = checksum_path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise M2RawReplayError(f"invalid SHA256SUMS at {checksum_path}: {exc}") from exc
    require(raw.endswith(b"\n"), f"unterminated SHA256SUMS: {checksum_path}")
    require(b"\r" not in raw, f"SHA256SUMS must use LF: {checksum_path}")
    lines = text[:-1].split("\n")
    require(lines and all(lines), f"blank SHA256SUMS row: {checksum_path}")
    entries: dict[str, str] = {}
    ordered: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        require(
            len(line) > 66 and line[64:66] == "  ",
            f"malformed SHA256SUMS row {checksum_path}:{line_number}",
        )
        digest = _lower_sha256(line[:64], label=f"SHA256SUMS digest at {line_number}")
        name = _safe_relative(line[66:], label=f"SHA256SUMS path at {line_number}")
        require(name != "SHA256SUMS", "SHA256SUMS cannot list itself")
        require(name not in entries, f"duplicate SHA256SUMS path: {name}")
        entries[name] = digest
        ordered.append(name)
    require(ordered == sorted(ordered), "SHA256SUMS paths must be sorted")
    require(
        set(entries) == EXPECTED_ARTIFACTS,
        "M2 checksum artifact closed set differs: "
        f"missing={sorted(EXPECTED_ARTIFACTS - set(entries))}, "
        f"extra={sorted(set(entries) - EXPECTED_ARTIFACTS)}",
    )

    actual: set[str] = set()
    for path in run_dir.rglob("*"):
        require(not path.is_symlink(), f"run artifact cannot be a symlink: {path}")
        if path.is_file() and path != checksum_path:
            name = path.relative_to(run_dir).as_posix()
            if name not in POST_CHECKSUM_FORMAL_ARTIFACTS:
                actual.add(name)
    require(
        actual == EXPECTED_ARTIFACTS,
        "run artifact closed set differs from SHA256SUMS: "
        f"missing={sorted(EXPECTED_ARTIFACTS - actual)}, "
        f"extra={sorted(actual - EXPECTED_ARTIFACTS)}",
    )
    for name, expected in entries.items():
        observed = _sha256_file(run_dir / PurePosixPath(name))
        require(observed == expected, f"checksum mismatch for {name}")
    return entries


def _numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise M2RawReplayError(
            "NumPy is required to replay raw M2 logits; validation fails closed"
        ) from exc


def _load_logits(path: Path) -> Any:
    np = _numpy()
    require(path.is_file() and not path.is_symlink(), f"missing logits: {path}")
    try:
        logits = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise M2RawReplayError(f"invalid NPY logits at {path}: {exc}") from exc
    require(isinstance(logits, np.ndarray), f"logits must be an NPY array: {path}")
    require(logits.dtype == np.dtype(np.float64), f"logits must be float64: {path}")
    require(logits.ndim == 1, f"logits must be a 1-D vector: {path}")
    require(
        logits.size == FROZEN_QWEN3_8B_VOCAB_SIZE,
        "logits must contain the frozen Qwen3-8B vocabulary of "
        f"{FROZEN_QWEN3_8B_VOCAB_SIZE} entries: {path}",
    )
    require(logits.flags.c_contiguous, f"logits must be C-contiguous: {path}")
    require(
        bool(np.isfinite(logits).all()), f"logits contain non-finite values: {path}"
    )
    return logits


def _reported_float(value: Any, observed: float, *, label: str) -> None:
    reported = _finite_number(value, label=label)
    require(reported == observed, f"{label} differs from raw replay")


def _validate_result_and_logits(
    run_dir: Path,
    result: dict[str, Any],
    entries: dict[str, str],
) -> tuple[dict[str, Any], float, float]:
    np = _numpy()
    require(set(result) == RESULT_FIELDS, "result.json fields differ from M2 v3")
    require(result.get("schema_version") == PROTOCOL_SCHEMA, "wrong result schema")
    run_id = _nonempty_string(result.get("run_id"), label="result run_id")
    mode = result.get("mode")
    require(mode in {"calibration", "formal"}, "result mode is invalid")
    if mode == "calibration":
        require(
            result.get("gate_status") == "CALIBRATED_NOT_ACCEPTED",
            "calibration gate status is invalid",
        )
        require(result.get("formal_run_passed") is False, "calibration claims a pass")
    else:
        require(
            result.get("gate_status") == "M2_ITEM8_FORMAL_HOLDOUT_PASSED",
            "formal gate status is invalid",
        )
        require(result.get("formal_run_passed") is True, "formal run did not pass")
    require(result.get("m2_accepted") is False, "single run claims M2 acceptance")
    require(
        result.get("m2_item8_accepted") is False,
        "single run claims aggregate item 8 acceptance",
    )

    tolerance = result.get("tolerance")
    require(
        isinstance(tolerance, dict) and set(tolerance) == {"atol", "rtol"},
        "result tolerance fields are invalid",
    )
    atol = _finite_number(tolerance["atol"], label="result atol")
    rtol = _finite_number(tolerance["rtol"], label="result rtol")
    require(0.0 <= atol <= 0.125, "result atol is outside the M2 cap")
    require(rtol == 0.0, "result rtol must be zero")

    measurements = result.get("measurements")
    require(isinstance(measurements, dict), "result measurements are missing")
    require(set(measurements) == set(EXPECTED_PHASES), "measurement phases differ")
    logits_by_phase: dict[str, Any] = {}
    token_ids: dict[str, int] = {}
    margins: dict[str, float] = {}
    shape: tuple[int, ...] | None = None
    for phase in EXPECTED_PHASES:
        measurement = measurements[phase]
        require(isinstance(measurement, dict), f"{phase} measurement is invalid")
        require(
            set(measurement)
            == {
                "elapsed_ms",
                "logits_file",
                "logits_sha256",
                "num_cached_tokens",
                "request_id",
                "token_id",
                "top1_margin",
                "trace_id",
            },
            f"{phase} measurement fields differ",
        )
        _nonempty_string(measurement["request_id"], label=f"{phase} request_id")
        require(
            measurement["trace_id"] == f"{run_id}:{phase}:measurement",
            f"{phase} trace_id differs",
        )
        expected_cached = 0 if phase in {"A1", "A2"} else 16
        require(
            measurement["num_cached_tokens"] == expected_cached,
            f"{phase} cached-token count differs",
        )
        require(
            _finite_number(measurement["elapsed_ms"], label=f"{phase} elapsed_ms") > 0,
            f"{phase} elapsed_ms must be positive",
        )
        expected_name = f"logits_{phase}.npy"
        require(
            measurement["logits_file"] == expected_name, f"{phase} logits name differs"
        )
        expected_sha = _lower_sha256(
            measurement["logits_sha256"], label=f"{phase} logits_sha256"
        )
        require(expected_sha == entries[expected_name], f"{phase} logits SHA differs")
        logits = _load_logits(run_dir / expected_name)
        if shape is None:
            shape = logits.shape
        require(logits.shape == shape, f"{phase} logits shape differs")
        token_id = int(np.argmax(logits))
        maximum = float(logits[token_id])
        second = float(np.partition(logits, -2)[-2])
        margin = maximum - second
        require(margin > 0.25, f"{phase} raw top-1 margin is not above 0.25")
        require(
            measurement["token_id"] == token_id, f"{phase} token_id differs from logits"
        )
        _reported_float(
            measurement["top1_margin"], margin, label=f"{phase} top1_margin"
        )
        logits_by_phase[phase] = logits
        token_ids[phase] = token_id
        margins[phase] = margin
    require(len(set(token_ids.values())) == 1, "raw logits decode different tokens")

    reported_comparisons = result.get("comparisons")
    require(isinstance(reported_comparisons, list), "result comparisons are missing")
    require(
        len(reported_comparisons) == len(EXPECTED_PAIRS), "comparison count differs"
    )
    observed_maximum = 0.0
    allclose_values: list[bool] = []
    for expected_pair, row in zip(EXPECTED_PAIRS, reported_comparisons, strict=True):
        require(isinstance(row, dict), "comparison row must be an object")
        require(
            set(row)
            == {
                "allclose",
                "left",
                "max_abs_error",
                "max_rel_error",
                "right",
                "token_equal",
            },
            f"comparison fields differ for {expected_pair}",
        )
        left_name, right_name = expected_pair
        require(
            (row["left"], row["right"]) == expected_pair,
            f"comparison order differs for {expected_pair}",
        )
        left = logits_by_phase[left_name]
        right = logits_by_phase[right_name]
        absolute = np.abs(left - right)
        denominator = np.maximum(np.abs(left), np.abs(right))
        relative = np.divide(
            absolute,
            denominator,
            out=np.zeros_like(absolute),
            where=denominator != 0,
        )
        max_abs = float(absolute.max(initial=0.0))
        max_rel = float(relative.max(initial=0.0))
        token_equal = token_ids[left_name] == token_ids[right_name]
        allclose = bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
        require(
            row["token_equal"] is token_equal,
            f"token_equal differs for {expected_pair}",
        )
        require(row["allclose"] is allclose, f"allclose differs for {expected_pair}")
        _reported_float(row["max_abs_error"], max_abs, label=f"{expected_pair} max_abs")
        _reported_float(row["max_rel_error"], max_rel, label=f"{expected_pair} max_rel")
        if expected_pair in EXACT_PAIRS:
            require(max_abs == 0.0, f"exact logits pair drifted: {expected_pair}")
        observed_maximum = max(observed_maximum, max_abs)
        allclose_values.append(allclose)
    within = all(allclose_values)
    require(
        result.get("within_requested_tolerance") is within,
        "within_requested_tolerance differs from raw replay",
    )
    if mode == "formal":
        require(within, "formal logits exceed the frozen tolerance")
    minimum_margin = min(margins.values())
    _reported_float(
        result.get("minimum_top1_margin"),
        minimum_margin,
        label="minimum_top1_margin",
    )
    return logits_by_phase, observed_maximum, minimum_margin


def _validate_execution_ids(
    execution_ids: dict[str, Any], *, result: dict[str, Any]
) -> dict[str, dict[str, str]]:
    run_id = result["run_id"]
    require(
        set(execution_ids) == {"measurements", "run_id", "schema_version", "transfers"},
        "execution_ids.json fields differ",
    )
    require(
        execution_ids["schema_version"] == PROTOCOL_SCHEMA, "wrong execution schema"
    )
    require(execution_ids["run_id"] == run_id, "execution run_id differs")
    measurements = execution_ids["measurements"]
    require(isinstance(measurements, dict), "execution measurements are invalid")
    require(set(measurements) == set(EXPECTED_PHASES), "execution phases differ")
    api_ids: set[str] = set()
    for phase in EXPECTED_PHASES:
        item = measurements[phase]
        require(
            isinstance(item, dict) and set(item) == {"request_id", "trace_id"},
            f"execution {phase} fields differ",
        )
        require(
            item
            == {
                "request_id": result["measurements"][phase]["request_id"],
                "trace_id": f"{run_id}:{phase}:measurement",
            },
            f"execution {phase} differs from result",
        )
        request_id = _nonempty_string(
            item["request_id"], label=f"{phase} API request ID"
        )
        require(request_id not in api_ids, "measurement API request IDs must be unique")
        api_ids.add(request_id)

    transfers = execution_ids["transfers"]
    require(isinstance(transfers, dict), "execution transfers are invalid")
    require(set(transfers) == set(TRANSFER_SPECS), "execution transfer set differs")
    normalized: dict[str, dict[str, str]] = {}
    engine_ids: set[str] = set()
    for label, (phase, direction, role) in TRANSFER_SPECS.items():
        item = transfers[label]
        require(
            isinstance(item, dict)
            and set(item) == {"engine_request_id", "request_id", "trace_id"},
            f"execution {label} fields differ",
        )
        request_id = _nonempty_string(item["request_id"], label=f"{label} request_id")
        engine_id = _nonempty_string(
            item["engine_request_id"], label=f"{label} engine_request_id"
        )
        expected_trace = f"{run_id}:{phase}:{role}"
        require(item["trace_id"] == expected_trace, f"{label} trace_id differs")
        if direction == "D2H":
            require(
                engine_id.startswith(f"{request_id}-") and engine_id != request_id,
                f"{label} API and EngineCore request IDs are unrelated",
            )
        else:
            require(engine_id == request_id, f"{label} prefetch request IDs differ")
            require(
                request_id == result["prefetch"][phase]["request_id"],
                f"{label} request ID differs from prefetch result",
            )
        require(engine_id not in engine_ids, "transfer EngineCore IDs must be unique")
        engine_ids.add(engine_id)
        normalized[label] = {
            "request_id": request_id,
            "engine_request_id": engine_id,
            "trace_id": expected_trace,
        }
    return normalized


def _endpoint(value: Any, *, label: str, tier: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} endpoint is missing")
    require(
        set(value) == {"allocation_generation", "digest", "physical_slot", "tier"},
        f"{label} endpoint fields differ",
    )
    require(value["tier"] == tier, f"{label} tier must be {tier}")
    _positive_int(value["physical_slot"], label=f"{label} slot", allow_zero=True)
    _positive_int(value["allocation_generation"], label=f"{label} generation")
    _lower_sha256(value["digest"], label=f"{label} digest")
    return value


def _validate_diagnostic_trace(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    transfers: dict[str, dict[str, str]],
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    require(len(rows) == 8, "diagnostic trace must contain exactly eight rows")
    terminal_rows: dict[str, dict[str, Any]] = {}
    job_ids: set[int] = set()
    submitted_fields = {
        "capture_ns",
        "direction",
        "event",
        "framing",
        "job_id",
        "payload_bytes",
        "phase",
        "request_id",
        "run_id",
        "schema_version",
        "source",
        "status",
        "submit_ns",
        "target",
    }
    terminal_fields = {
        "capture_ns",
        "direction",
        "event",
        "failure_reason",
        "framing",
        "job_id",
        "payload_bytes",
        "phase",
        "reported_bytes",
        "request_id",
        "run_id",
        "schema_version",
        "source",
        "status",
        "target",
        "terminal_ns",
    }
    expected_transfers = {
        (transfers[label]["engine_request_id"], direction): label
        for label, (_phase, direction, _role) in TRANSFER_SPECS.items()
    }
    require(
        len(expected_transfers) == len(TRANSFER_SPECS),
        "diagnostic transfer identities must be unique",
    )
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        event = row.get("event")
        require(event in {"submitted", "terminal"}, "diagnostic event is invalid")
        fields = submitted_fields if event == "submitted" else terminal_fields
        require(set(row) == fields, f"diagnostic {event} fields differ")
        request_id = row["request_id"]
        direction = row["direction"]
        identity = (request_id, direction)
        require(
            identity in expected_transfers, "diagnostic transfer closed set differs"
        )
        events = indexed.setdefault(identity, {})
        require(event not in events, f"diagnostic transfer has duplicate {event}")
        events[event] = row
    require(
        set(indexed) == set(expected_transfers),
        "diagnostic transfer closed set differs",
    )

    for label, (_phase, direction, _role) in TRANSFER_SPECS.items():
        engine_id = transfers[label]["engine_request_id"]
        pair = indexed[(engine_id, direction)]
        require(
            set(pair) == {"submitted", "terminal"},
            f"{label} diagnostic event pair differs",
        )
        submitted = pair["submitted"]
        terminal = pair["terminal"]
        require(set(submitted) == submitted_fields, f"{label} submitted fields differ")
        require(set(terminal) == terminal_fields, f"{label} terminal fields differ")
        common = {
            "schema_version": DIAGNOSTIC_SCHEMA,
            "run_id": run_id,
            "phase": "ABBA",
            "request_id": engine_id,
            "direction": direction,
            "framing": "DAGKV_PAYLOAD_V1",
        }
        for key, expected in common.items():
            require(submitted[key] == expected, f"{label} submitted {key} differs")
            require(terminal[key] == expected, f"{label} terminal {key} differs")
        require(submitted["event"] == "submitted", f"{label} submit event differs")
        require(submitted["status"] == "in_flight", f"{label} submit status differs")
        require(submitted["target"] is None, f"{label} submitted target must be null")
        require(terminal["event"] == "terminal", f"{label} terminal event differs")
        require(terminal["status"] == "completed", f"{label} did not complete")
        require(terminal["failure_reason"] in {None, ""}, f"{label} has a failure")
        job_id = _positive_int(
            submitted["job_id"], label=f"{label} job_id", allow_zero=True
        )
        require(terminal["job_id"] == job_id, f"{label} job_id changed")
        require(job_id not in job_ids, "diagnostic job ID was reused")
        job_ids.add(job_id)
        payload_bytes = _positive_int(
            submitted["payload_bytes"], label=f"{label} payload_bytes"
        )
        require(terminal["payload_bytes"] == payload_bytes, f"{label} bytes changed")
        require(
            terminal["reported_bytes"] == payload_bytes,
            f"{label} reported bytes differ",
        )
        capture_ns = _positive_int(submitted["capture_ns"], label=f"{label} capture_ns")
        require(terminal["capture_ns"] == capture_ns, f"{label} capture_ns changed")
        submit_ns = _positive_int(submitted["submit_ns"], label=f"{label} submit_ns")
        terminal_ns = _positive_int(
            terminal["terminal_ns"], label=f"{label} terminal_ns"
        )
        require(submit_ns <= terminal_ns, f"{label} terminal predates submission")
        source_tier, target_tier = (
            ("GPU", "CPU") if direction == "D2H" else ("CPU", "GPU")
        )
        submitted_source = _endpoint(
            submitted["source"], label=f"{label} submitted source", tier=source_tier
        )
        source = _endpoint(
            terminal["source"], label=f"{label} source", tier=source_tier
        )
        target = _endpoint(
            terminal["target"], label=f"{label} target", tier=target_tier
        )
        require(submitted_source == source, f"{label} source identity changed")
        require(source["digest"] == target["digest"], f"{label} digest changed in DMA")
        terminal_rows[label] = terminal

    for phase in ("B1", "B2"):
        d2h = terminal_rows[f"{phase}_D2H"]
        h2d = terminal_rows[f"{phase}_H2D"]
        require(
            d2h["target"] == h2d["source"], f"{phase} CPU allocation was not replayed"
        )
        require(
            d2h["payload_bytes"] == h2d["payload_bytes"], f"{phase} byte count differs"
        )
        d2h_gpu = d2h["source"]
        h2d_gpu = h2d["target"]
        source_identity = (
            d2h_gpu["physical_slot"],
            d2h_gpu["allocation_generation"],
        )
        target_identity = (
            h2d_gpu["physical_slot"],
            h2d_gpu["allocation_generation"],
        )
        require(
            source_identity != target_identity, f"{phase} reused stale GPU identity"
        )
        if source_identity[0] == target_identity[0]:
            require(
                target_identity[1] > source_identity[1],
                f"{phase} GPU generation did not advance",
            )

    b1_d2h = terminal_rows["B1_D2H"]
    b2_d2h = terminal_rows["B2_D2H"]
    b1_h2d = terminal_rows["B1_H2D"]
    b2_h2d = terminal_rows["B2_H2D"]
    require(
        b1_d2h["source"]["digest"] == b2_d2h["source"]["digest"],
        "B1/B2 canonical KV digests differ",
    )
    for endpoint_name, left, right in (
        ("CPU", b1_d2h["target"], b2_d2h["target"]),
        ("producer GPU", b1_d2h["source"], b2_d2h["source"]),
        ("replay GPU", b1_h2d["target"], b2_h2d["target"]),
    ):
        left_identity = (left["physical_slot"], left["allocation_generation"])
        right_identity = (right["physical_slot"], right["allocation_generation"])
        require(
            left_identity != right_identity, f"B2 reused B1 {endpoint_name} identity"
        )

    expected_bytes = result.get("diagnostic_bytes")
    require(
        isinstance(expected_bytes, dict) and set(expected_bytes) == set(TRANSFER_SPECS),
        "result diagnostic_bytes fields differ",
    )
    for label, row in terminal_rows.items():
        require(
            expected_bytes[label] == row["payload_bytes"],
            f"{label} result bytes differ",
        )
    require(
        len(set(expected_bytes.values())) == 1, "ABBA diagnostic byte counts differ"
    )
    transfer_digests = result.get("transfer_digests")
    require(
        isinstance(transfer_digests, dict) and set(transfer_digests) == {"B1", "B2"},
        "result transfer_digests fields differ",
    )
    for phase in ("B1", "B2"):
        digest = _lower_sha256(transfer_digests[phase], label=f"{phase} result digest")
        require(
            digest == terminal_rows[f"{phase}_D2H"]["source"]["digest"],
            f"{phase} digest differs",
        )
    return terminal_rows


def _validate_lifecycle_block(
    row: dict[str, Any], *, label: str, endpoint: dict[str, Any], byte_count: int
) -> None:
    blocks = row.get("blocks")
    require(
        isinstance(blocks, list) and len(blocks) == 1, f"{label} must name one block"
    )
    block = blocks[0]
    require(
        isinstance(block, dict)
        and set(block)
        == {"allocation_generation", "block_id", "byte_count", "identity_kind"},
        f"{label} lifecycle block fields differ",
    )
    require(block["identity_kind"] == "physical_slot", f"{label} identity kind differs")
    require(
        block["block_id"] == f"cpu:{endpoint['physical_slot']}",
        f"{label} CPU slot differs",
    )
    require(
        block["allocation_generation"] == endpoint["allocation_generation"],
        f"{label} CPU generation differs",
    )
    require(block["byte_count"] == byte_count, f"{label} lifecycle block bytes differ")


def _validate_native_trace(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    transfers: dict[str, dict[str, str]],
    diagnostic: dict[str, dict[str, Any]],
    execution_ids: dict[str, Any],
    result: dict[str, Any],
) -> None:
    scheduler: dict[tuple[str, str], list[dict[str, Any]]] = {}
    lifecycle: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        trace_id = row.get("trace_id")
        require(
            isinstance(trace_id, str) and trace_id.startswith(f"{run_id}:"),
            "native row has an unrelated trace_id",
        )
        if "run_id" in row:
            require(row["run_id"] == run_id, "native row has an unrelated run_id")
        require(row.get("status") != "failed", "native trace contains a failed event")
        event = _nonempty_string(row.get("event"), label="native event")
        direction = {
            "store_scheduled": "D2H",
            "store_complete": "D2H",
            "load_scheduled": "H2D",
            "load_complete": "H2D",
        }.get(event)
        if direction is not None:
            scheduler.setdefault((trace_id, direction), []).append(row)
        if event == "kv_lifecycle" and row.get("action") in {"save", "prefetch"}:
            direction = "D2H" if row["action"] == "save" else "H2D"
            lifecycle.setdefault((trace_id, direction), []).append(row)

    expected = {
        (item["trace_id"], TRANSFER_SPECS[label][1])
        for label, item in transfers.items()
    }
    require(set(scheduler) == expected, "native scheduler transfer closed set differs")
    require(set(lifecycle) == expected, "native lifecycle transfer closed set differs")

    native_bytes = result.get("native_bytes")
    require(
        isinstance(native_bytes, dict) and set(native_bytes) == set(TRANSFER_SPECS),
        "result native_bytes fields differ",
    )
    for label, (phase, direction, _) in TRANSFER_SPECS.items():
        transfer = transfers[label]
        key = (transfer["trace_id"], direction)
        scheduler_pair = scheduler[key]
        lifecycle_pair = lifecycle[key]
        prefix = "store" if direction == "D2H" else "load"
        action = "save" if direction == "D2H" else "prefetch"
        require(
            len(scheduler_pair) == 2
            and {row["event"] for row in scheduler_pair}
            == {f"{prefix}_scheduled", f"{prefix}_complete"},
            f"{label} native scheduler pair differs",
        )
        require(
            len(lifecycle_pair) == 2
            and {row.get("status") for row in lifecycle_pair}
            == {"scheduled", "completed"},
            f"{label} lifecycle pair differs",
        )
        scheduler_by_event = {row["event"]: row for row in scheduler_pair}
        lifecycle_by_status = {row["status"]: row for row in lifecycle_pair}
        scheduled = scheduler_by_event[f"{prefix}_scheduled"]
        completed = scheduler_by_event[f"{prefix}_complete"]
        life_scheduled = lifecycle_by_status["scheduled"]
        life_completed = lifecycle_by_status["completed"]
        require(
            len(scheduler_by_event) == 2 and len(lifecycle_by_status) == 2,
            f"{label} native event identities differ",
        )
        engine_id = transfer["engine_request_id"]
        for row_name, row in (("scheduled", scheduled), ("complete", completed)):
            require(
                row.get("request_id") == engine_id,
                f"{label} {row_name} request ID differs",
            )
            require(
                row.get("job_id") == diagnostic[label]["job_id"],
                f"{label} native job ID differs",
            )
            require(
                _finite_number(row.get("time"), label=f"{label} {row_name} time") > 0,
                f"{label} native time must be positive",
            )
        require(
            scheduled["time"] <= completed["time"],
            f"{label} completion predates schedule",
        )
        byte_key = f"native_{prefix}_bytes"
        byte_count = _positive_int(scheduled.get(byte_key), label=f"{label} {byte_key}")
        require(completed.get(byte_key) == byte_count, f"{label} native bytes changed")
        require(
            native_bytes[label] == byte_count, f"{label} result native bytes differ"
        )
        require(
            byte_count == diagnostic[label]["payload_bytes"],
            f"{label} native/diagnostic bytes differ",
        )

        expected_source, expected_target = (
            ("gpu", "cpu") if direction == "D2H" else ("cpu", "gpu")
        )
        for _state, row in (
            ("scheduled", life_scheduled),
            ("completed", life_completed),
        ):
            require(
                row.get("schema_version") == LIFECYCLE_SCHEMA,
                f"{label} lifecycle schema differs",
            )
            require(
                row.get("event") == "kv_lifecycle", f"{label} lifecycle event differs"
            )
            require(row.get("action") == action, f"{label} lifecycle action differs")
            require(row.get("phase") == phase, f"{label} lifecycle phase differs")
            require(
                row.get("request_id") == engine_id, f"{label} lifecycle request differs"
            )
            require(
                row.get("accounting_trace_id") == transfer["trace_id"],
                f"{label} accounting trace differs",
            )
            require(
                row.get("workflow_id") == f"m2-abba:{run_id}",
                f"{label} workflow differs",
            )
            require(
                row.get("source") == "vllm.offloading.scheduler",
                f"{label} lifecycle source differs",
            )
            require(
                row.get("source_tier") == expected_source,
                f"{label} source tier differs",
            )
            require(
                row.get("target_tier") == expected_target,
                f"{label} target tier differs",
            )
            require(
                row.get("block_count") == 1, f"{label} lifecycle block count differs"
            )
            require(
                row.get("byte_count") == byte_count, f"{label} lifecycle bytes differ"
            )
        scheduled_event_id = _nonempty_string(
            life_scheduled.get("event_id"), label=f"{label} scheduled event_id"
        )
        require(
            life_scheduled.get("parent_event_id") is None,
            f"{label} schedule has a parent",
        )
        require(
            life_scheduled.get("observed_byte_count") is None,
            f"{label} schedule observed bytes",
        )
        require(
            life_completed.get("parent_event_id") == scheduled_event_id,
            f"{label} lifecycle parent differs",
        )
        require(
            life_completed.get("observed_byte_count") == byte_count,
            f"{label} lifecycle observed bytes differ",
        )
        require(
            life_completed.get("operation_id") == life_scheduled.get("operation_id"),
            f"{label} lifecycle operation differs",
        )
        cpu_endpoint = (
            diagnostic[label]["target"]
            if direction == "D2H"
            else diagnostic[label]["source"]
        )
        _validate_lifecycle_block(
            life_scheduled,
            label=f"{label} scheduled",
            endpoint=cpu_endpoint,
            byte_count=byte_count,
        )
        _validate_lifecycle_block(
            life_completed,
            label=f"{label} completed",
            endpoint=cpu_endpoint,
            byte_count=byte_count,
        )

    measurements = execution_ids["measurements"]
    for phase in EXPECTED_PHASES:
        trace_id = measurements[phase]["trace_id"]
        selected = [
            row
            for row in rows
            if row.get("trace_id") == trace_id and row.get("event") == "lookup"
        ]
        require(len(selected) == 1, f"{phase} must have exactly one lookup row")
        row = selected[0]
        request_id = _nonempty_string(
            row.get("request_id"), label=f"{phase} lookup request_id"
        )
        api_id = measurements[phase]["request_id"]
        require(
            request_id.startswith(f"{api_id}-") and request_id != api_id,
            f"{phase} API and EngineCore lookup IDs are unrelated",
        )
        expected_hits = 0 if phase in {"A1", "A2"} else 16
        require(
            row.get("gpu_hit_tokens") == expected_hits,
            f"{phase} GPU lookup hits differ",
        )
        require(
            row.get("native_load_hit_tokens") == 0,
            f"{phase} performed on-demand CPU load",
        )
        require(
            row.get("skip_reading_prefix_cache") is False,
            f"{phase} bypassed prefix cache",
        )

    for phase in ("B1", "B2"):
        producer = transfers[f"{phase}_D2H"]
        producer_lookup = [
            row
            for row in rows
            if row.get("trace_id") == producer["trace_id"]
            and row.get("event") == "lookup"
        ]
        require(len(producer_lookup) == 1, f"{phase} producer lookup count differs")
        require(
            producer_lookup[0].get("request_id") == producer["engine_request_id"],
            f"{phase} producer lookup ID differs",
        )
        require(
            producer_lookup[0].get("gpu_hit_tokens") == 0,
            f"{phase} producer was not cold",
        )
        require(
            producer_lookup[0].get("native_load_hit_tokens") == 0,
            f"{phase} producer loaded CPU KV",
        )
        prefetch = transfers[f"{phase}_H2D"]
        prefetch_lookup = [
            row
            for row in rows
            if row.get("trace_id") == prefetch["trace_id"]
            and row.get("event") == "lookup"
        ]
        require(len(prefetch_lookup) == 1, f"{phase} prefetch lookup count differs")
        row = prefetch_lookup[0]
        require(
            row.get("request_id") == prefetch["engine_request_id"],
            f"{phase} prefetch lookup ID differs",
        )
        require(row.get("gpu_hit_tokens") == 0, f"{phase} prefetch did not start cold")
        require(
            row.get("native_load_hit_tokens") == 16,
            f"{phase} prefetch did not load one block",
        )
        require(
            row.get("dag_preload_event") == "m2_explicit_prefetch",
            f"{phase} preload event differs",
        )
        require(
            row.get("dag_preload_sources") == ["m2_frozen_abba"],
            f"{phase} preload source differs",
        )


def _validate_prefetch_and_artifacts(result: dict[str, Any]) -> None:
    require(
        result.get("artifacts")
        == {
            "diagnostic_trace": "diagnostic_transfers.jsonl",
            "native_trace": "native_lifecycle.jsonl",
            "protocol": "protocol.md",
            "provenance": "provenance.json",
        },
        "result artifact map differs",
    )
    prefetch = result.get("prefetch")
    require(
        isinstance(prefetch, dict) and set(prefetch) == {"B1", "B2"},
        "prefetch fields differ",
    )
    expected_fields = {
        "allocated_block_ids",
        "completed",
        "external_hit_tokens",
        "loaded_tokens",
        "local_gpu_hit_tokens",
        "lookup_pending",
        "reason",
        "request_id",
        "started",
        "steps",
        "wall_ms",
    }
    for phase in ("B1", "B2"):
        item = prefetch[phase]
        require(
            isinstance(item, dict) and set(item) == expected_fields,
            f"{phase} prefetch fields differ",
        )
        require(
            item["started"] is True and item["completed"] is True,
            f"{phase} prefetch did not complete",
        )
        require(item["reason"] == "completed", f"{phase} prefetch reason differs")
        require(item["local_gpu_hit_tokens"] == 0, f"{phase} prefetch had GPU hits")
        require(
            item["external_hit_tokens"] == 16, f"{phase} external hit count differs"
        )
        require(item["loaded_tokens"] == 16, f"{phase} loaded-token count differs")
        require(item["lookup_pending"] is False, f"{phase} prefetch remained pending")
        _nonempty_string(item["request_id"], label=f"{phase} prefetch request_id")
        _positive_int(item["steps"], label=f"{phase} prefetch steps")
        require(
            _finite_number(item["wall_ms"], label=f"{phase} prefetch wall_ms") > 0,
            f"{phase} wall_ms must be positive",
        )
        blocks = item["allocated_block_ids"]
        require(
            isinstance(blocks, list)
            and len(blocks) == 1
            and isinstance(blocks[0], list)
            and len(blocks[0]) == 1,
            f"{phase} allocated block shape differs",
        )
        _positive_int(blocks[0][0], label=f"{phase} allocated block", allow_zero=True)


def _validate_file_manifest(
    capture: Any,
    *,
    label: str,
    list_key: str,
    content_keys: tuple[str, ...],
    expected_keys: set[str],
    require_hashes: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(isinstance(capture, dict), f"{label} capture is missing")
    items = capture.get(list_key)
    require(isinstance(items, list) and items, f"{label} file list is empty")
    paths: list[str] = []
    content: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        require(
            isinstance(item, dict) and set(item) == expected_keys,
            f"{label} file {index} fields differ",
        )
        path = _safe_relative(item["path"], label=f"{label} file path")
        paths.append(path)
        _positive_int(item["size"], label=f"{label} {path} size", allow_zero=True)
        if "mtime_ns" in expected_keys:
            _positive_int(item["mtime_ns"], label=f"{label} {path} mtime_ns")
        if "inode" in expected_keys:
            _positive_int(item["inode"], label=f"{label} {path} inode")
        if require_hashes:
            _lower_sha256(item.get("sha256"), label=f"{label} {path} sha256")
        elif item.get("sha256") is not None:
            _lower_sha256(item["sha256"], label=f"{label} {path} sha256")
        content.append({key: item[key] for key in content_keys})
    require(
        paths == sorted(paths) and len(paths) == len(set(paths)),
        f"{label} paths differ",
    )
    return capture, content


def _validate_git_capture(
    capture: Any,
    *,
    label: str,
    run_dir: Path,
    entries: dict[str, str],
) -> str:
    require(isinstance(capture, dict), f"{label} Git capture is missing")
    require(
        set(capture)
        == {
            "dirty",
            "head",
            "root",
            "snapshot_sha256",
            "status_short",
            "tracked_patch",
            "tracked_patch_sha256",
            "untracked_archive",
            "untracked_archive_sha256",
            "untracked_files",
        },
        f"{label} Git capture fields differ",
    )
    _resolved_absolute_path(capture["root"], label=f"{label} Git root")
    head = capture["head"]
    require(
        isinstance(head, str)
        and len(head) == 40
        and all(character in "0123456789abcdef" for character in head),
        f"{label} Git HEAD is invalid",
    )
    status = capture["status_short"]
    require(
        isinstance(status, list)
        and all(isinstance(row, str) and row for row in status)
        and status == sorted(status),
        f"{label} Git status is invalid",
    )
    require(type(capture["dirty"]) is bool, f"{label} dirty flag is invalid")
    require(capture["dirty"] is bool(status), f"{label} dirty flag differs from status")
    patch_name = f"source_state/{label}.tracked.patch"
    archive_name = f"source_state/{label}.untracked.tar"
    require(capture["tracked_patch"] == patch_name, f"{label} patch path differs")
    require(
        capture["untracked_archive"] == archive_name, f"{label} archive path differs"
    )
    patch_path = run_dir / patch_name
    archive_path = run_dir / archive_name
    patch = patch_path.read_bytes()
    patch_sha = _sha256_bytes(patch)
    require(capture["tracked_patch_sha256"] == patch_sha, f"{label} patch SHA differs")
    require(entries[patch_name] == patch_sha, f"{label} patch checksum differs")
    archive_sha = _sha256_file(archive_path)
    require(
        capture["untracked_archive_sha256"] == archive_sha,
        f"{label} archive SHA differs",
    )
    require(entries[archive_name] == archive_sha, f"{label} archive checksum differs")

    untracked = capture["untracked_files"]
    require(isinstance(untracked, list), f"{label} untracked list is invalid")
    names: list[str] = []
    for item in untracked:
        require(isinstance(item, dict), f"{label} untracked entry is invalid")
        entry_type = item.get("type")
        expected = {"path", "sha256", "size", "type"}
        if entry_type == "symlink":
            expected.add("target")
        require(set(item) == expected, f"{label} untracked fields differ")
        name = _safe_relative(item["path"], label=f"{label} untracked path")
        names.append(name)
        require(entry_type in {"file", "symlink"}, f"{label} untracked type differs")
        _positive_int(item["size"], label=f"{label} {name} size", allow_zero=True)
        _lower_sha256(item["sha256"], label=f"{label} {name} SHA")
        if entry_type == "symlink":
            _nonempty_string(item["target"], label=f"{label} {name} target")
    require(
        names == sorted(names) and len(names) == len(set(names)),
        f"{label} untracked order differs",
    )
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            require(
                [member.name for member in members] == names,
                f"{label} tar closed set differs",
            )
            for member, expected in zip(members, untracked, strict=True):
                if expected["type"] == "file":
                    require(
                        member.isfile(),
                        f"{label} tar member type differs: {member.name}",
                    )
                    stream = archive.extractfile(member)
                    require(stream is not None, f"{label} tar member cannot be read")
                    payload = stream.read()
                else:
                    require(
                        member.issym(),
                        f"{label} tar symlink type differs: {member.name}",
                    )
                    require(
                        member.linkname == expected["target"],
                        f"{label} tar link target differs",
                    )
                    payload = os.fsencode(member.linkname)
                require(
                    len(payload) == expected["size"], f"{label} tar member size differs"
                )
                require(
                    _sha256_bytes(payload) == expected["sha256"],
                    f"{label} tar member SHA differs",
                )
    except (OSError, tarfile.TarError) as exc:
        raise M2RawReplayError(f"invalid {label} untracked archive: {exc}") from exc

    if not capture["dirty"]:
        require(
            patch == b"" and not untracked, f"clean {label} capture contains changes"
        )
    snapshot = {
        "head": head,
        "tracked_diff_sha256": patch_sha,
        "tracked_diff_bytes": len(patch),
        "untracked": untracked,
    }
    snapshot_sha = _lower_sha256(
        capture["snapshot_sha256"], label=f"{label} snapshot SHA"
    )
    require(
        snapshot_sha == _canonical_digest(snapshot), f"{label} snapshot SHA differs"
    )
    return snapshot_sha


def _validate_provenance(
    provenance: dict[str, Any],
    *,
    result: dict[str, Any],
    run_dir: Path,
    entries: dict[str, str],
) -> tuple[str, str]:
    require(set(provenance) == PROVENANCE_FIELDS, "provenance fields differ from M2 v3")
    require(provenance["schema_version"] == PROTOCOL_SCHEMA, "wrong provenance schema")
    require(provenance["run_id"] == result["run_id"], "provenance run_id differs")
    require(provenance["mode"] == result["mode"], "provenance mode differs")
    require(provenance["full_provenance"] is True, "full provenance is required")
    require(
        provenance["prompt_token_ids"] == list(range(1000, 1017)),
        "prompt profile differs",
    )
    require(provenance["block_size"] == 16, "provenance block size differs")
    cpu_bytes = _positive_int(provenance["cpu_bytes"], label="provenance cpu_bytes")
    require(
        provenance["tolerance"] == result["tolerance"], "provenance tolerance differs"
    )
    argv = provenance["argv"]
    require(
        isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv),
        "provenance argv is invalid",
    )
    _nonempty_string(provenance["python"], label="provenance python")
    executable = _resolved_absolute_path(
        provenance["executable"], label="provenance executable"
    )
    started = _timestamp(provenance["started_at_utc"], label="started_at_utc")
    completed = _timestamp(result["completed_at_utc"], label="result completed_at_utc")

    if result["mode"] == "calibration":
        require(
            provenance["frozen_tolerance"] is None, "calibration has frozen tolerance"
        )
        require(provenance["calibration_cohort"] is None, "calibration has a cohort")
    else:
        require(
            isinstance(provenance["frozen_tolerance"], dict),
            "formal tolerance is missing",
        )
        require(
            isinstance(provenance["calibration_cohort"], dict),
            "formal cohort is missing",
        )

    connector = provenance["connector_config"]
    require(
        isinstance(connector, dict)
        and set(connector)
        == {
            "cpu_bytes_to_use",
            "dagkv_diagnostic_phase",
            "dagkv_diagnostic_run_id",
            "dagkv_diagnostic_trace_file",
            "fanout_layerwise_load",
            "lifecycle_accounting_enabled",
            "spec_module_path",
            "spec_name",
        },
        "connector provenance fields differ",
    )
    require(connector["cpu_bytes_to_use"] == cpu_bytes, "connector CPU bytes differ")
    require(connector["dagkv_diagnostic_phase"] == "ABBA", "connector phase differs")
    require(
        connector["dagkv_diagnostic_run_id"] == result["run_id"],
        "connector run_id differs",
    )
    require(
        Path(connector["dagkv_diagnostic_trace_file"]).resolve()
        == (run_dir / "diagnostic_transfers.jsonl").resolve(),
        "connector diagnostic path differs",
    )
    require(connector["fanout_layerwise_load"] is False, "fanout load must be disabled")
    require(
        connector["lifecycle_accounting_enabled"] is True,
        "lifecycle accounting is disabled",
    )
    require(
        connector["spec_module_path"] == "dagkv_vllm_m2.spec",
        "connector spec module differs",
    )
    require(
        connector["spec_name"] == "DAGKVDiagnosticCPUOffloadingSpec",
        "connector spec differs",
    )

    engine = provenance["engine_config"]
    require(
        isinstance(engine, dict) and set(engine) == ENGINE_CONFIG_FIELDS,
        "engine config fields differ",
    )
    frozen_engine = {
        "async_scheduling": False,
        "block_size": 16,
        "data_parallel_size": 1,
        "disable_hybrid_kv_cache_manager": True,
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.82,
        "logprobs_mode": "raw_logits",
        "max_logprobs": -1,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 1,
        "pipeline_parallel_size": 1,
        "scheduling_policy": "fcfs",
        "seed": 20260724,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
    }
    for key, expected in frozen_engine.items():
        require(engine[key] == expected, f"engine config {key} differs")
    require(
        engine["attention_config"]
        == {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "attention config differs",
    )

    implementation, implementation_content = _validate_file_manifest(
        provenance["implementation"],
        label="implementation",
        list_key="files",
        content_keys=("path", "size", "sha256"),
        expected_keys={"path", "size", "sha256"},
        require_hashes=True,
    )
    require(
        set(implementation) == {"files", "manifest_sha256"},
        "implementation capture fields differ",
    )
    implementation_sha = _lower_sha256(
        implementation["manifest_sha256"], label="implementation manifest"
    )
    require(
        implementation_sha == _canonical_digest(implementation_content),
        "implementation manifest differs",
    )
    implementation_paths = {item["path"]: item for item in implementation["files"]}
    for required_path in (
        "integrations/vllm_m2/dagkv_vllm_m2/spec.py",
        "research/REFERENCES.md",
        "research/imported/RELATED_WORK_MATRIX.md",
        "research/protocols/M2_VLLM_REPLAY_PROTOCOL.md",
        "tools/m2_raw_replay.py",
        "tools/nvidia_driver_userspace_bundle.py",
        "tools/run_m2_vllm_abba.py",
    ):
        require(
            required_path in implementation_paths,
            f"implementation omits {required_path}",
        )
    protocol_entry = implementation_paths[
        "research/protocols/M2_VLLM_REPLAY_PROTOCOL.md"
    ]
    require(
        protocol_entry["sha256"] == entries["protocol.md"],
        "protocol artifact differs from implementation",
    )

    model, model_content = _validate_file_manifest(
        provenance["model"],
        label="model",
        list_key="files",
        content_keys=("path", "size", "kind", "sha256"),
        expected_keys={"path", "size", "kind", "sha256", "mtime_ns", "inode"},
    )
    require(
        set(model) == {"files", "full_hashes", "manifest_sha256", "root"},
        "model capture fields differ",
    )
    require(model["full_hashes"] is True, "model hashes are incomplete")
    require(engine["model"] == model["root"], "engine/model roots differ")
    model_sha = _lower_sha256(model["manifest_sha256"], label="model manifest")
    require(model_sha == _canonical_digest(model_content), "model manifest differs")
    model_paths = {item["path"]: item for item in model["files"]}
    require(
        "config.json" in model_paths and "model.safetensors.index.json" in model_paths,
        "model metadata closed set is incomplete",
    )
    require(
        any(item["kind"] == "weight" for item in model["files"]),
        "model weights are missing",
    )
    for item in model["files"]:
        expected_kind = (
            "weight" if item["path"].endswith(".safetensors") else "metadata"
        )
        require(item["kind"] == expected_kind, f"model kind differs for {item['path']}")
        _lower_sha256(item["sha256"], label=f"model {item['path']} SHA")

    runtime = provenance["runtime_binaries"]
    runtime, extension_content = _validate_file_manifest(
        runtime,
        label="runtime",
        list_key="vllm_extensions",
        content_keys=("path", "size", "sha256"),
        expected_keys={"path", "size", "sha256", "mtime_ns", "inode"},
        require_hashes=True,
    )
    require(
        set(runtime)
        == {
            "full_hashes",
            "manifest_sha256",
            "python_executable",
            "root",
            "vllm_extensions",
        },
        "runtime capture fields differ",
    )
    require(runtime["full_hashes"] is True, "runtime hashes are incomplete")
    runtime_root = _resolved_absolute_path(runtime["root"], label="runtime root")
    require(
        all(item["path"].endswith(".so") for item in runtime["vllm_extensions"]),
        "runtime extension set is invalid",
    )
    python_entry = runtime["python_executable"]
    require(
        isinstance(python_entry, dict)
        and set(python_entry) == {"inode", "mtime_ns", "path", "sha256", "size"},
        "runtime Python capture fields differ",
    )
    runtime_python = _resolved_absolute_path(
        python_entry["path"], label="runtime Python path"
    )
    require(
        executable == runtime_python,
        "provenance executable does not resolve to the captured runtime Python",
    )
    _positive_int(python_entry["size"], label="runtime Python size")
    _positive_int(python_entry["mtime_ns"], label="runtime Python mtime")
    _positive_int(python_entry["inode"], label="runtime Python inode")
    _lower_sha256(python_entry["sha256"], label="runtime Python SHA")
    runtime_content = {
        "vllm_extensions": extension_content,
        "python_executable": {
            key: python_entry[key] for key in ("path", "size", "sha256")
        },
    }
    runtime_sha = _lower_sha256(runtime["manifest_sha256"], label="runtime manifest")
    require(
        runtime_sha == _canonical_digest(runtime_content), "runtime manifest differs"
    )

    dependencies = provenance["dependencies"]
    require(
        isinstance(dependencies, dict)
        and set(dependencies) == {"manifest_sha256", "packages"},
        "dependency capture fields differ",
    )
    packages = dependencies["packages"]
    require(isinstance(packages, list) and packages, "dependency list is empty")
    package_pairs: list[tuple[str, str]] = []
    for item in packages:
        require(
            isinstance(item, dict) and set(item) == {"name", "version"},
            "dependency entry fields differ",
        )
        package_pairs.append(
            (
                _nonempty_string(item["name"], label="dependency name"),
                _nonempty_string(item["version"], label="dependency version"),
            )
        )
    require(
        package_pairs == sorted(set(package_pairs)), "dependency list order differs"
    )
    dependency_sha = _lower_sha256(
        dependencies["manifest_sha256"], label="dependency manifest"
    )
    require(
        dependency_sha == _canonical_digest(packages), "dependency manifest differs"
    )

    dagkv_sha = _validate_git_capture(
        provenance["dagkv_git"], label="dagkv", run_dir=run_dir, entries=entries
    )
    vllm_sha = _validate_git_capture(
        provenance["vllm_git"], label="vllm", run_dir=run_dir, entries=entries
    )
    vllm_root = _resolved_absolute_path(
        provenance["vllm_git"]["root"], label="vllm Git root"
    )
    require(runtime_root == vllm_root, "runtime and vLLM Git roots differ")
    system = provenance["system"]
    require(isinstance(system, dict) and system, "system provenance is missing")
    nvidia_content_digest = _validate_nvidia_driver_userspace(
        provenance["nvidia_driver_userspace"],
        argv=argv,
        system=system,
        started=started,
    )
    preflight = provenance["preflight"]
    require(
        isinstance(preflight, dict)
        and set(preflight)
        == {
            "cuda_version",
            "gpu_count",
            "gpu_name",
            "model_architectures",
            "model_config_sha256",
            "torch_version",
            "vllm_module",
            "vllm_version",
        },
        "preflight fields differ",
    )
    require(preflight["gpu_count"] == 1, "preflight GPU count differs")
    vllm_module = _resolved_absolute_path(
        preflight["vllm_module"], label="preflight vLLM module"
    )
    require(
        vllm_module.is_relative_to(vllm_root),
        "preflight vLLM module is outside the captured vLLM root",
    )
    require(
        any("Qwen3" in str(value) for value in preflight["model_architectures"]),
        "preflight model is not Qwen3",
    )
    require(
        preflight["model_config_sha256"] == model_paths["config.json"]["sha256"],
        "preflight model config SHA differs",
    )

    components = provenance["reproducibility_components"]
    require(
        isinstance(components, dict) and set(components) == COMPONENT_FIELDS,
        "reproducibility component fields differ",
    )
    static_connector = {
        key: value
        for key, value in connector.items()
        if key not in {"dagkv_diagnostic_trace_file", "dagkv_diagnostic_run_id"}
    }
    expected_components = {
        "implementation_manifest_sha256": implementation_sha,
        "vllm_snapshot_sha256": vllm_sha,
        "model_manifest_sha256": model_sha,
        "runtime_binary_manifest_sha256": runtime_sha,
        "dependency_manifest_sha256": dependency_sha,
        "nvidia_driver_userspace_content_digest": nvidia_content_digest,
        "system": system,
        "prompt_token_ids": list(range(1000, 1017)),
        "block_size": 16,
        "cpu_bytes": cpu_bytes,
        "engine_config": engine,
        "connector_config": static_connector,
    }
    require(
        components == expected_components,
        "reproducibility components differ from provenance",
    )
    fingerprint = _lower_sha256(
        provenance["reproducibility_fingerprint"], label="reproducibility fingerprint"
    )
    require(
        fingerprint == _canonical_digest(components),
        "reproducibility fingerprint differs",
    )
    require(
        result["reproducibility_fingerprint"] == fingerprint,
        "result fingerprint differs",
    )

    postflight = provenance["postflight"]
    require(
        isinstance(postflight, dict)
        and set(postflight)
        == {
            "completed_at_utc",
            "dagkv_git_snapshot_sha256",
            "implementation_manifest_sha256",
            "model_file_stats_unchanged",
            "libcuda_mapping_unchanged",
            "nvidia_driver_userspace_content_digest",
            "nvidia_driver_userspace_manifest_sha256",
            "nvidia_driver_userspace_unchanged",
            "runtime_binary_stats_unchanged",
            "vllm_git_snapshot_sha256",
        },
        "postflight fields differ",
    )
    post_completed = _timestamp(
        postflight["completed_at_utc"], label="postflight completed_at_utc"
    )
    require(
        started <= post_completed <= completed, "provenance timestamps are out of order"
    )
    require(
        postflight["implementation_manifest_sha256"] == implementation_sha,
        "postflight implementation differs",
    )
    require(
        postflight["dagkv_git_snapshot_sha256"] == dagkv_sha,
        "postflight DAGKV snapshot differs",
    )
    require(
        postflight["vllm_git_snapshot_sha256"] == vllm_sha,
        "postflight vLLM snapshot differs",
    )
    require(
        postflight["model_file_stats_unchanged"] is True, "postflight model changed"
    )
    require(
        postflight["runtime_binary_stats_unchanged"] is True,
        "postflight runtime changed",
    )
    require(
        postflight["nvidia_driver_userspace_content_digest"] == nvidia_content_digest,
        "postflight NVIDIA content digest differs",
    )
    require(
        postflight["nvidia_driver_userspace_manifest_sha256"]
        == provenance["nvidia_driver_userspace"]["manifest_sha256"]
        == provenance["nvidia_driver_userspace"]["expected_manifest_sha256"],
        "postflight NVIDIA manifest digest differs",
    )
    require(
        postflight["nvidia_driver_userspace_unchanged"] is True,
        "postflight NVIDIA userspace bundle changed",
    )
    require(
        postflight["libcuda_mapping_unchanged"] is True,
        "postflight libcuda mapping changed",
    )
    return fingerprint, implementation_sha


def validate_raw_run(run_dir: Path) -> RawReplayValidation:
    """Replay a complete M2 run directory from immutable raw artifacts."""

    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"M2 run directory is missing: {run_dir}")
    entries = _validate_sha256sums(run_dir)
    result = _read_json_object(run_dir / "result.json", label="result.json")
    _validate_prefetch_and_artifacts(result)
    _, observed_max, minimum_margin = _validate_result_and_logits(
        run_dir, result, entries
    )
    execution_ids = _read_json_object(
        run_dir / "execution_ids.json", label="execution_ids.json"
    )
    transfers = _validate_execution_ids(execution_ids, result=result)
    diagnostic_rows = _read_jsonl(
        run_dir / "diagnostic_transfers.jsonl", label="diagnostic trace"
    )
    diagnostic = _validate_diagnostic_trace(
        diagnostic_rows,
        run_id=result["run_id"],
        transfers=transfers,
        result=result,
    )
    native_rows = _read_jsonl(run_dir / "native_lifecycle.jsonl", label="native trace")
    _validate_native_trace(
        native_rows,
        run_id=result["run_id"],
        transfers=transfers,
        diagnostic=diagnostic,
        execution_ids=execution_ids,
        result=result,
    )
    provenance = _read_json_object(run_dir / "provenance.json", label="provenance.json")
    fingerprint, implementation_sha = _validate_provenance(
        provenance,
        result=result,
        run_dir=run_dir,
        entries=entries,
    )
    return RawReplayValidation(
        run_id=result["run_id"],
        mode=result["mode"],
        observed_max_abs_error=observed_max,
        minimum_top1_margin=minimum_margin,
        reproducibility_fingerprint=fingerprint,
        implementation_manifest_sha256=implementation_sha,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validated = validate_raw_run(args.run_dir)
    except (M2RawReplayError, OSError) as exc:
        print(f"M2 raw replay failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"M2 raw replay passed: {validated.run_id} "
        f"(max_abs={validated.observed_max_abs_error})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
