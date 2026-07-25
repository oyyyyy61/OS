#!/usr/bin/env python3
"""Independently replay and seal one complete M2 formal campaign bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.nvidia_driver_userspace_bundle import (  # noqa: E402
    BundleValidation,
    NvidiaUserspaceBundleError,
    validate_bundle,
)

CAMPAIGN_SCHEMA = "dagkv.m2.formal_campaign_preregistration.v2"
ATTEMPT_SCHEMA = "dagkv.m2.formal_campaign_attempt.v1"
FORMAL_PROTOCOL_SCHEMA = "dagkv.m2.formal_campaign.v1"
DATA_PLANE_PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v3"
ACCEPTANCE_SCHEMA = "dagkv.m2.item8.acceptance.v2"
SEAL_SCHEMA = "dagkv.m2.formal_bundle_seal.v2"
FORMAL_RUN_COUNT = 20
RUN_RECORD_COUNT = FORMAL_RUN_COUNT * 2
FULL_RECORD_COUNT = RUN_RECORD_COUNT + 2
CUDA_LIBRARY_PATH = "/usr/local/cuda/lib64"
LOADER_INJECTION_VARIABLES = ("LD_AUDIT", "LD_PRELOAD")
INTEGRATION_ROOT = REPO_ROOT / "integrations" / "vllm_m2"
MARKER_SCHEMA = "dagkv.m2.formal_launch_marker.v1"
LAUNCH_MARKER_REPOSITORY_PATH = "evidence/m2/FORMAL_LAUNCH_MARKER.json"
MARKER_CLAIM_SCOPE = "M2_ITEM8_CORRECTNESS_ONLY_NO_PERFORMANCE_CLAIM"

PREREGISTRATION_NAME = "FORMAL_CAMPAIGN_PREREGISTRATION.json"
ATTEMPTS_NAME = "FORMAL_ATTEMPTS.jsonl"
ACCEPTANCE_NAME = "M2_ITEM8_ACCEPTANCE_MANIFEST.json"
SEAL_NAME = "M2_FORMAL_BUNDLE_SEAL.json"
FORMAL_RUN_MANIFEST = "M2_ITEM8_FORMAL_RUN_MANIFEST.json"
ACCEPTANCE_STATEMENT = (
    "Exactly twenty frozen M2 item 8 holdouts passed. This closes item 8 only; "
    "the aggregate M2 gate remains open and this evidence supports no latency, "
    "throughput, hit-rate, scheduling-policy, or paper-performance claim."
)
SEAL_STATEMENT = (
    "The complete preregistered 20-process M2 item-8 formal bundle was replayed "
    "and sealed. This closes item 8 only and supports no performance claim."
)

PREREGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_root",
        "created_at_utc",
        "preparation_git_head",
        "launch_marker_repository_path",
        "formal_campaign_protocol_schema",
        "data_plane_protocol_schema",
        "formal_campaign_protocol_sha256",
        "data_plane_protocol_sha256",
        "expected_runs",
        "production_run_count",
        "test_injected_run_count",
        "run_names",
        "formal_attempt_prefix_record_count",
        "selection_rule",
        "expected_implementation_manifest_sha256",
        "expected_reproducibility_fingerprint",
        "nvidia_userspace_bundle_root",
        "expected_nvidia_userspace_bundle_manifest_sha256",
        "expected_nvidia_userspace_bundle_content_digest",
        "expected_nvidia_driver_version",
        "parent_binding",
        "frozen_files",
        "python_executable",
        "model",
        "vllm_root",
        "cpu_bytes",
        "cuda_device",
        "runner_timeout_s",
        "process_timeout_s",
        "aggregation_timeout_s",
        "terminate_grace_s",
        "kill_wait_s",
        "retry_policy",
        "attempts_file",
        "acceptance_output",
        "runner_command_template",
        "aggregate_command",
        "environment_overrides",
    }
)
FROZEN_FILE_NAMES = frozenset(
    {
        "formal_protocol",
        "data_plane_protocol",
        "runner",
        "launcher",
        "formal_aggregator",
        "formal_evidence",
        "raw_replay",
        "calibration_evidence",
        "process_supervisor",
        "nvidia_bundle_validator",
        "frozen_tolerance",
        "calibration_manifest",
    }
)
PARENT_BINDING_FIELDS = frozenset(
    {
        "calibration_manifest_sha256",
        "frozen_tolerance_sha256",
        "reproducibility_fingerprint",
        "frozen_at_utc",
        "calibration_run_count",
        "nvidia_userspace_bundle_root",
        "expected_nvidia_userspace_bundle_manifest_sha256",
        "expected_nvidia_userspace_bundle_content_digest",
        "expected_nvidia_driver_version",
    }
)
RUN_SUBMITTED_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "attempt_id",
        "kind",
        "event",
        "timestamp_utc",
        "sequence",
        "run_name",
        "command",
        "output_dir",
        "stdout",
        "stderr",
        "preregistration_sha256",
        "execution_binding",
    }
)
TERMINAL_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "attempt_id",
        "kind",
        "event",
        "timestamp_utc",
        "status",
        "pid",
        "exit_code",
        "duration_s",
        "started_at_utc",
        "ended_at_utc",
        "timed_out",
        "sigterm_sent",
        "sigkill_sent",
        "stdout",
        "stderr",
        "artifact_inventory",
        "error",
        "validation",
    }
)
RUN_TERMINAL_FIELDS = TERMINAL_BASE_FIELDS | {"sequence", "run_name"}
AGGREGATE_SUBMITTED_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "attempt_id",
        "kind",
        "event",
        "timestamp_utc",
        "command",
        "output",
        "stdout",
        "stderr",
        "preregistration_sha256",
        "execution_binding",
        "formal_prefix",
    }
)
RUN_VALIDATION_FIELDS = frozenset(
    {
        "run_id",
        "result_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
        "formal_run_manifest_sha256",
        "frozen_tolerance_sha256",
        "calibration_manifest_sha256",
        "implementation_manifest_sha256",
        "reproducibility_fingerprint",
        "protocol_sha256",
        "observed_max_abs_error",
        "minimum_top1_margin",
        "dagkv_git_head",
        "dagkv_snapshot_sha256",
    }
)
ACCEPTANCE_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_schema",
        "gate_status",
        "run_count",
        "passed_run_count",
        "m2_item8_accepted",
        "m2_accepted",
        "performance_claims_supported",
        "frozen_tolerance_sha256",
        "calibration_manifest_sha256",
        "reproducibility_fingerprint",
        "protocol_sha256",
        "nvidia_userspace_bundle_root",
        "nvidia_userspace_bundle_manifest_sha256",
        "nvidia_userspace_bundle_content_digest",
        "nvidia_driver_version",
        "runs",
        "statement",
    }
)
ACCEPTANCE_RUN_FIELDS = frozenset(
    {
        "run_id",
        "formal_run_manifest_sha256",
        "result_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
    }
)
SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "sealed_at_utc",
        "formal_campaign_preregistration_file",
        "formal_campaign_preregistration_sha256",
        "formal_attempts_file",
        "formal_prefix_bytes",
        "formal_prefix_record_count",
        "formal_prefix_sha256",
        "full_journal_bytes",
        "full_journal_record_count",
        "full_journal_sha256",
        "acceptance_file",
        "acceptance_sha256",
        "calibration_manifest_sha256",
        "frozen_tolerance_sha256",
        "implementation_manifest_sha256",
        "reproducibility_fingerprint",
        "nvidia_userspace_bundle_root",
        "nvidia_userspace_bundle_manifest_sha256",
        "nvidia_userspace_bundle_content_digest",
        "nvidia_driver_version",
        "execution_binding",
        "dagkv_snapshot_sha256",
        "run_count",
        "ordered_runs",
        "m2_item8_accepted",
        "m2_accepted",
        "performance_claims_supported",
        "statement",
    }
)
SEAL_RUN_FIELDS = frozenset(
    {
        "sequence",
        "run_name",
        "attempt_id",
        "run_id",
        "result_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
        "formal_run_manifest_sha256",
        "dagkv_git_head",
        "dagkv_snapshot_sha256",
    }
)
MARKER_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_root",
        "campaign_preregistration_sha256",
        "preparation_git_head",
        "created_at_utc",
        "claim_scope",
    }
)
EXECUTION_BINDING_FIELDS = frozenset(
    {
        "preparation_git_head",
        "execution_git_head",
        "launch_marker_repository_path",
        "launch_marker_sha256",
    }
)


class FormalEvidenceError(RuntimeError):
    """Raised when formal orchestration evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class FormalBundleValidation:
    campaign_root: Path
    campaign_id: str
    preregistration_sha256: str
    formal_prefix_bytes: int
    formal_prefix_sha256: str
    full_journal_bytes: int
    full_journal_sha256: str
    acceptance_sha256: str
    calibration_manifest_sha256: str
    frozen_tolerance_sha256: str
    implementation_manifest_sha256: str
    reproducibility_fingerprint: str
    nvidia_bundle_validation: BundleValidation
    execution_binding: dict[str, str]
    dagkv_snapshot_sha256: str
    ordered_runs: tuple[dict[str, Any], ...]
    input_inventory: tuple[tuple[Any, ...], ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalEvidenceError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise FormalEvidenceError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _lower_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase SHA-256",
    )
    return value


def _git_head(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-1 Git object ID",
    )
    return value


def _git_bytes(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = (
            exc.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        raise FormalEvidenceError(
            f"Git command failed ({' '.join(arguments)}): {stderr}"
        ) from exc
    return completed.stdout


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FormalEvidenceError(f"non-finite JSON constant: {value}")


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except FormalEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalEvidenceError(f"invalid {label}: {exc}") from exc


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    require(path.stat().st_nlink == 1, f"{label} must have exactly one hard link")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FormalEvidenceError(f"cannot read {label}: {exc}") from exc
    payload = _decode_json(raw, label=label)
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload, raw


def _exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    require(set(payload) == expected, f"{label} fields drifted")


def _timestamp(value: Any, *, label: str) -> datetime:
    require(isinstance(value, str), f"{label} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise FormalEvidenceError(f"invalid {label}: {value}") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _file_identity(path: Path) -> tuple[Any, ...]:
    stat_result = path.stat(follow_symlinks=False)
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_nlink,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _capture_tree_inventory(
    root: Path, *, include_seal: bool
) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == SEAL_NAME and not include_seal:
            continue
        require(not path.is_symlink(), f"formal bundle contains symlink: {relative}")
        if path.is_file():
            rows.append((relative, "file", *_file_identity(path), _sha256_file(path)))
        elif path.is_dir():
            rows.append((relative, "directory", *_file_identity(path)))
        else:
            raise FormalEvidenceError(
                f"formal bundle contains special node: {relative}"
            )
    return tuple(rows)


def _artifact_inventory(path: Path) -> list[dict[str, Any]]:
    if path.is_file() and not path.is_symlink():
        return [{"path": path.name, "type": "file", "size": path.stat().st_size}]
    require(path.is_dir(), f"missing artifact directory: {path}")
    inventory: list[dict[str, Any]] = []
    for artifact in sorted(path.rglob("*")):
        relative = artifact.relative_to(path).as_posix()
        require(
            not artifact.is_symlink(),
            f"artifact inventory contains symlink: {artifact}",
        )
        if artifact.is_file():
            inventory.append(
                {"path": relative, "type": "file", "size": artifact.stat().st_size}
            )
        elif artifact.is_dir():
            inventory.append({"path": relative, "type": "directory"})
        else:
            raise FormalEvidenceError(
                f"artifact inventory contains special node: {artifact}"
            )
    return inventory


def _file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing file: {path}")
    require(path.stat().st_nlink == 1, f"formal root file has multiple links: {path}")
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _expected_selection(run_names: Sequence[str]) -> dict[str, Any]:
    return {
        "attempts_per_run": 1,
        "eligible_kind": "formal_run",
        "ordered_run_names": list(run_names),
        "required_events": ["submitted", "terminal"],
        "required_terminal_status": "passed",
        "retry_count": 0,
        "stop_on_first_failure": True,
    }


def _verify_frozen_files(preregistration: Mapping[str, Any]) -> None:
    frozen = preregistration.get("frozen_files")
    require(isinstance(frozen, dict), "formal preregistration lacks frozen files")
    require(set(frozen) == FROZEN_FILE_NAMES, "formal frozen-file set drifted")
    for label, entry in frozen.items():
        require(isinstance(entry, dict), f"frozen {label} entry must be an object")
        require(
            set(entry) == {"path", "size", "sha256"}, f"frozen {label} fields drifted"
        )
        path = Path(str(entry["path"]))
        require(path.is_file() and not path.is_symlink(), f"frozen {label} is missing")
        require(
            type(entry["size"]) is int and entry["size"] >= 0,
            f"frozen {label} size invalid",
        )
        require(
            path.stat().st_size == entry["size"]
            and _sha256_file(path)
            == _lower_sha256(entry["sha256"], label=f"frozen {label}"),
            f"frozen {label} content drifted",
        )


def _positive_number(value: Any, *, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0,
        f"{label} must be a positive finite number",
    )
    return float(value)


def _validate_nvidia_bundle_binding(
    preregistration: Mapping[str, Any],
) -> BundleValidation:
    """Independently reconstruct the NVIDIA bundle and live driver binding."""

    injected = [name for name in LOADER_INJECTION_VARIABLES if name in os.environ]
    require(
        not injected,
        f"loader injection environment is forbidden: {','.join(injected)}",
    )
    root_value = preregistration.get("nvidia_userspace_bundle_root")
    require(
        isinstance(root_value, str) and root_value,
        "formal NVIDIA userspace bundle root is invalid",
    )
    root = Path(root_value)
    require(root.is_absolute(), "formal NVIDIA userspace bundle root is not absolute")
    manifest_sha = _lower_sha256(
        preregistration.get("expected_nvidia_userspace_bundle_manifest_sha256"),
        label="formal NVIDIA userspace bundle manifest",
    )
    content_digest = _lower_sha256(
        preregistration.get("expected_nvidia_userspace_bundle_content_digest"),
        label="formal NVIDIA userspace bundle content",
    )
    expected_driver = preregistration.get("expected_nvidia_driver_version")
    require(
        isinstance(expected_driver, str)
        and expected_driver
        and expected_driver.strip() == expected_driver,
        "formal expected NVIDIA driver version is invalid",
    )
    try:
        validation = validate_bundle(
            root,
            expected_manifest_sha256=manifest_sha,
        )
    except (NvidiaUserspaceBundleError, OSError) as exc:
        raise FormalEvidenceError(
            f"NVIDIA userspace bundle replay failed: {exc}"
        ) from exc
    require(
        validation.manifest_sha256 == manifest_sha,
        "formal NVIDIA userspace bundle manifest digest drifted",
    )
    require(
        validation.content_digest == content_digest,
        "formal NVIDIA userspace bundle content digest drifted",
    )
    require(
        validation.kernel_module_version == expected_driver,
        "formal NVIDIA kernel driver version drifted",
    )
    require(
        validation.runtime.rootfs.parent == root,
        "formal NVIDIA bundle rootfs binding drifted",
    )
    return validation


def _validate_execution_freeze(
    preregistration: Mapping[str, Any], *, root: Path
) -> BundleValidation:
    """Reconstruct the two commands and environment without trusting the launcher."""

    frozen = preregistration["frozen_files"]
    python_record = preregistration["python_executable"]
    require(isinstance(python_record, dict), "formal Python record must be an object")
    require(
        set(python_record) == {"path", "size", "sha256"},
        "formal Python record fields drifted",
    )
    python_path = Path(str(python_record["path"]))
    require(
        python_path.is_absolute()
        and python_path.is_file()
        and os.access(python_path, os.X_OK),
        "formal Python executable is missing or not executable",
    )
    require(
        type(python_record["size"]) is int
        and python_record["size"] >= 0
        and python_path.stat().st_size == python_record["size"]
        and _sha256_file(python_path)
        == _lower_sha256(python_record["sha256"], label="formal Python executable"),
        "formal Python executable content drifted",
    )

    model = Path(str(preregistration["model"]))
    vllm_root = Path(str(preregistration["vllm_root"]))
    require(
        model.is_absolute() and model.is_dir() and not model.is_symlink(),
        "formal model root is missing or unsafe",
    )
    require(
        vllm_root.is_absolute() and vllm_root.is_dir() and not vllm_root.is_symlink(),
        "formal vLLM root is missing or unsafe",
    )
    cpu_bytes = preregistration["cpu_bytes"]
    cuda_device = preregistration["cuda_device"]
    require(
        type(cpu_bytes) is int and cpu_bytes > 0,
        "formal CPU allocation must be a positive integer",
    )
    require(
        type(cuda_device) is int and cuda_device >= 0,
        "formal CUDA device must be a non-negative integer",
    )
    for field in (
        "runner_timeout_s",
        "process_timeout_s",
        "aggregation_timeout_s",
        "terminate_grace_s",
        "kill_wait_s",
    ):
        _positive_number(preregistration[field], label=f"formal {field}")

    bundle_validation = _validate_nvidia_bundle_binding(preregistration)
    runner_template = [
        str(python_path),
        str(frozen["runner"]["path"]),
        "--output-dir",
        "<RUN_DIR>",
        "--mode",
        "formal",
        "--tolerance-file",
        str(frozen["frozen_tolerance"]["path"]),
        "--calibration-manifest",
        str(frozen["calibration_manifest"]["path"]),
        "--model",
        str(model),
        "--vllm-root",
        str(vllm_root),
        "--expected-nvidia-driver-version",
        preregistration["expected_nvidia_driver_version"],
        "--nvidia-userspace-bundle-root",
        preregistration["nvidia_userspace_bundle_root"],
        "--expected-nvidia-userspace-bundle-manifest-sha256",
        preregistration["expected_nvidia_userspace_bundle_manifest_sha256"],
        "--expected-nvidia-userspace-bundle-content-digest",
        preregistration["expected_nvidia_userspace_bundle_content_digest"],
        "--cpu-bytes",
        str(cpu_bytes),
        "--timeout-s",
        str(float(preregistration["runner_timeout_s"])),
        "--cuda-device",
        str(cuda_device),
        "--full-provenance",
    ]
    require(
        preregistration["runner_command_template"] == runner_template,
        "formal runner command template drifted",
    )
    aggregate_command = [
        str(python_path),
        str(frozen["formal_aggregator"]["path"]),
        "--campaign-dir",
        str(root),
        "--calibration-manifest",
        str(frozen["calibration_manifest"]["path"]),
        "--frozen-tolerance",
        str(frozen["frozen_tolerance"]["path"]),
        "--output",
        str(root / ACCEPTANCE_NAME),
    ]
    require(
        preregistration["aggregate_command"] == aggregate_command,
        "formal aggregate command drifted",
    )
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": str(cuda_device),
        "HF_HUB_OFFLINE": "1",
        "LD_LIBRARY_PATH": (f"{bundle_validation.library_path}:{CUDA_LIBRARY_PATH}"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(INTEGRATION_ROOT),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    require(
        preregistration["environment_overrides"] == expected_environment,
        "formal environment override freeze drifted",
    )
    return bundle_validation


def _validate_execution_binding(
    binding: Any,
    *,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
) -> dict[str, str]:
    require(isinstance(binding, dict), "formal execution binding must be an object")
    _exact_fields(binding, EXECUTION_BINDING_FIELDS, label="formal execution binding")
    preparation_head = _git_head(
        binding["preparation_git_head"], label="formal preparation Git HEAD"
    )
    execution_head = _git_head(
        binding["execution_git_head"], label="formal execution Git HEAD"
    )
    require(
        preparation_head == preregistration["preparation_git_head"],
        "formal execution preparation HEAD drifted",
    )
    marker_path = binding["launch_marker_repository_path"]
    require(
        marker_path
        == preregistration["launch_marker_repository_path"]
        == LAUNCH_MARKER_REPOSITORY_PATH,
        "formal execution marker path drifted",
    )
    marker_sha = _lower_sha256(
        binding["launch_marker_sha256"], label="formal launch marker"
    )
    parent_row = (
        _git_bytes("rev-list", "--parents", "-n", "1", execution_head)
        .decode("ascii")
        .split()
    )
    require(
        parent_row == [execution_head, preparation_head],
        "formal execution HEAD is not the direct single-parent marker commit",
    )
    changed_paths = (
        _git_bytes("diff-tree", "--no-commit-id", "--name-only", "-r", execution_head)
        .decode("utf-8")
        .splitlines()
    )
    require(
        changed_paths == [marker_path],
        "formal execution commit changed files beyond the launch marker",
    )
    marker_raw = _git_bytes("cat-file", "blob", f"{execution_head}:{marker_path}")
    require(
        hashlib.sha256(marker_raw).hexdigest() == marker_sha,
        "formal launch marker Git object hash drifted",
    )
    marker = _decode_json(marker_raw, label="formal launch marker")
    require(isinstance(marker, dict), "formal launch marker must be an object")
    _exact_fields(marker, MARKER_FIELDS, label="formal launch marker")
    require(
        marker["schema_version"] == MARKER_SCHEMA
        and marker["campaign_id"] == preregistration["campaign_id"]
        and marker["campaign_root"] == preregistration["campaign_root"]
        and marker["campaign_preregistration_sha256"] == preregistration_sha256
        and marker["preparation_git_head"] == preparation_head
        and marker["claim_scope"] == MARKER_CLAIM_SCOPE,
        "formal launch marker content binding drifted",
    )
    marker_created = _timestamp(
        marker["created_at_utc"], label="formal launch marker creation"
    )
    preregistered = _timestamp(
        preregistration["created_at_utc"], label="formal preregistration creation"
    )
    require(
        marker_created >= preregistered,
        "formal launch marker predates preregistration",
    )
    return {
        "preparation_git_head": preparation_head,
        "execution_git_head": execution_head,
        "launch_marker_repository_path": marker_path,
        "launch_marker_sha256": marker_sha,
    }


def _current_implementation_sha256() -> str:
    try:
        from tools.run_m2_vllm_abba import _implementation_capture  # noqa: PLC0415

        value = _implementation_capture().get("manifest_sha256")
    except (ImportError, OSError, AttributeError) as exc:
        raise FormalEvidenceError(
            f"cannot capture current implementation: {exc}"
        ) from exc
    return _lower_sha256(value, label="current implementation manifest")


def _validate_parent(preregistration: Mapping[str, Any]) -> Any:
    try:
        from tools.aggregate_m2_formal import (  # noqa: PLC0415
            FormalAggregationError,
            _validate_parent_evidence,
        )

        frozen = preregistration["frozen_files"]
        parent = _validate_parent_evidence(
            Path(frozen["calibration_manifest"]["path"]),
            Path(frozen["frozen_tolerance"]["path"]),
        )
    except (FormalAggregationError, KeyError, OSError) as exc:
        raise FormalEvidenceError(
            f"formal parent evidence replay failed: {exc}"
        ) from exc
    return parent


def _validate_calibration_nvidia_binding(
    preregistration: Mapping[str, Any],
) -> dict[str, str]:
    """Bind the formal bundle identity to the independently replayed calibration."""

    calibration_path = Path(
        preregistration["frozen_files"]["calibration_manifest"]["path"]
    )
    calibration, _ = _read_json_object(
        calibration_path,
        label="calibration manifest for formal NVIDIA binding",
    )
    preregistration_name = calibration.get("campaign_preregistration_file")
    require(
        preregistration_name == "CAMPAIGN_PREREGISTRATION.json",
        "calibration preregistration filename drifted",
    )
    calibration_preregistration, calibration_preregistration_raw = _read_json_object(
        calibration_path.parent / preregistration_name,
        label="calibration preregistration for formal NVIDIA binding",
    )
    require(
        hashlib.sha256(calibration_preregistration_raw).hexdigest()
        == calibration.get("campaign_preregistration_sha256"),
        "calibration preregistration hash binding drifted",
    )
    binding = {
        "nvidia_userspace_bundle_root": preregistration["nvidia_userspace_bundle_root"],
        "expected_nvidia_userspace_bundle_manifest_sha256": preregistration[
            "expected_nvidia_userspace_bundle_manifest_sha256"
        ],
        "expected_nvidia_userspace_bundle_content_digest": preregistration[
            "expected_nvidia_userspace_bundle_content_digest"
        ],
        "expected_nvidia_driver_version": preregistration[
            "expected_nvidia_driver_version"
        ],
    }
    require(
        all(
            calibration_preregistration.get(field) == value
            for field, value in binding.items()
        ),
        "calibration NVIDIA userspace/driver binding differs from formal evidence",
    )
    return binding


def _validate_preregistration(
    root: Path,
    *,
    expected_preregistration_sha256: str | None,
) -> tuple[dict[str, Any], str, Any, BundleValidation]:
    path = root / PREREGISTRATION_NAME
    preregistration, raw = _read_json_object(path, label="formal preregistration")
    _exact_fields(
        preregistration, PREREGISTRATION_FIELDS, label="formal preregistration"
    )
    observed_sha = hashlib.sha256(raw).hexdigest()
    if expected_preregistration_sha256 is not None:
        require(
            observed_sha
            == _lower_sha256(
                expected_preregistration_sha256,
                label="expected formal preregistration",
            ),
            "formal preregistration SHA-256 differs",
        )
    require(
        preregistration["schema_version"] == CAMPAIGN_SCHEMA,
        "formal preregistration schema drifted",
    )
    require(
        preregistration["campaign_root"] == str(root),
        "formal preregistration root binding drifted",
    )
    require(
        preregistration["formal_campaign_protocol_schema"] == FORMAL_PROTOCOL_SCHEMA
        and preregistration["data_plane_protocol_schema"] == DATA_PLANE_PROTOCOL_SCHEMA,
        "formal protocol schema drifted",
    )
    run_names = [f"run-{index:03d}" for index in range(1, FORMAL_RUN_COUNT + 1)]
    require(
        preregistration["expected_runs"] == FORMAL_RUN_COUNT
        and preregistration["production_run_count"] == FORMAL_RUN_COUNT
        and preregistration["test_injected_run_count"] is False,
        "formal production run-count freeze drifted",
    )
    require(
        preregistration["run_names"] == run_names, "formal ordered run names drifted"
    )
    require(
        preregistration["formal_attempt_prefix_record_count"] == RUN_RECORD_COUNT,
        "formal prefix count drifted",
    )
    require(
        preregistration["selection_rule"] == _expected_selection(run_names),
        "formal selection rule drifted",
    )
    require(
        preregistration["retry_policy"] == "none_stop_on_first_failure",
        "formal retry policy drifted",
    )
    require(
        preregistration["attempts_file"] == ATTEMPTS_NAME,
        "formal attempt filename drifted",
    )
    require(
        preregistration["acceptance_output"] == ACCEPTANCE_NAME,
        "formal acceptance filename drifted",
    )
    campaign_id = preregistration["campaign_id"]
    require(
        isinstance(campaign_id, str)
        and campaign_id.startswith("m2-formal-")
        and len(campaign_id.removeprefix("m2-formal-")) == 32,
        "formal campaign ID is malformed",
    )
    _timestamp(
        preregistration["created_at_utc"], label="formal preregistration creation"
    )
    implementation = _lower_sha256(
        preregistration["expected_implementation_manifest_sha256"],
        label="expected implementation manifest",
    )
    fingerprint = _lower_sha256(
        preregistration["expected_reproducibility_fingerprint"],
        label="expected reproducibility fingerprint",
    )
    require(
        _current_implementation_sha256() == implementation,
        "current implementation differs from formal preregistration",
    )
    _git_head(
        preregistration["preparation_git_head"],
        label="formal preregistration preparation Git HEAD",
    )
    require(
        preregistration["launch_marker_repository_path"]
        == LAUNCH_MARKER_REPOSITORY_PATH,
        "formal preregistration launch marker path drifted",
    )
    _verify_frozen_files(preregistration)
    bundle_before = _validate_execution_freeze(preregistration, root=root)
    frozen = preregistration["frozen_files"]
    require(
        preregistration["formal_campaign_protocol_sha256"]
        == frozen["formal_protocol"]["sha256"]
        and preregistration["data_plane_protocol_sha256"]
        == frozen["data_plane_protocol"]["sha256"],
        "formal protocol file hash binding drifted",
    )
    parent_binding = preregistration["parent_binding"]
    require(isinstance(parent_binding, dict), "formal parent binding must be an object")
    _exact_fields(parent_binding, PARENT_BINDING_FIELDS, label="formal parent binding")
    parent = _validate_parent(preregistration)
    calibration_nvidia_binding = _validate_calibration_nvidia_binding(preregistration)
    require(
        parent.calibration_manifest_sha256
        == parent_binding["calibration_manifest_sha256"]
        == frozen["calibration_manifest"]["sha256"],
        "formal calibration manifest binding drifted",
    )
    require(
        parent.frozen_tolerance_sha256
        == parent_binding["frozen_tolerance_sha256"]
        == frozen["frozen_tolerance"]["sha256"],
        "formal tolerance binding drifted",
    )
    require(
        parent.reproducibility_fingerprint
        == parent_binding["reproducibility_fingerprint"]
        == fingerprint,
        "formal parent fingerprint binding drifted",
    )
    require(
        parent.frozen_at_utc.isoformat() == parent_binding["frozen_at_utc"]
        and len(parent.calibration_run_ids)
        == parent_binding["calibration_run_count"]
        == 59,
        "formal parent cohort binding drifted",
    )
    require(
        all(
            parent_binding[field] == value
            for field, value in calibration_nvidia_binding.items()
        ),
        "formal parent NVIDIA userspace/driver binding drifted",
    )
    return preregistration, observed_sha, parent, bundle_before


def _validate_formal_run(run_dir: Path) -> dict[str, Any]:
    try:
        from tools.aggregate_m2_formal import (  # noqa: PLC0415
            FormalAggregationError,
            _validate_run,
        )
        from tools.m2_raw_replay import (  # noqa: PLC0415
            M2RawReplayError,
            validate_raw_run,
        )

        raw = validate_raw_run(run_dir)
        validated = _validate_run(run_dir)
    except (FormalAggregationError, M2RawReplayError, OSError) as exc:
        raise FormalEvidenceError(
            f"formal run replay failed for {run_dir.name}: {exc}"
        ) from exc
    require(
        raw.mode == "formal" and raw.run_id == validated.run_id,
        f"raw identity drifted for {run_dir.name}",
    )
    require(
        raw.reproducibility_fingerprint == validated.reproducibility_fingerprint,
        f"raw fingerprint drifted for {run_dir.name}",
    )
    provenance, _ = _read_json_object(
        run_dir / "provenance.json", label=f"{run_dir.name} provenance"
    )
    dagkv_git = provenance.get("dagkv_git")
    require(isinstance(dagkv_git, dict), f"{run_dir.name} lacks DAGKV Git capture")
    dagkv_head = _git_head(
        dagkv_git.get("head"), label=f"{run_dir.name} DAGKV Git HEAD"
    )
    return {
        "run_id": validated.run_id,
        "result_sha256": validated.result_sha256,
        "provenance_sha256": validated.provenance_sha256,
        "sha256sums_sha256": validated.sha256sums_sha256,
        "formal_run_manifest_sha256": validated.formal_run_manifest_sha256,
        "frozen_tolerance_sha256": validated.frozen_tolerance_sha256,
        "calibration_manifest_sha256": validated.calibration_manifest_sha256,
        "implementation_manifest_sha256": raw.implementation_manifest_sha256,
        "reproducibility_fingerprint": validated.reproducibility_fingerprint,
        "protocol_sha256": validated.protocol_sha256,
        "observed_max_abs_error": raw.observed_max_abs_error,
        "minimum_top1_margin": raw.minimum_top1_margin,
        "dagkv_git_head": dagkv_head,
        "dagkv_snapshot_sha256": validated.dagkv_snapshot_sha256,
    }


def _expected_command(template: Sequence[str], run_dir: Path) -> list[str]:
    require(
        isinstance(template, list) and template, "runner command template is invalid"
    )
    replaced = [str(run_dir) if value == "<RUN_DIR>" else value for value in template]
    require(
        all(isinstance(value, str) for value in replaced),
        "runner command contains non-string value",
    )
    require(
        replaced.count(str(run_dir)) == 1,
        "runner command output path binding is ambiguous",
    )
    return replaced


def _validate_log(record: Any, expected_path: Path, *, label: str) -> None:
    require(isinstance(record, dict), f"{label} record must be an object")
    require(set(record) == {"path", "size", "sha256"}, f"{label} record fields drifted")
    observed = _file_record(expected_path)
    require(record == observed, f"{label} hash or size drifted")


def _validate_success_outcome(
    submitted: Mapping[str, Any],
    terminal: Mapping[str, Any],
    *,
    previous_terminal_at: datetime,
) -> datetime:
    submitted_at = _timestamp(submitted["timestamp_utc"], label="submission timestamp")
    started_at = _timestamp(terminal["started_at_utc"], label="process start timestamp")
    ended_at = _timestamp(terminal["ended_at_utc"], label="process end timestamp")
    terminal_at = _timestamp(terminal["timestamp_utc"], label="terminal timestamp")
    require(
        previous_terminal_at <= submitted_at <= started_at <= ended_at <= terminal_at,
        "formal attempt timestamps are out of order",
    )
    require(
        type(terminal["pid"]) is int and terminal["pid"] > 0,
        "formal attempt PID is invalid",
    )
    require(terminal["exit_code"] == 0, "formal attempt exit code is nonzero")
    duration = terminal["duration_s"]
    require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration >= 0,
        "formal attempt duration is invalid",
    )
    require(
        terminal["status"] == "passed"
        and terminal["timed_out"] is False
        and terminal["sigterm_sent"] is False
        and terminal["sigkill_sent"] is False
        and terminal["error"] is None,
        "formal attempt terminal is not an unqualified pass",
    )
    return terminal_at


def _read_journal(path: Path) -> tuple[list[dict[str, Any]], bytes, list[bytes]]:
    require(path.is_file() and not path.is_symlink(), f"missing formal journal: {path}")
    require(path.stat().st_nlink == 1, "formal journal must have one hard link")
    raw = path.read_bytes()
    require(
        raw.endswith(b"\n") and b"\r" not in raw,
        "formal journal must use terminated LF rows",
    )
    lines = raw.splitlines(keepends=True)
    require(
        len(lines) == FULL_RECORD_COUNT,
        "formal journal must contain exactly 42 records",
    )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        payload = _decode_json(line, label=f"formal journal row {index}")
        require(
            isinstance(payload, dict), f"formal journal row {index} must be an object"
        )
        rows.append(payload)
    return rows, raw, lines


def _expected_root_entries(
    run_names: Sequence[str], *, include_seal: bool
) -> list[str]:
    entries = {
        PREREGISTRATION_NAME,
        ATTEMPTS_NAME,
        ACCEPTANCE_NAME,
        "aggregate.stdout.log",
        "aggregate.stderr.log",
    }
    if include_seal:
        entries.add(SEAL_NAME)
    for run_name in run_names:
        entries.update({run_name, f"{run_name}.stdout.log", f"{run_name}.stderr.log"})
    return sorted(entries)


def _validate_acceptance(
    root: Path,
    *,
    preregistration: Mapping[str, Any],
    run_validations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    path = root / ACCEPTANCE_NAME
    manifest, raw = _read_json_object(path, label="item-8 acceptance manifest")
    _exact_fields(manifest, ACCEPTANCE_FIELDS, label="item-8 acceptance manifest")
    require(
        manifest["schema_version"] == ACCEPTANCE_SCHEMA
        and manifest["protocol_schema"] == DATA_PLANE_PROTOCOL_SCHEMA,
        "item-8 acceptance schema drifted",
    )
    require(
        manifest["gate_status"] == "M2_ITEM8_ACCEPTED"
        and manifest["run_count"] == FORMAL_RUN_COUNT
        and manifest["passed_run_count"] == FORMAL_RUN_COUNT
        and manifest["m2_item8_accepted"] is True
        and manifest["m2_accepted"] is False
        and manifest["performance_claims_supported"] is False,
        "item-8 acceptance claim boundary drifted",
    )
    parent = preregistration["parent_binding"]
    require(
        manifest["frozen_tolerance_sha256"] == parent["frozen_tolerance_sha256"]
        and manifest["calibration_manifest_sha256"]
        == parent["calibration_manifest_sha256"]
        and manifest["reproducibility_fingerprint"]
        == parent["reproducibility_fingerprint"]
        and manifest["protocol_sha256"]
        == preregistration["data_plane_protocol_sha256"],
        "item-8 acceptance frozen binding drifted",
    )
    require(
        manifest["nvidia_userspace_bundle_root"]
        == parent["nvidia_userspace_bundle_root"]
        and manifest["nvidia_userspace_bundle_manifest_sha256"]
        == parent["expected_nvidia_userspace_bundle_manifest_sha256"]
        and manifest["nvidia_userspace_bundle_content_digest"]
        == parent["expected_nvidia_userspace_bundle_content_digest"]
        and manifest["nvidia_driver_version"]
        == parent["expected_nvidia_driver_version"],
        "item-8 acceptance NVIDIA userspace/driver binding drifted",
    )
    expected_runs = sorted(
        (
            {
                "run_id": row["run_id"],
                "formal_run_manifest_sha256": row["formal_run_manifest_sha256"],
                "result_sha256": row["result_sha256"],
                "provenance_sha256": row["provenance_sha256"],
                "sha256sums_sha256": row["sha256sums_sha256"],
            }
            for row in run_validations
        ),
        key=lambda row: row["run_id"],
    )
    require(isinstance(manifest["runs"], list), "item-8 acceptance runs must be a list")
    for row in manifest["runs"]:
        require(isinstance(row, dict), "item-8 acceptance run entry must be an object")
        _exact_fields(row, ACCEPTANCE_RUN_FIELDS, label="item-8 acceptance run")
    require(manifest["runs"] == expected_runs, "item-8 acceptance run mapping drifted")
    require(
        manifest["statement"] == ACCEPTANCE_STATEMENT,
        "item-8 acceptance statement drifted",
    )
    return manifest, hashlib.sha256(raw).hexdigest()


def replay_formal_bundle(
    campaign_root: Path,
    *,
    expected_preregistration_sha256: str | None = None,
    include_seal: bool = False,
) -> FormalBundleValidation:
    """Replay every formal campaign input and return its deterministic bindings."""

    root = campaign_root.expanduser().resolve()
    require(
        root.is_dir() and not root.is_symlink(),
        f"formal campaign root is missing or unsafe: {root}",
    )
    run_names = [f"run-{index:03d}" for index in range(1, FORMAL_RUN_COUNT + 1)]
    require(
        sorted(entry.name for entry in root.iterdir())
        == _expected_root_entries(run_names, include_seal=include_seal),
        "formal campaign root is not the exact declared closed set",
    )
    input_inventory = _capture_tree_inventory(root, include_seal=include_seal)
    (
        preregistration,
        preregistration_sha,
        parent,
        bundle_before,
    ) = _validate_preregistration(
        root,
        expected_preregistration_sha256=expected_preregistration_sha256,
    )
    rows, journal_raw, journal_lines = _read_journal(root / ATTEMPTS_NAME)
    campaign_id = preregistration["campaign_id"]
    execution_binding = _validate_execution_binding(
        rows[0].get("execution_binding"),
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha,
    )
    preregistered_at = _timestamp(
        preregistration["created_at_utc"], label="preregistration timestamp"
    )
    previous_terminal_at = preregistered_at
    validations: list[dict[str, Any]] = []
    ordered_runs: list[dict[str, Any]] = []

    for sequence, run_name in enumerate(run_names, start=1):
        submitted = rows[(sequence - 1) * 2]
        terminal = rows[(sequence - 1) * 2 + 1]
        _exact_fields(submitted, RUN_SUBMITTED_FIELDS, label=f"{run_name} submitted")
        _exact_fields(terminal, RUN_TERMINAL_FIELDS, label=f"{run_name} terminal")
        attempt_id = f"{campaign_id}:{run_name}"
        for record in (submitted, terminal):
            require(
                record["schema_version"] == ATTEMPT_SCHEMA
                and record["campaign_id"] == campaign_id
                and record["attempt_id"] == attempt_id
                and record["kind"] == "formal_run"
                and record["sequence"] == sequence
                and record["run_name"] == run_name,
                f"{run_name} attempt identity drifted",
            )
        run_dir = root / run_name
        require(submitted["event"] == "submitted", f"{run_name} lacks submitted event")
        require(terminal["event"] == "terminal", f"{run_name} lacks terminal event")
        require(
            submitted["preregistration_sha256"] == preregistration_sha,
            f"{run_name} preregistration binding drifted",
        )
        require(
            submitted["execution_binding"] == execution_binding,
            f"{run_name} execution binding drifted",
        )
        require(
            submitted["output_dir"] == run_name, f"{run_name} output binding drifted"
        )
        require(
            submitted["stdout"] == f"{run_name}.stdout.log"
            and submitted["stderr"] == f"{run_name}.stderr.log",
            f"{run_name} submitted log names drifted",
        )
        require(
            submitted["command"]
            == _expected_command(preregistration["runner_command_template"], run_dir),
            f"{run_name} command drifted",
        )
        previous_terminal_at = _validate_success_outcome(
            submitted, terminal, previous_terminal_at=previous_terminal_at
        )
        _validate_log(
            terminal["stdout"],
            root / f"{run_name}.stdout.log",
            label=f"{run_name} stdout",
        )
        _validate_log(
            terminal["stderr"],
            root / f"{run_name}.stderr.log",
            label=f"{run_name} stderr",
        )
        require(
            terminal["artifact_inventory"] == _artifact_inventory(run_dir),
            f"{run_name} artifact inventory drifted",
        )
        validation = _validate_formal_run(run_dir)
        require(
            set(validation) == RUN_VALIDATION_FIELDS,
            f"{run_name} replay fields drifted",
        )
        require(
            terminal["validation"] == validation,
            f"{run_name} terminal validation drifted",
        )
        require(
            validation["implementation_manifest_sha256"]
            == preregistration["expected_implementation_manifest_sha256"],
            f"{run_name} implementation drifted",
        )
        require(
            validation["reproducibility_fingerprint"]
            == preregistration["expected_reproducibility_fingerprint"],
            f"{run_name} fingerprint drifted",
        )
        require(
            validation["frozen_tolerance_sha256"]
            == preregistration["parent_binding"]["frozen_tolerance_sha256"],
            f"{run_name} tolerance binding drifted",
        )
        require(
            validation["calibration_manifest_sha256"]
            == preregistration["parent_binding"]["calibration_manifest_sha256"],
            f"{run_name} calibration binding drifted",
        )
        require(
            validation["protocol_sha256"]
            == preregistration["data_plane_protocol_sha256"],
            f"{run_name} protocol binding drifted",
        )
        require(
            validation["dagkv_git_head"] == execution_binding["execution_git_head"],
            f"{run_name} DAGKV execution HEAD drifted",
        )
        validations.append(validation)
        ordered_runs.append(
            {
                "sequence": sequence,
                "run_name": run_name,
                "attempt_id": attempt_id,
                "run_id": validation["run_id"],
                "result_sha256": validation["result_sha256"],
                "provenance_sha256": validation["provenance_sha256"],
                "sha256sums_sha256": validation["sha256sums_sha256"],
                "formal_run_manifest_sha256": validation["formal_run_manifest_sha256"],
                "dagkv_git_head": validation["dagkv_git_head"],
                "dagkv_snapshot_sha256": validation["dagkv_snapshot_sha256"],
            }
        )

    run_ids = {row["run_id"] for row in ordered_runs}
    require(len(run_ids) == FORMAL_RUN_COUNT, "formal run IDs must be unique")
    require(
        run_ids.isdisjoint(parent.calibration_run_ids),
        "formal and calibration run IDs overlap",
    )
    dagkv_heads = {row["dagkv_git_head"] for row in ordered_runs}
    require(
        dagkv_heads == {execution_binding["execution_git_head"]},
        "formal runs do not share the committed execution HEAD",
    )
    dagkv_snapshots = {row["dagkv_snapshot_sha256"] for row in ordered_runs}
    require(
        len(dagkv_snapshots) == 1,
        "formal runs do not share one DAGKV snapshot",
    )
    dagkv_snapshot_sha256 = next(iter(dagkv_snapshots))
    prefix_raw = b"".join(journal_lines[:RUN_RECORD_COUNT])
    prefix = {
        "prefix_bytes": len(prefix_raw),
        "prefix_record_count": RUN_RECORD_COUNT,
        "prefix_sha256": hashlib.sha256(prefix_raw).hexdigest(),
    }
    aggregate_submitted = rows[RUN_RECORD_COUNT]
    aggregate_terminal = rows[RUN_RECORD_COUNT + 1]
    _exact_fields(
        aggregate_submitted, AGGREGATE_SUBMITTED_FIELDS, label="aggregate submitted"
    )
    _exact_fields(aggregate_terminal, TERMINAL_BASE_FIELDS, label="aggregate terminal")
    aggregate_id = f"{campaign_id}:aggregate"
    for record in (aggregate_submitted, aggregate_terminal):
        require(
            record["schema_version"] == ATTEMPT_SCHEMA
            and record["campaign_id"] == campaign_id
            and record["attempt_id"] == aggregate_id
            and record["kind"] == "aggregate",
            "formal aggregate identity drifted",
        )
    require(
        aggregate_submitted["event"] == "submitted"
        and aggregate_submitted["command"] == preregistration["aggregate_command"]
        and aggregate_submitted["output"] == ACCEPTANCE_NAME
        and aggregate_submitted["stdout"] == "aggregate.stdout.log"
        and aggregate_submitted["stderr"] == "aggregate.stderr.log"
        and aggregate_submitted["preregistration_sha256"] == preregistration_sha
        and aggregate_submitted["execution_binding"] == execution_binding
        and aggregate_submitted["formal_prefix"] == prefix,
        "formal aggregate submission drifted",
    )
    previous_terminal_at = _validate_success_outcome(
        aggregate_submitted,
        aggregate_terminal,
        previous_terminal_at=previous_terminal_at,
    )
    del previous_terminal_at
    require(
        aggregate_terminal["event"] == "terminal",
        "formal aggregate lacks terminal event",
    )
    _validate_log(
        aggregate_terminal["stdout"],
        root / "aggregate.stdout.log",
        label="aggregate stdout",
    )
    _validate_log(
        aggregate_terminal["stderr"],
        root / "aggregate.stderr.log",
        label="aggregate stderr",
    )
    require(
        aggregate_terminal["artifact_inventory"]
        == _artifact_inventory(root / ACCEPTANCE_NAME),
        "aggregate acceptance inventory drifted",
    )
    acceptance, acceptance_sha = _validate_acceptance(
        root, preregistration=preregistration, run_validations=validations
    )
    expected_aggregate_validation = {
        "acceptance_sha256": acceptance_sha,
        "run_count": FORMAL_RUN_COUNT,
        "gate_status": "M2_ITEM8_ACCEPTED",
        "reproducibility_fingerprint": preregistration[
            "expected_reproducibility_fingerprint"
        ],
    }
    require(
        aggregate_terminal["validation"] == expected_aggregate_validation,
        "aggregate terminal acceptance binding drifted",
    )
    require(
        acceptance["run_count"] == len(ordered_runs),
        "acceptance ordered run count drifted",
    )
    require(
        sorted(entry.name for entry in root.iterdir())
        == _expected_root_entries(run_names, include_seal=include_seal),
        "formal campaign root changed during replay",
    )
    require(
        _capture_tree_inventory(root, include_seal=include_seal) == input_inventory,
        "formal bundle changed during replay",
    )
    require(
        _sha256_file(root / PREREGISTRATION_NAME) == preregistration_sha,
        "formal preregistration changed during replay",
    )
    require(
        (root / ATTEMPTS_NAME).read_bytes() == journal_raw,
        "formal journal changed during replay",
    )
    require(
        _sha256_file(root / ACCEPTANCE_NAME) == acceptance_sha,
        "formal acceptance changed during replay",
    )
    _verify_frozen_files(preregistration)
    require(
        _validate_parent(preregistration) == parent,
        "formal parent evidence changed during replay",
    )
    require(
        _validate_calibration_nvidia_binding(preregistration)
        == {
            field: preregistration[field]
            for field in (
                "nvidia_userspace_bundle_root",
                "expected_nvidia_userspace_bundle_manifest_sha256",
                "expected_nvidia_userspace_bundle_content_digest",
                "expected_nvidia_driver_version",
            )
        },
        "calibration NVIDIA binding changed during formal replay",
    )
    require(
        _validate_nvidia_bundle_binding(preregistration) == bundle_before,
        "NVIDIA userspace bundle changed during formal evidence replay",
    )
    return FormalBundleValidation(
        campaign_root=root,
        campaign_id=campaign_id,
        preregistration_sha256=preregistration_sha,
        formal_prefix_bytes=len(prefix_raw),
        formal_prefix_sha256=prefix["prefix_sha256"],
        full_journal_bytes=len(journal_raw),
        full_journal_sha256=hashlib.sha256(journal_raw).hexdigest(),
        acceptance_sha256=acceptance_sha,
        calibration_manifest_sha256=preregistration["parent_binding"][
            "calibration_manifest_sha256"
        ],
        frozen_tolerance_sha256=preregistration["parent_binding"][
            "frozen_tolerance_sha256"
        ],
        implementation_manifest_sha256=preregistration[
            "expected_implementation_manifest_sha256"
        ],
        reproducibility_fingerprint=preregistration[
            "expected_reproducibility_fingerprint"
        ],
        nvidia_bundle_validation=bundle_before,
        execution_binding=execution_binding,
        dagkv_snapshot_sha256=dagkv_snapshot_sha256,
        ordered_runs=tuple(ordered_runs),
        input_inventory=input_inventory,
    )


def _seal_payload(
    validation: FormalBundleValidation, *, sealed_at_utc: str
) -> dict[str, Any]:
    return {
        "schema_version": SEAL_SCHEMA,
        "campaign_id": validation.campaign_id,
        "sealed_at_utc": sealed_at_utc,
        "formal_campaign_preregistration_file": PREREGISTRATION_NAME,
        "formal_campaign_preregistration_sha256": validation.preregistration_sha256,
        "formal_attempts_file": ATTEMPTS_NAME,
        "formal_prefix_bytes": validation.formal_prefix_bytes,
        "formal_prefix_record_count": RUN_RECORD_COUNT,
        "formal_prefix_sha256": validation.formal_prefix_sha256,
        "full_journal_bytes": validation.full_journal_bytes,
        "full_journal_record_count": FULL_RECORD_COUNT,
        "full_journal_sha256": validation.full_journal_sha256,
        "acceptance_file": ACCEPTANCE_NAME,
        "acceptance_sha256": validation.acceptance_sha256,
        "calibration_manifest_sha256": validation.calibration_manifest_sha256,
        "frozen_tolerance_sha256": validation.frozen_tolerance_sha256,
        "implementation_manifest_sha256": validation.implementation_manifest_sha256,
        "reproducibility_fingerprint": validation.reproducibility_fingerprint,
        "nvidia_userspace_bundle_root": str(
            validation.nvidia_bundle_validation.runtime.rootfs.parent
        ),
        "nvidia_userspace_bundle_manifest_sha256": (
            validation.nvidia_bundle_validation.manifest_sha256
        ),
        "nvidia_userspace_bundle_content_digest": (
            validation.nvidia_bundle_validation.content_digest
        ),
        "nvidia_driver_version": (
            validation.nvidia_bundle_validation.kernel_module_version
        ),
        "execution_binding": dict(validation.execution_binding),
        "dagkv_snapshot_sha256": validation.dagkv_snapshot_sha256,
        "run_count": FORMAL_RUN_COUNT,
        "ordered_runs": list(validation.ordered_runs),
        "m2_item8_accepted": True,
        "m2_accepted": False,
        "performance_claims_supported": False,
        "statement": SEAL_STATEMENT,
    }


def _publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_formal_bundle_seal(
    campaign_root: Path,
    *,
    expected_preregistration_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Replay a seal-free production bundle and create its sole bundle seal."""

    root = campaign_root.expanduser().resolve()
    destination = root / SEAL_NAME
    require(
        not os.path.lexists(destination),
        f"formal bundle seal already exists: {destination}",
    )
    validation = replay_formal_bundle(
        root,
        expected_preregistration_sha256=expected_preregistration_sha256,
        include_seal=False,
    )
    require(
        _capture_tree_inventory(root, include_seal=False) == validation.input_inventory,
        "formal bundle changed before seal publication",
    )
    require(
        _sha256_file(root / PREREGISTRATION_NAME) == validation.preregistration_sha256,
        "formal preregistration changed before seal publication",
    )
    journal_raw = (root / ATTEMPTS_NAME).read_bytes()
    require(
        len(journal_raw) == validation.full_journal_bytes
        and hashlib.sha256(journal_raw).hexdigest() == validation.full_journal_sha256,
        "formal journal changed before seal publication",
    )
    require(
        _sha256_file(root / ACCEPTANCE_NAME) == validation.acceptance_sha256,
        "formal acceptance changed before seal publication",
    )
    preregistration, _ = _read_json_object(
        root / PREREGISTRATION_NAME,
        label="formal preregistration before seal publication",
    )
    require(
        _validate_nvidia_bundle_binding(preregistration)
        == validation.nvidia_bundle_validation,
        "NVIDIA userspace bundle changed before formal seal publication",
    )
    payload = _seal_payload(validation, sealed_at_utc=datetime.now(UTC).isoformat())
    _publish_json_exclusive(destination, payload)
    seal_sha = _sha256_file(destination)
    published, observed_sha, _ = validate_published_formal_bundle(
        destination,
        expected_seal_sha256=seal_sha,
        expected_preregistration_sha256=expected_preregistration_sha256,
    )
    require(
        published == payload and observed_sha == seal_sha,
        "freshly published formal seal did not replay exactly",
    )
    return payload, seal_sha


def validate_published_formal_bundle(
    seal_path: Path,
    *,
    expected_seal_sha256: str | None = None,
    expected_preregistration_sha256: str | None = None,
) -> tuple[dict[str, Any], str, FormalBundleValidation]:
    """Replay a published formal bundle and require its seal to match exactly."""

    seal_path = seal_path.expanduser().resolve()
    require(seal_path.name == SEAL_NAME, f"formal seal must be named {SEAL_NAME}")
    seal, raw = _read_json_object(seal_path, label="formal bundle seal")
    _exact_fields(seal, SEAL_FIELDS, label="formal bundle seal")
    seal_sha = hashlib.sha256(raw).hexdigest()
    if expected_seal_sha256 is not None:
        require(
            seal_sha
            == _lower_sha256(expected_seal_sha256, label="expected formal seal"),
            "formal bundle seal SHA-256 differs",
        )
    preregistration_sha = (
        expected_preregistration_sha256
        or seal["formal_campaign_preregistration_sha256"]
    )
    validation = replay_formal_bundle(
        seal_path.parent,
        expected_preregistration_sha256=preregistration_sha,
        include_seal=True,
    )
    require(
        seal_path.read_bytes() == raw,
        "formal bundle seal changed during replay",
    )
    _timestamp(seal["sealed_at_utc"], label="formal bundle seal timestamp")
    expected = _seal_payload(validation, sealed_at_utc=seal["sealed_at_utc"])
    require(seal == expected, "formal bundle seal does not match replayed evidence")
    require(
        isinstance(seal["execution_binding"], dict),
        "formal seal execution binding must be an object",
    )
    _exact_fields(
        seal["execution_binding"],
        EXECUTION_BINDING_FIELDS,
        label="formal seal execution binding",
    )
    for row in seal["ordered_runs"]:
        require(isinstance(row, dict), "formal seal ordered run must be an object")
        _exact_fields(row, SEAL_RUN_FIELDS, label="formal seal ordered run")
    return seal, seal_sha, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-preregistration-sha256")
    parser.add_argument("--expected-seal-sha256")
    parser.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.publish:
            if args.expected_preregistration_sha256 is None:
                raise FormalEvidenceError(
                    "--publish requires --expected-preregistration-sha256"
                )
            payload, seal_sha = publish_formal_bundle_seal(
                args.path,
                expected_preregistration_sha256=args.expected_preregistration_sha256,
            )
            print(
                "M2 formal bundle sealed: "
                f"{args.path.resolve() / SEAL_NAME} sha256={seal_sha} "
                f"runs={payload['run_count']}"
            )
        else:
            seal, seal_sha, _ = validate_published_formal_bundle(
                args.path,
                expected_seal_sha256=args.expected_seal_sha256,
                expected_preregistration_sha256=args.expected_preregistration_sha256,
            )
            print(
                f"M2 formal bundle replay passed: {seal['campaign_id']} "
                f"sha256={seal_sha}"
            )
    except (FormalEvidenceError, OSError) as exc:
        print(f"M2 formal evidence validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
