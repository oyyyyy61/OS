#!/usr/bin/env python3
"""Launch the frozen 20-process M2 item-8 formal holdout campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from tools.run_m2_calibration_campaign import (  # noqa: E402
    CalibrationCampaignError as ProcessSupervisorError,
)
from tools.run_m2_calibration_campaign import (  # noqa: E402
    ProcessExecutionInterrupted,
    ProcessOutcome,
    _run_process,
    _runner_environment,
)

CAMPAIGN_SCHEMA = "dagkv.m2.formal_campaign_preregistration.v2"
ATTEMPT_SCHEMA = "dagkv.m2.formal_campaign_attempt.v1"
FORMAL_CAMPAIGN_PROTOCOL_SCHEMA = "dagkv.m2.formal_campaign.v1"
DATA_PLANE_PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v3"
ACCEPTANCE_SCHEMA = "dagkv.m2.item8.acceptance.v2"
PRODUCTION_RUN_COUNT = 20
FORMAL_ATOL = 0.125
FORMAL_RTOL = 0.0
MARKER_SCHEMA = "dagkv.m2.formal_launch_marker.v1"
LAUNCH_MARKER_REPOSITORY_PATH = "evidence/m2/FORMAL_LAUNCH_MARKER.json"
MARKER_CLAIM_SCOPE = "M2_ITEM8_CORRECTNESS_ONLY_NO_PERFORMANCE_CLAIM"

LAUNCHER_PATH = Path(__file__).resolve()
DEFAULT_RUNNER = REPO_ROOT / "tools" / "run_m2_vllm_abba.py"
DEFAULT_AGGREGATOR = REPO_ROOT / "tools" / "aggregate_m2_formal.py"
RAW_REPLAY_PATH = REPO_ROOT / "tools" / "m2_raw_replay.py"
EVIDENCE_VALIDATOR_PATH = REPO_ROOT / "tools" / "m2_calibration_evidence.py"
FORMAL_EVIDENCE_PATH = REPO_ROOT / "tools" / "m2_formal_evidence.py"
PROCESS_SUPERVISOR_PATH = REPO_ROOT / "tools" / "run_m2_calibration_campaign.py"
NVIDIA_BUNDLE_VALIDATOR_PATH = REPO_ROOT / "tools" / "nvidia_driver_userspace_bundle.py"
DATA_PLANE_PROTOCOL_PATH = (
    REPO_ROOT / "research" / "protocols" / "M2_VLLM_REPLAY_PROTOCOL.md"
)
FORMAL_CAMPAIGN_PROTOCOL_PATH = (
    REPO_ROOT / "research" / "protocols" / "M2_FORMAL_CAMPAIGN_PROTOCOL.md"
)
DEFAULT_PYTHON = Path("/home/data/25_oyzx/Agentrix/vllm/.venv/bin/python")
DEFAULT_MODEL = Path("/home/data/25_oyzx/moqae_runtime_gpu/modelscope/Qwen/Qwen3-8B")
DEFAULT_VLLM_ROOT = Path("/home/data/25_oyzx/Agentrix/vllm")
LOADER_INJECTION_VARIABLES = ("LD_AUDIT", "LD_PRELOAD")

PREREGISTRATION_NAME = "FORMAL_CAMPAIGN_PREREGISTRATION.json"
ATTEMPTS_NAME = "FORMAL_ATTEMPTS.jsonl"
ACCEPTANCE_NAME = "M2_ITEM8_ACCEPTANCE_MANIFEST.json"
FORMAL_RUN_MANIFEST = "M2_ITEM8_FORMAL_RUN_MANIFEST.json"
BUNDLE_SEAL_NAME = "M2_FORMAL_BUNDLE_SEAL.json"
ACCEPTANCE_STATEMENT = (
    "Exactly twenty frozen M2 item 8 holdouts passed. This closes item 8 only; "
    "the aggregate M2 gate remains open and this evidence supports no latency, "
    "throughput, hit-rate, scheduling-policy, or paper-performance claim."
)


class FormalCampaignError(RuntimeError):
    """Raised when a formal campaign cannot continue under its freeze."""


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Immutable inputs for one non-resumable formal holdout campaign."""

    campaign_root: Path
    calibration_manifest: Path
    frozen_tolerance: Path
    expected_implementation_manifest_sha256: str
    expected_reproducibility_fingerprint: str
    nvidia_userspace_bundle_root: Path
    expected_nvidia_userspace_bundle_manifest_sha256: str
    expected_nvidia_userspace_bundle_content_digest: str
    expected_nvidia_driver_version: str
    python_executable: Path = DEFAULT_PYTHON
    runner: Path = DEFAULT_RUNNER
    aggregator: Path = DEFAULT_AGGREGATOR
    model: Path = DEFAULT_MODEL
    vllm_root: Path = DEFAULT_VLLM_ROOT
    cpu_bytes: int = 1 << 30
    runner_timeout_s: float = 60.0
    process_timeout_s: float = 1800.0
    aggregation_timeout_s: float = 900.0
    terminate_grace_s: float = 30.0
    kill_wait_s: float = 30.0
    cuda_device: int = 0


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalCampaignError(message)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, *, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _fresh_validate_nvidia_bundle(config: CampaignConfig) -> BundleValidation:
    """Rebuild and bind the exact NVIDIA userspace/loaded-driver contract."""

    injected = [name for name in LOADER_INJECTION_VARIABLES if name in os.environ]
    _require(
        not injected,
        f"loader injection environment is forbidden: {','.join(injected)}",
    )
    try:
        validation = validate_bundle(
            config.nvidia_userspace_bundle_root,
            expected_manifest_sha256=(
                config.expected_nvidia_userspace_bundle_manifest_sha256
            ),
        )
    except (NvidiaUserspaceBundleError, OSError) as exc:
        raise FormalCampaignError(
            f"NVIDIA userspace bundle validation failed: {exc}"
        ) from exc
    _require(
        validation.manifest_sha256
        == config.expected_nvidia_userspace_bundle_manifest_sha256,
        "NVIDIA userspace bundle manifest digest drifted",
    )
    _require(
        validation.content_digest
        == config.expected_nvidia_userspace_bundle_content_digest,
        "NVIDIA userspace bundle content digest drifted",
    )
    _require(
        validation.kernel_module_version == config.expected_nvidia_driver_version,
        "NVIDIA kernel driver version drifted",
    )
    library_path = str(validation.library_path)
    _require(
        library_path and ":" not in library_path,
        "NVIDIA bundle library path cannot be represented in LD_LIBRARY_PATH",
    )
    return validation


def _validate_git_head(value: str, *, label: str) -> str:
    _require(
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
        raise FormalCampaignError(
            f"Git command failed ({' '.join(arguments)}): {stderr}"
        ) from exc
    return completed.stdout


def _clean_git_head(*, phase: str) -> str:
    root = _git_bytes("rev-parse", "--show-toplevel").decode("utf-8").strip()
    _require(Path(root).resolve() == REPO_ROOT.resolve(), "DAGKV Git root drifted")
    status = _git_bytes("status", "--porcelain=v1", "--untracked-files=all")
    _require(not status, f"DAGKV worktree must be clean during {phase}")
    head = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    return _validate_git_head(head, label=f"{phase} Git HEAD")


def _preparation_git_head() -> str:
    return _clean_git_head(phase="formal preparation")


def _marker_payload(
    marker_bytes: bytes,
    *,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in payload, f"duplicate formal marker key: {key}")
            payload[key] = value
        return payload

    def reject_constant(value: str) -> None:
        raise FormalCampaignError(f"non-finite formal marker value: {value}")

    try:
        marker = json.loads(
            marker_bytes.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalCampaignError(
            f"formal launch marker is invalid JSON: {exc}"
        ) from exc
    _require(isinstance(marker, dict), "formal launch marker must be a JSON object")
    _require(set(marker) == MARKER_FIELDS, "formal launch marker fields drifted")
    _require(marker.get("schema_version") == MARKER_SCHEMA, "marker schema drifted")
    _require(
        marker.get("campaign_id") == preregistration.get("campaign_id")
        and marker.get("campaign_root") == preregistration.get("campaign_root")
        and marker.get("campaign_preregistration_sha256") == preregistration_sha256
        and marker.get("preparation_git_head")
        == preregistration.get("preparation_git_head")
        and marker.get("claim_scope") == MARKER_CLAIM_SCOPE,
        "formal launch marker binding drifted",
    )
    try:
        marker_created = datetime.fromisoformat(str(marker.get("created_at_utc")))
        preregistered = datetime.fromisoformat(str(preregistration["created_at_utc"]))
    except (KeyError, ValueError) as exc:
        raise FormalCampaignError("formal launch marker timestamp is invalid") from exc
    _require(
        marker_created.tzinfo is not None
        and preregistered.tzinfo is not None
        and marker_created >= preregistered,
        "formal launch marker must postdate preregistration",
    )
    return marker


def _establish_execution_binding(
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
) -> dict[str, str]:
    execution_head = _clean_git_head(phase="formal execution")
    preparation_head = _validate_git_head(
        preregistration.get("preparation_git_head"),
        label="preparation Git HEAD",
    )
    parent_row = (
        _git_bytes("rev-list", "--parents", "-n", "1", execution_head)
        .decode("ascii")
        .split()
    )
    _require(
        parent_row == [execution_head, preparation_head],
        "execution HEAD must be the direct single-parent child of preparation HEAD",
    )
    changed_paths = (
        _git_bytes("diff-tree", "--no-commit-id", "--name-only", "-r", execution_head)
        .decode("utf-8")
        .splitlines()
    )
    _require(
        changed_paths == [LAUNCH_MARKER_REPOSITORY_PATH],
        "execution commit must modify only the formal launch marker",
    )
    marker_path = REPO_ROOT / LAUNCH_MARKER_REPOSITORY_PATH
    _require(
        marker_path.is_file()
        and not marker_path.is_symlink()
        and marker_path.stat().st_nlink == 1,
        "formal launch marker worktree file is missing or unsafe",
    )
    worktree_bytes = marker_path.read_bytes()
    object_bytes = _git_bytes(
        "cat-file", "blob", f"{execution_head}:{LAUNCH_MARKER_REPOSITORY_PATH}"
    )
    _require(
        worktree_bytes == object_bytes,
        "formal launch marker differs from the committed Git object",
    )
    _marker_payload(
        worktree_bytes,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
    )
    return {
        "preparation_git_head": preparation_head,
        "execution_git_head": execution_head,
        "launch_marker_repository_path": LAUNCH_MARKER_REPOSITORY_PATH,
        "launch_marker_sha256": hashlib.sha256(worktree_bytes).hexdigest(),
    }


def _revalidate_execution_binding(
    expected: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
) -> None:
    _require(
        set(expected) == EXECUTION_BINDING_FIELDS,
        "formal execution binding fields drifted",
    )
    observed = _establish_execution_binding(preregistration, preregistration_sha256)
    _require(observed == expected, "formal execution Git binding drifted")


def _test_execution_binding(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preparation_git_head": preregistration.get("preparation_git_head"),
        "execution_git_head": None,
        "launch_marker_repository_path": LAUNCH_MARKER_REPOSITORY_PATH,
        "launch_marker_sha256": None,
    }


@contextmanager
def _campaign_execution_lock(root: Path) -> Iterator[None]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    descriptor = os.open(root, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FormalCampaignError(
                "formal campaign is already executing under another launcher"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_attempt(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.is_symlink(), f"attempt journal cannot be a symlink: {path}")
    existed = path.exists()
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_directory(path.parent)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalCampaignError(f"invalid {label} at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _resolved_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{label} is missing or unsafe: {resolved}",
    )
    return resolved


def _command_path(path: Path, *, label: str) -> Path:
    absolute = path.expanduser().absolute()
    _require(absolute.is_file(), f"{label} is missing: {absolute}")
    return absolute


def _frozen_file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_frozen_files(entries: Mapping[str, Mapping[str, Any]]) -> None:
    for label, entry in entries.items():
        path = Path(str(entry.get("path")))
        _require(
            path.is_file() and not path.is_symlink(),
            f"frozen {label} file vanished or became unsafe: {path}",
        )
        _require(
            path.stat().st_size == entry.get("size")
            and _sha256_file(path) == entry.get("sha256"),
            f"frozen {label} file changed during campaign: {path}",
        )


def _artifact_inventory(path: Path) -> list[dict[str, Any]]:
    if path.is_file() and not path.is_symlink():
        return [{"path": path.name, "type": "file", "size": path.stat().st_size}]
    if not path.is_dir():
        return []
    inventory: list[dict[str, Any]] = []
    for artifact in sorted(path.rglob("*")):
        relative = artifact.relative_to(path).as_posix()
        if artifact.is_symlink():
            inventory.append({"path": relative, "type": "symlink"})
        elif artifact.is_file():
            inventory.append(
                {"path": relative, "type": "file", "size": artifact.stat().st_size}
            )
        elif artifact.is_dir():
            inventory.append({"path": relative, "type": "directory"})
    return inventory


def _current_implementation_manifest_sha256() -> str:
    try:
        from tools.run_m2_vllm_abba import _implementation_capture  # noqa: PLC0415

        value = _implementation_capture().get("manifest_sha256")
    except (ImportError, OSError, AttributeError) as exc:
        raise FormalCampaignError(
            f"cannot capture the current M2 implementation: {exc}"
        ) from exc
    return _validate_digest(value, label="current implementation manifest")


def _validate_parent_inputs(
    calibration_manifest: Path,
    frozen_tolerance: Path,
    *,
    expected_implementation_manifest_sha256: str,
    expected_reproducibility_fingerprint: str,
    expected_nvidia_userspace_bundle_root: Path,
    expected_nvidia_userspace_bundle_manifest_sha256: str,
    expected_nvidia_userspace_bundle_content_digest: str,
    expected_nvidia_driver_version: str,
) -> dict[str, Any]:
    try:
        from tools.aggregate_m2_formal import (  # noqa: PLC0415
            FormalAggregationError,
            _validate_parent_evidence,
        )

        parent = _validate_parent_evidence(calibration_manifest, frozen_tolerance)
    except (FormalAggregationError, OSError) as exc:
        raise FormalCampaignError(f"invalid formal parent evidence: {exc}") from exc

    calibration = _read_json_object(calibration_manifest, label="calibration manifest")
    _require(
        calibration.get("implementation_manifest_sha256")
        == expected_implementation_manifest_sha256,
        "calibration implementation manifest differs from the formal freeze",
    )
    _require(
        parent.reproducibility_fingerprint == expected_reproducibility_fingerprint,
        "calibration/tolerance fingerprint differs from the formal freeze",
    )
    preregistration_name = calibration.get("campaign_preregistration_file")
    _require(
        preregistration_name == "CAMPAIGN_PREREGISTRATION.json",
        "calibration preregistration filename drifted",
    )
    calibration_preregistration = _read_json_object(
        calibration_manifest.parent / preregistration_name,
        label="calibration campaign preregistration",
    )
    expected_bundle_binding = {
        "nvidia_userspace_bundle_root": str(expected_nvidia_userspace_bundle_root),
        "expected_nvidia_userspace_bundle_manifest_sha256": (
            expected_nvidia_userspace_bundle_manifest_sha256
        ),
        "expected_nvidia_userspace_bundle_content_digest": (
            expected_nvidia_userspace_bundle_content_digest
        ),
        "expected_nvidia_driver_version": expected_nvidia_driver_version,
    }
    _require(
        all(
            calibration_preregistration.get(field) == value
            for field, value in expected_bundle_binding.items()
        ),
        "calibration NVIDIA userspace/driver binding differs from the formal freeze",
    )
    return {
        "calibration_manifest_sha256": parent.calibration_manifest_sha256,
        "frozen_tolerance_sha256": parent.frozen_tolerance_sha256,
        "reproducibility_fingerprint": parent.reproducibility_fingerprint,
        "frozen_at_utc": parent.frozen_at_utc.isoformat(),
        "calibration_run_count": len(parent.calibration_run_ids),
        **expected_bundle_binding,
    }


def _validate_expected_runs(expected_runs: int) -> None:
    _require(
        type(expected_runs) is int and 0 < expected_runs <= PRODUCTION_RUN_COUNT,
        f"expected run count must be in [1, {PRODUCTION_RUN_COUNT}]",
    )


def _selection_rule(run_names: Sequence[str]) -> dict[str, Any]:
    return {
        "attempts_per_run": 1,
        "eligible_kind": "formal_run",
        "ordered_run_names": list(run_names),
        "required_events": ["submitted", "terminal"],
        "required_terminal_status": "passed",
        "retry_count": 0,
        "stop_on_first_failure": True,
    }


def _normalized_config(
    config: CampaignConfig,
) -> tuple[
    CampaignConfig,
    dict[str, dict[str, Any]],
    dict[str, Any],
    BundleValidation,
]:
    expected_implementation = _validate_digest(
        config.expected_implementation_manifest_sha256,
        label="expected implementation manifest",
    )
    expected_fingerprint = _validate_digest(
        config.expected_reproducibility_fingerprint,
        label="expected reproducibility fingerprint",
    )
    expected_bundle_manifest = _validate_digest(
        config.expected_nvidia_userspace_bundle_manifest_sha256,
        label="expected NVIDIA userspace bundle manifest",
    )
    expected_bundle_content = _validate_digest(
        config.expected_nvidia_userspace_bundle_content_digest,
        label="expected NVIDIA userspace bundle content",
    )
    _require(
        isinstance(config.expected_nvidia_driver_version, str)
        and config.expected_nvidia_driver_version.strip()
        == config.expected_nvidia_driver_version
        and config.expected_nvidia_driver_version,
        "expected NVIDIA driver version must be a non-empty canonical string",
    )
    _require(
        _current_implementation_manifest_sha256() == expected_implementation,
        "current M2 implementation differs from the expected formal freeze",
    )
    _require(
        type(config.cpu_bytes) is int and config.cpu_bytes > 0,
        "cpu_bytes must be a positive integer",
    )
    for label, value in (
        ("runner_timeout_s", config.runner_timeout_s),
        ("process_timeout_s", config.process_timeout_s),
        ("aggregation_timeout_s", config.aggregation_timeout_s),
        ("terminate_grace_s", config.terminate_grace_s),
        ("kill_wait_s", config.kill_wait_s),
    ):
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0,
            f"{label} must be positive",
        )
    _require(
        type(config.cuda_device) is int and config.cuda_device >= 0,
        "cuda_device must be a non-negative integer",
    )

    runner = _resolved_file(config.runner, label="runner")
    aggregator = _resolved_file(config.aggregator, label="formal aggregator")
    raw_replay = _resolved_file(RAW_REPLAY_PATH, label="raw replay validator")
    evidence = _resolved_file(
        EVIDENCE_VALIDATOR_PATH, label="calibration evidence validator"
    )
    formal_evidence = _resolved_file(
        FORMAL_EVIDENCE_PATH, label="formal evidence validator"
    )
    process_supervisor = _resolved_file(
        PROCESS_SUPERVISOR_PATH, label="process supervisor"
    )
    nvidia_bundle_validator = _resolved_file(
        NVIDIA_BUNDLE_VALIDATOR_PATH,
        label="NVIDIA userspace bundle validator",
    )
    data_plane_protocol = _resolved_file(
        DATA_PLANE_PROTOCOL_PATH, label="M2 data-plane protocol"
    )
    formal_protocol = _resolved_file(
        FORMAL_CAMPAIGN_PROTOCOL_PATH, label="formal campaign protocol"
    )
    launcher = _resolved_file(LAUNCHER_PATH, label="formal launcher")
    calibration_manifest = _resolved_file(
        config.calibration_manifest, label="calibration manifest"
    )
    frozen_tolerance = _resolved_file(config.frozen_tolerance, label="frozen tolerance")
    python_executable = _command_path(
        config.python_executable, label="Python executable"
    )
    model = config.model.expanduser().resolve()
    vllm_root = config.vllm_root.expanduser().resolve()
    campaign_root = config.campaign_root.expanduser().resolve()
    bundle_root = Path(
        os.path.abspath(config.nvidia_userspace_bundle_root.expanduser())
    )
    _require(model.is_dir(), f"model directory is missing: {model}")
    _require(vllm_root.is_dir(), f"vLLM root is missing: {vllm_root}")
    calibration_root = calibration_manifest.parent.resolve()
    for protected in (
        REPO_ROOT.resolve(),
        model,
        vllm_root,
        calibration_root,
        bundle_root,
    ):
        _require(
            campaign_root != protected and not campaign_root.is_relative_to(protected),
            f"formal campaign root is inside protected input root: {protected}",
        )
    _require(
        frozen_tolerance.parent.resolve() != campaign_root,
        "frozen tolerance must remain outside the formal campaign root",
    )

    normalized = CampaignConfig(
        campaign_root=campaign_root,
        calibration_manifest=calibration_manifest,
        frozen_tolerance=frozen_tolerance,
        expected_implementation_manifest_sha256=expected_implementation,
        expected_reproducibility_fingerprint=expected_fingerprint,
        nvidia_userspace_bundle_root=bundle_root,
        expected_nvidia_userspace_bundle_manifest_sha256=expected_bundle_manifest,
        expected_nvidia_userspace_bundle_content_digest=expected_bundle_content,
        expected_nvidia_driver_version=config.expected_nvidia_driver_version,
        python_executable=python_executable,
        runner=runner,
        aggregator=aggregator,
        model=model,
        vllm_root=vllm_root,
        cpu_bytes=config.cpu_bytes,
        runner_timeout_s=float(config.runner_timeout_s),
        process_timeout_s=float(config.process_timeout_s),
        aggregation_timeout_s=float(config.aggregation_timeout_s),
        terminate_grace_s=float(config.terminate_grace_s),
        kill_wait_s=float(config.kill_wait_s),
        cuda_device=config.cuda_device,
    )
    bundle_validation = _fresh_validate_nvidia_bundle(normalized)
    _require(
        bundle_validation.runtime.rootfs.parent == bundle_root,
        "validated NVIDIA bundle rootfs binding drifted",
    )
    frozen_files = {
        "formal_protocol": _frozen_file_entry(formal_protocol),
        "data_plane_protocol": _frozen_file_entry(data_plane_protocol),
        "runner": _frozen_file_entry(runner),
        "launcher": _frozen_file_entry(launcher),
        "formal_aggregator": _frozen_file_entry(aggregator),
        "formal_evidence": _frozen_file_entry(formal_evidence),
        "raw_replay": _frozen_file_entry(raw_replay),
        "calibration_evidence": _frozen_file_entry(evidence),
        "process_supervisor": _frozen_file_entry(process_supervisor),
        "nvidia_bundle_validator": _frozen_file_entry(nvidia_bundle_validator),
        "frozen_tolerance": _frozen_file_entry(frozen_tolerance),
        "calibration_manifest": _frozen_file_entry(calibration_manifest),
    }
    parent_binding = _validate_parent_inputs(
        calibration_manifest,
        frozen_tolerance,
        expected_implementation_manifest_sha256=expected_implementation,
        expected_reproducibility_fingerprint=expected_fingerprint,
        expected_nvidia_userspace_bundle_root=bundle_root,
        expected_nvidia_userspace_bundle_manifest_sha256=expected_bundle_manifest,
        expected_nvidia_userspace_bundle_content_digest=expected_bundle_content,
        expected_nvidia_driver_version=config.expected_nvidia_driver_version,
    )
    _require(
        parent_binding["calibration_manifest_sha256"]
        == frozen_files["calibration_manifest"]["sha256"],
        "validated calibration manifest hash differs from its frozen file hash",
    )
    _require(
        parent_binding["frozen_tolerance_sha256"]
        == frozen_files["frozen_tolerance"]["sha256"],
        "validated tolerance hash differs from its frozen file hash",
    )
    return normalized, frozen_files, parent_binding, bundle_validation


def _run_command(config: CampaignConfig, run_dir: Path) -> list[str]:
    return [
        str(config.python_executable),
        str(config.runner),
        "--output-dir",
        str(run_dir),
        "--mode",
        "formal",
        "--tolerance-file",
        str(config.frozen_tolerance),
        "--calibration-manifest",
        str(config.calibration_manifest),
        "--model",
        str(config.model),
        "--vllm-root",
        str(config.vllm_root),
        "--expected-nvidia-driver-version",
        config.expected_nvidia_driver_version,
        "--nvidia-userspace-bundle-root",
        str(config.nvidia_userspace_bundle_root),
        "--expected-nvidia-userspace-bundle-manifest-sha256",
        config.expected_nvidia_userspace_bundle_manifest_sha256,
        "--expected-nvidia-userspace-bundle-content-digest",
        config.expected_nvidia_userspace_bundle_content_digest,
        "--cpu-bytes",
        str(config.cpu_bytes),
        "--timeout-s",
        str(config.runner_timeout_s),
        "--cuda-device",
        str(config.cuda_device),
        "--full-provenance",
    ]


def _aggregate_command(config: CampaignConfig, acceptance_path: Path) -> list[str]:
    return [
        str(config.python_executable),
        str(config.aggregator),
        "--campaign-dir",
        str(config.campaign_root),
        "--calibration-manifest",
        str(config.calibration_manifest),
        "--frozen-tolerance",
        str(config.frozen_tolerance),
        "--output",
        str(acceptance_path),
    ]


def _preregistration_payload(
    config: CampaignConfig,
    frozen_files: Mapping[str, Mapping[str, Any]],
    parent_binding: Mapping[str, Any],
    bundle_validation: BundleValidation,
    *,
    expected_runs: int,
    preparation_git_head: str | None,
) -> dict[str, Any]:
    run_names = [f"run-{index:03d}" for index in range(1, expected_runs + 1)]
    environment = _runner_environment(
        config.cuda_device,
        nvidia_library_path=bundle_validation.library_path,
    )
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": f"m2-formal-{uuid.uuid4().hex}",
        "campaign_root": str(config.campaign_root),
        "created_at_utc": _utc_now(),
        "preparation_git_head": preparation_git_head,
        "launch_marker_repository_path": LAUNCH_MARKER_REPOSITORY_PATH,
        "formal_campaign_protocol_schema": FORMAL_CAMPAIGN_PROTOCOL_SCHEMA,
        "data_plane_protocol_schema": DATA_PLANE_PROTOCOL_SCHEMA,
        "formal_campaign_protocol_sha256": frozen_files["formal_protocol"]["sha256"],
        "data_plane_protocol_sha256": frozen_files["data_plane_protocol"]["sha256"],
        "expected_runs": expected_runs,
        "production_run_count": PRODUCTION_RUN_COUNT,
        "test_injected_run_count": expected_runs != PRODUCTION_RUN_COUNT,
        "run_names": run_names,
        "formal_attempt_prefix_record_count": expected_runs * 2,
        "selection_rule": _selection_rule(run_names),
        "expected_implementation_manifest_sha256": (
            config.expected_implementation_manifest_sha256
        ),
        "expected_reproducibility_fingerprint": (
            config.expected_reproducibility_fingerprint
        ),
        "nvidia_userspace_bundle_root": str(config.nvidia_userspace_bundle_root),
        "expected_nvidia_userspace_bundle_manifest_sha256": (
            config.expected_nvidia_userspace_bundle_manifest_sha256
        ),
        "expected_nvidia_userspace_bundle_content_digest": (
            config.expected_nvidia_userspace_bundle_content_digest
        ),
        "expected_nvidia_driver_version": config.expected_nvidia_driver_version,
        "parent_binding": dict(parent_binding),
        "frozen_files": dict(frozen_files),
        "python_executable": _frozen_file_entry(config.python_executable),
        "model": str(config.model),
        "vllm_root": str(config.vllm_root),
        "cpu_bytes": config.cpu_bytes,
        "cuda_device": config.cuda_device,
        "runner_timeout_s": config.runner_timeout_s,
        "process_timeout_s": config.process_timeout_s,
        "aggregation_timeout_s": config.aggregation_timeout_s,
        "terminate_grace_s": config.terminate_grace_s,
        "kill_wait_s": config.kill_wait_s,
        "retry_policy": "none_stop_on_first_failure",
        "attempts_file": ATTEMPTS_NAME,
        "acceptance_output": ACCEPTANCE_NAME,
        "runner_command_template": _run_command(config, Path("<RUN_DIR>")),
        "aggregate_command": _aggregate_command(
            config, config.campaign_root / ACCEPTANCE_NAME
        ),
        "environment_overrides": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "LD_LIBRARY_PATH",
                "PYTHONNOUSERSITE",
                "PYTHONPATH",
                "TOKENIZERS_PARALLELISM",
                "TRANSFORMERS_OFFLINE",
                "VLLM_WORKER_MULTIPROC_METHOD",
            )
        },
    }


def prepare_campaign(
    config: CampaignConfig,
    *,
    _expected_runs: int = PRODUCTION_RUN_COUNT,
) -> str:
    """Create and fsync the sole preregistration file in a brand-new root."""

    _validate_expected_runs(_expected_runs)
    preparation_git_head = (
        _preparation_git_head() if _expected_runs == PRODUCTION_RUN_COUNT else None
    )
    frozen_config, frozen_files, parent_binding, bundle_validation = _normalized_config(
        config
    )
    if _expected_runs == PRODUCTION_RUN_COUNT:
        _require(
            _preparation_git_head() == preparation_git_head,
            "DAGKV Git HEAD changed during formal preparation",
        )
    root = frozen_config.campaign_root
    _require(not os.path.lexists(root), f"campaign root must be brand new: {root}")
    _require(root.parent.is_dir(), f"campaign parent is missing: {root.parent}")
    preregistration = _preregistration_payload(
        frozen_config,
        frozen_files,
        parent_binding,
        bundle_validation,
        expected_runs=_expected_runs,
        preparation_git_head=preparation_git_head,
    )
    root.mkdir(mode=0o750)
    _fsync_directory(root.parent)
    path = root / PREREGISTRATION_NAME
    _write_json_atomic(path, preregistration)
    return _sha256_file(path)


def _prepared_campaign(
    campaign_root: Path,
    expected_preregistration_sha256: str,
    *,
    expected_runs: int,
) -> tuple[
    CampaignConfig,
    dict[str, Any],
    dict[str, dict[str, Any]],
    str,
]:
    _validate_expected_runs(expected_runs)
    expected_sha = _validate_digest(
        expected_preregistration_sha256, label="expected preregistration"
    )
    root = campaign_root.expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid campaign root: {root}")
    _require(
        sorted(entry.name for entry in root.iterdir()) == [PREREGISTRATION_NAME],
        "prepared campaign root must contain only its preregistration",
    )
    preregistration_path = root / PREREGISTRATION_NAME
    _require(
        _sha256_file(preregistration_path) == expected_sha,
        "prepared campaign preregistration SHA-256 does not match",
    )
    preregistration = _read_json_object(
        preregistration_path, label="formal campaign preregistration"
    )
    _require(
        preregistration.get("schema_version") == CAMPAIGN_SCHEMA,
        "formal preregistration schema drifted",
    )
    _require(
        preregistration.get("campaign_root") == str(root),
        "formal campaign root binding drifted",
    )
    _require(
        preregistration.get("formal_campaign_protocol_schema")
        == FORMAL_CAMPAIGN_PROTOCOL_SCHEMA
        and preregistration.get("data_plane_protocol_schema")
        == DATA_PLANE_PROTOCOL_SCHEMA,
        "formal protocol schema binding drifted",
    )
    _require(
        preregistration.get("launch_marker_repository_path")
        == LAUNCH_MARKER_REPOSITORY_PATH,
        "formal launch marker path drifted",
    )
    run_names = [f"run-{index:03d}" for index in range(1, expected_runs + 1)]
    _require(
        preregistration.get("expected_runs") == expected_runs
        and preregistration.get("production_run_count") == PRODUCTION_RUN_COUNT
        and preregistration.get("test_injected_run_count")
        is (expected_runs != PRODUCTION_RUN_COUNT),
        "prepared run-count freeze drifted",
    )
    preparation_head = preregistration.get("preparation_git_head")
    if expected_runs == PRODUCTION_RUN_COUNT:
        _validate_git_head(preparation_head, label="prepared Git HEAD")
    else:
        _require(
            preparation_head is None,
            "test-injected formal campaign cannot carry a production Git head",
        )
    _require(preregistration.get("run_names") == run_names, "run names drifted")
    _require(
        preregistration.get("formal_attempt_prefix_record_count") == expected_runs * 2,
        "formal attempt prefix size drifted",
    )
    _require(
        preregistration.get("selection_rule") == _selection_rule(run_names),
        "formal selection rule drifted",
    )
    _require(
        preregistration.get("retry_policy") == "none_stop_on_first_failure",
        "formal retry policy drifted",
    )
    _require(
        preregistration.get("attempts_file") == ATTEMPTS_NAME
        and preregistration.get("acceptance_output") == ACCEPTANCE_NAME,
        "formal output names drifted",
    )
    try:
        config = CampaignConfig(
            campaign_root=root,
            calibration_manifest=Path(
                preregistration["frozen_files"]["calibration_manifest"]["path"]
            ),
            frozen_tolerance=Path(
                preregistration["frozen_files"]["frozen_tolerance"]["path"]
            ),
            expected_implementation_manifest_sha256=preregistration[
                "expected_implementation_manifest_sha256"
            ],
            expected_reproducibility_fingerprint=preregistration[
                "expected_reproducibility_fingerprint"
            ],
            nvidia_userspace_bundle_root=Path(
                preregistration["nvidia_userspace_bundle_root"]
            ),
            expected_nvidia_userspace_bundle_manifest_sha256=preregistration[
                "expected_nvidia_userspace_bundle_manifest_sha256"
            ],
            expected_nvidia_userspace_bundle_content_digest=preregistration[
                "expected_nvidia_userspace_bundle_content_digest"
            ],
            expected_nvidia_driver_version=preregistration[
                "expected_nvidia_driver_version"
            ],
            python_executable=Path(preregistration["python_executable"]["path"]),
            runner=Path(preregistration["frozen_files"]["runner"]["path"]),
            aggregator=Path(
                preregistration["frozen_files"]["formal_aggregator"]["path"]
            ),
            model=Path(preregistration["model"]),
            vllm_root=Path(preregistration["vllm_root"]),
            cpu_bytes=preregistration["cpu_bytes"],
            runner_timeout_s=preregistration["runner_timeout_s"],
            process_timeout_s=preregistration["process_timeout_s"],
            aggregation_timeout_s=preregistration["aggregation_timeout_s"],
            terminate_grace_s=preregistration["terminate_grace_s"],
            kill_wait_s=preregistration["kill_wait_s"],
            cuda_device=preregistration["cuda_device"],
        )
    except (KeyError, TypeError) as exc:
        raise FormalCampaignError(
            f"prepared formal campaign configuration is malformed: {exc}"
        ) from exc
    frozen_config, frozen_files, parent_binding, bundle_validation = _normalized_config(
        config
    )
    _require(
        preregistration.get("frozen_files") == frozen_files,
        "prepared formal frozen-file inventory drifted",
    )
    _require(
        preregistration.get("parent_binding") == parent_binding,
        "prepared formal parent binding drifted",
    )
    _require(
        preregistration.get("formal_campaign_protocol_sha256")
        == frozen_files["formal_protocol"]["sha256"]
        and preregistration.get("data_plane_protocol_sha256")
        == frozen_files["data_plane_protocol"]["sha256"],
        "prepared protocol SHA-256 binding drifted",
    )
    _require(
        preregistration.get("python_executable")
        == _frozen_file_entry(frozen_config.python_executable),
        "prepared Python executable drifted",
    )
    _require(
        preregistration.get("runner_command_template")
        == _run_command(frozen_config, Path("<RUN_DIR>"))
        and preregistration.get("aggregate_command")
        == _aggregate_command(frozen_config, root / ACCEPTANCE_NAME),
        "prepared formal command freeze drifted",
    )
    environment = _runner_environment(
        frozen_config.cuda_device,
        nvidia_library_path=bundle_validation.library_path,
    )
    environment_keys = (
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "LD_LIBRARY_PATH",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
        "VLLM_WORKER_MULTIPROC_METHOD",
    )
    _require(
        preregistration.get("environment_overrides")
        == {key: environment[key] for key in environment_keys},
        "prepared formal environment drifted",
    )
    campaign_id = preregistration.get("campaign_id")
    _require(
        isinstance(campaign_id, str)
        and campaign_id.startswith("m2-formal-")
        and len(campaign_id.removeprefix("m2-formal-")) == 32,
        "prepared formal campaign ID is malformed",
    )
    _verify_frozen_files(frozen_files)
    return frozen_config, preregistration, frozen_files, expected_sha


def _campaign_paths(root: Path, run_name: str) -> tuple[Path, Path, Path]:
    return (
        root / run_name,
        root / f"{run_name}.stdout.log",
        root / f"{run_name}.stderr.log",
    )


def _terminal_payload(
    submitted: Mapping[str, Any],
    *,
    status: str,
    outcome: ProcessOutcome | None,
    stdout_path: Path,
    stderr_path: Path,
    artifact_path: Path,
    error: str | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "campaign_id": submitted["campaign_id"],
        "attempt_id": submitted["attempt_id"],
        "kind": submitted["kind"],
        "event": "terminal",
        "timestamp_utc": _utc_now(),
        "status": status,
        "pid": outcome.pid if outcome is not None else None,
        "exit_code": outcome.exit_code if outcome is not None else None,
        "duration_s": outcome.duration_s if outcome is not None else None,
        "started_at_utc": outcome.started_at_utc if outcome is not None else None,
        "ended_at_utc": outcome.ended_at_utc if outcome is not None else None,
        "timed_out": outcome.timed_out if outcome is not None else False,
        "sigterm_sent": outcome.sigterm_sent if outcome is not None else False,
        "sigkill_sent": outcome.sigkill_sent if outcome is not None else False,
        "stdout": {
            "path": stdout_path.name,
            "size": stdout_path.stat().st_size if stdout_path.is_file() else None,
            "sha256": _sha256_file(stdout_path) if stdout_path.is_file() else None,
        },
        "stderr": {
            "path": stderr_path.name,
            "size": stderr_path.stat().st_size if stderr_path.is_file() else None,
            "sha256": _sha256_file(stderr_path) if stderr_path.is_file() else None,
        },
        "artifact_inventory": _artifact_inventory(artifact_path),
        "error": error,
    }
    for key in ("sequence", "run_name"):
        if key in submitted:
            payload[key] = submitted[key]
    if validation is not None:
        payload["validation"] = dict(validation)
    return payload


def _validate_completed_run(
    run_dir: Path,
    *,
    expected_implementation_manifest_sha256: str,
    expected_reproducibility_fingerprint: str,
    expected_frozen_tolerance_sha256: str,
    expected_calibration_manifest_sha256: str,
    expected_execution_git_head: str,
) -> dict[str, Any]:
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
        raise FormalCampaignError(
            f"formal run artifact validation failed for {run_dir.name}: {exc}"
        ) from exc

    result = _read_json_object(run_dir / "result.json", label="formal result")
    provenance = _read_json_object(
        run_dir / "provenance.json", label="formal provenance"
    )
    formal = _read_json_object(
        run_dir / FORMAL_RUN_MANIFEST, label="per-run formal manifest"
    )
    implementation = provenance.get("implementation")
    observed_implementation = (
        implementation.get("manifest_sha256")
        if isinstance(implementation, dict)
        else None
    )
    dagkv_git = provenance.get("dagkv_git")
    _require(isinstance(dagkv_git, dict), f"{run_dir.name} lacks DAGKV Git capture")
    dagkv_git_head = dagkv_git.get("head")
    _require(
        dagkv_git_head == expected_execution_git_head,
        f"{run_dir.name} DAGKV Git HEAD differs from execution binding",
    )
    dagkv_snapshot_sha256 = _validate_digest(
        validated.dagkv_snapshot_sha256,
        label=f"{run_dir.name} DAGKV snapshot",
    )
    _require(
        result.get("gate_status") == "M2_ITEM8_FORMAL_HOLDOUT_PASSED"
        and result.get("formal_run_passed") is True
        and result.get("m2_item8_accepted") is False
        and result.get("m2_accepted") is False,
        f"{run_dir.name} formal gate status drifted",
    )
    _require(
        result.get("within_requested_tolerance") is True
        and result.get("tolerance") == {"atol": FORMAL_ATOL, "rtol": FORMAL_RTOL},
        f"{run_dir.name} formal tolerance drifted",
    )
    _require(
        raw.mode == "formal" and raw.run_id == validated.run_id,
        f"{run_dir.name} raw replay identity drifted",
    )
    _require(
        raw.implementation_manifest_sha256
        == observed_implementation
        == expected_implementation_manifest_sha256,
        f"{run_dir.name} implementation manifest drifted",
    )
    _require(
        raw.reproducibility_fingerprint
        == validated.reproducibility_fingerprint
        == expected_reproducibility_fingerprint,
        f"{run_dir.name} reproducibility fingerprint drifted",
    )
    _require(
        validated.frozen_tolerance_sha256 == expected_frozen_tolerance_sha256
        and formal.get("frozen_tolerance_sha256") == expected_frozen_tolerance_sha256,
        f"{run_dir.name} frozen tolerance binding drifted",
    )
    _require(
        validated.calibration_manifest_sha256 == expected_calibration_manifest_sha256
        and formal.get("calibration_manifest_sha256")
        == expected_calibration_manifest_sha256,
        f"{run_dir.name} calibration manifest binding drifted",
    )
    return {
        "run_id": validated.run_id,
        "result_sha256": validated.result_sha256,
        "provenance_sha256": validated.provenance_sha256,
        "sha256sums_sha256": validated.sha256sums_sha256,
        "formal_run_manifest_sha256": validated.formal_run_manifest_sha256,
        "frozen_tolerance_sha256": validated.frozen_tolerance_sha256,
        "calibration_manifest_sha256": validated.calibration_manifest_sha256,
        "implementation_manifest_sha256": observed_implementation,
        "reproducibility_fingerprint": validated.reproducibility_fingerprint,
        "protocol_sha256": validated.protocol_sha256,
        "observed_max_abs_error": raw.observed_max_abs_error,
        "minimum_top1_margin": raw.minimum_top1_margin,
        "dagkv_git_head": dagkv_git_head,
        "dagkv_snapshot_sha256": dagkv_snapshot_sha256,
    }


def _sealed_formal_prefix(
    attempts_path: Path,
    *,
    campaign_id: str,
    run_names: Sequence[str],
    preregistration_sha256: str,
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        attempts_path.is_file() and not attempts_path.is_symlink(),
        "formal attempt journal is missing before aggregation",
    )
    journal_raw = attempts_path.read_bytes()
    lines = journal_raw.splitlines(keepends=True)
    expected_records = len(run_names) * 2
    _require(
        len(lines) == expected_records
        and all(line.endswith(b"\n") for line in lines[:expected_records]),
        "formal attempt prefix has an invalid record boundary",
    )
    for sequence, run_name in enumerate(run_names, start=1):
        try:
            submitted = json.loads(lines[(sequence - 1) * 2])
            terminal = json.loads(lines[(sequence - 1) * 2 + 1])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormalCampaignError(
                f"invalid formal attempt record for {run_name}: {exc}"
            ) from exc
        attempt_id = f"{campaign_id}:{run_name}"
        for record in (submitted, terminal):
            _require(isinstance(record, dict), f"invalid attempt row for {run_name}")
            _require(
                record.get("schema_version") == ATTEMPT_SCHEMA
                and record.get("campaign_id") == campaign_id
                and record.get("attempt_id") == attempt_id
                and record.get("kind") == "formal_run"
                and record.get("sequence") == sequence
                and record.get("run_name") == run_name,
                f"formal attempt identity drifted for {run_name}",
            )
        _require(
            submitted.get("event") == "submitted"
            and submitted.get("preregistration_sha256") == preregistration_sha256,
            f"formal submitted record drifted for {run_name}",
        )
        _require(
            submitted.get("execution_binding") == execution_binding,
            f"formal execution binding drifted for {run_name}",
        )
        _require(
            terminal.get("event") == "terminal"
            and terminal.get("status") == "passed"
            and type(terminal.get("pid")) is int
            and terminal["pid"] > 0,
            f"formal terminal record drifted for {run_name}",
        )
    prefix_raw = b"".join(lines[:expected_records])
    return {
        "prefix_bytes": len(prefix_raw),
        "prefix_record_count": expected_records,
        "prefix_sha256": hashlib.sha256(prefix_raw).hexdigest(),
    }


def _validated_run_records(
    root: Path,
    *,
    run_names: Sequence[str],
    config: CampaignConfig,
    parent_binding: Mapping[str, Any],
    execution_git_head: str,
) -> list[dict[str, Any]]:
    return [
        _validate_completed_run(
            root / run_name,
            expected_implementation_manifest_sha256=(
                config.expected_implementation_manifest_sha256
            ),
            expected_reproducibility_fingerprint=(
                config.expected_reproducibility_fingerprint
            ),
            expected_frozen_tolerance_sha256=parent_binding["frozen_tolerance_sha256"],
            expected_calibration_manifest_sha256=parent_binding[
                "calibration_manifest_sha256"
            ],
            expected_execution_git_head=execution_git_head,
        )
        for run_name in run_names
    ]


def _validate_acceptance_manifest(
    path: Path,
    *,
    expected_runs: int,
    parent_binding: Mapping[str, Any],
    validated_runs: Sequence[Mapping[str, Any]],
    data_plane_protocol_sha256: str,
) -> dict[str, Any]:
    manifest = _read_json_object(path, label="item-8 acceptance manifest")
    _require(
        manifest.get("schema_version") == ACCEPTANCE_SCHEMA
        and manifest.get("protocol_schema") == DATA_PLANE_PROTOCOL_SCHEMA,
        "item-8 acceptance schema drifted",
    )
    _require(
        manifest.get("gate_status") == "M2_ITEM8_ACCEPTED"
        and manifest.get("m2_item8_accepted") is True
        and manifest.get("m2_accepted") is False
        and manifest.get("performance_claims_supported") is False,
        "item-8 acceptance claim boundary drifted",
    )
    _require(
        manifest.get("run_count") == expected_runs
        and manifest.get("passed_run_count") == expected_runs,
        "item-8 acceptance run count drifted",
    )
    _require(
        manifest.get("frozen_tolerance_sha256")
        == parent_binding["frozen_tolerance_sha256"]
        and manifest.get("calibration_manifest_sha256")
        == parent_binding["calibration_manifest_sha256"]
        and manifest.get("reproducibility_fingerprint")
        == parent_binding["reproducibility_fingerprint"],
        "item-8 acceptance parent binding drifted",
    )
    _require(
        manifest.get("protocol_sha256") == data_plane_protocol_sha256,
        "item-8 acceptance protocol hash drifted",
    )
    _require(
        manifest.get("nvidia_userspace_bundle_root")
        == parent_binding["nvidia_userspace_bundle_root"]
        and manifest.get("nvidia_userspace_bundle_manifest_sha256")
        == parent_binding["expected_nvidia_userspace_bundle_manifest_sha256"]
        and manifest.get("nvidia_userspace_bundle_content_digest")
        == parent_binding["expected_nvidia_userspace_bundle_content_digest"]
        and manifest.get("nvidia_driver_version")
        == parent_binding["expected_nvidia_driver_version"],
        "item-8 acceptance NVIDIA userspace/driver binding drifted",
    )
    expected_entries = sorted(
        (
            {
                "run_id": row["run_id"],
                "formal_run_manifest_sha256": row["formal_run_manifest_sha256"],
                "result_sha256": row["result_sha256"],
                "provenance_sha256": row["provenance_sha256"],
                "sha256sums_sha256": row["sha256sums_sha256"],
            }
            for row in validated_runs
        ),
        key=lambda row: row["run_id"],
    )
    _require(manifest.get("runs") == expected_entries, "acceptance run set drifted")
    _require(
        manifest.get("statement") == ACCEPTANCE_STATEMENT,
        "item-8 acceptance statement drifted",
    )
    return manifest


def _expected_root_entries(run_names: Sequence[str]) -> list[str]:
    entries = {
        PREREGISTRATION_NAME,
        ATTEMPTS_NAME,
        ACCEPTANCE_NAME,
        "aggregate.stdout.log",
        "aggregate.stderr.log",
    }
    for run_name in run_names:
        entries.update({run_name, f"{run_name}.stdout.log", f"{run_name}.stderr.log"})
    return sorted(entries)


def _replay_production_evidence(
    acceptance_path: Path,
    *,
    config: CampaignConfig,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    frozen_files: Mapping[str, Mapping[str, Any]],
    aggregate_terminal_expected: bool,
) -> None:
    root = config.campaign_root
    run_names = preregistration["run_names"]
    bundle_before = _fresh_validate_nvidia_bundle(config)
    _require(
        sorted(entry.name for entry in root.iterdir())
        == _expected_root_entries(run_names),
        "formal campaign root is not the declared closed set",
    )
    _verify_frozen_files(frozen_files)
    _require(
        _current_implementation_manifest_sha256()
        == config.expected_implementation_manifest_sha256,
        "current implementation changed during formal evidence replay",
    )
    _require(
        _sha256_file(root / PREREGISTRATION_NAME) == preregistration_sha256,
        "formal preregistration changed during evidence replay",
    )
    execution_binding = _establish_execution_binding(
        preregistration, preregistration_sha256
    )
    parent_binding = _validate_parent_inputs(
        config.calibration_manifest,
        config.frozen_tolerance,
        expected_implementation_manifest_sha256=(
            config.expected_implementation_manifest_sha256
        ),
        expected_reproducibility_fingerprint=(
            config.expected_reproducibility_fingerprint
        ),
        expected_nvidia_userspace_bundle_root=(config.nvidia_userspace_bundle_root),
        expected_nvidia_userspace_bundle_manifest_sha256=(
            config.expected_nvidia_userspace_bundle_manifest_sha256
        ),
        expected_nvidia_userspace_bundle_content_digest=(
            config.expected_nvidia_userspace_bundle_content_digest
        ),
        expected_nvidia_driver_version=config.expected_nvidia_driver_version,
    )
    _require(
        parent_binding == preregistration["parent_binding"],
        "formal parent evidence changed during replay",
    )
    attempts_path = root / ATTEMPTS_NAME
    raw = attempts_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    expected_count = PRODUCTION_RUN_COUNT * 2 + 1 + int(aggregate_terminal_expected)
    _require(
        len(lines) == expected_count and all(line.endswith(b"\n") for line in lines),
        "formal journal record count or boundary drifted",
    )
    sealed = _sealed_formal_prefix(
        root / ATTEMPTS_NAME,
        campaign_id=preregistration["campaign_id"],
        run_names=run_names,
        preregistration_sha256=preregistration_sha256,
        execution_binding=execution_binding,
    )
    try:
        aggregate_submitted = json.loads(lines[PRODUCTION_RUN_COUNT * 2])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalCampaignError(
            f"invalid aggregate submission record: {exc}"
        ) from exc
    _require(
        isinstance(aggregate_submitted, dict)
        and aggregate_submitted.get("schema_version") == ATTEMPT_SCHEMA
        and aggregate_submitted.get("campaign_id") == preregistration["campaign_id"]
        and aggregate_submitted.get("attempt_id")
        == f"{preregistration['campaign_id']}:aggregate"
        and aggregate_submitted.get("kind") == "aggregate"
        and aggregate_submitted.get("event") == "submitted"
        and aggregate_submitted.get("preregistration_sha256") == preregistration_sha256
        and aggregate_submitted.get("execution_binding") == execution_binding
        and aggregate_submitted.get("formal_prefix") == sealed,
        "formal aggregate submission drifted",
    )
    validated_runs = _validated_run_records(
        root,
        run_names=run_names,
        config=config,
        parent_binding=parent_binding,
        execution_git_head=execution_binding["execution_git_head"],
    )
    _require(
        len({row["dagkv_snapshot_sha256"] for row in validated_runs}) == 1,
        "replayed formal holdouts do not share one DAGKV snapshot",
    )
    manifest = _validate_acceptance_manifest(
        acceptance_path,
        expected_runs=PRODUCTION_RUN_COUNT,
        parent_binding=parent_binding,
        validated_runs=validated_runs,
        data_plane_protocol_sha256=frozen_files["data_plane_protocol"]["sha256"],
    )
    if aggregate_terminal_expected:
        try:
            terminal = json.loads(lines[-1])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormalCampaignError(
                f"invalid aggregate terminal record: {exc}"
            ) from exc
        validation = terminal.get("validation") if isinstance(terminal, dict) else None
        _require(
            isinstance(terminal, dict)
            and terminal.get("schema_version") == ATTEMPT_SCHEMA
            and terminal.get("campaign_id") == preregistration["campaign_id"]
            and terminal.get("attempt_id")
            == f"{preregistration['campaign_id']}:aggregate"
            and terminal.get("kind") == "aggregate"
            and terminal.get("event") == "terminal"
            and terminal.get("status") == "passed"
            and isinstance(validation, dict)
            and validation.get("acceptance_sha256") == _sha256_file(acceptance_path)
            and validation.get("run_count") == PRODUCTION_RUN_COUNT
            and validation.get("gate_status") == manifest["gate_status"],
            "formal aggregate terminal drifted",
        )
    parent_after = _validate_parent_inputs(
        config.calibration_manifest,
        config.frozen_tolerance,
        expected_implementation_manifest_sha256=(
            config.expected_implementation_manifest_sha256
        ),
        expected_reproducibility_fingerprint=(
            config.expected_reproducibility_fingerprint
        ),
        expected_nvidia_userspace_bundle_root=config.nvidia_userspace_bundle_root,
        expected_nvidia_userspace_bundle_manifest_sha256=(
            config.expected_nvidia_userspace_bundle_manifest_sha256
        ),
        expected_nvidia_userspace_bundle_content_digest=(
            config.expected_nvidia_userspace_bundle_content_digest
        ),
        expected_nvidia_driver_version=config.expected_nvidia_driver_version,
    )
    _require(
        parent_after == parent_binding,
        "formal parent evidence changed during replay",
    )
    _require(
        _fresh_validate_nvidia_bundle(config) == bundle_before,
        "NVIDIA userspace bundle changed during formal evidence replay",
    )


def _revalidate_production_candidate(
    acceptance_path: Path,
    *,
    config: CampaignConfig,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    frozen_files: Mapping[str, Mapping[str, Any]],
) -> None:
    _replay_production_evidence(
        acceptance_path,
        config=config,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
        frozen_files=frozen_files,
        aggregate_terminal_expected=False,
    )


def _revalidate_production_bundle(
    acceptance_path: Path,
    *,
    config: CampaignConfig,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    frozen_files: Mapping[str, Mapping[str, Any]],
) -> None:
    _replay_production_evidence(
        acceptance_path,
        config=config,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
        frozen_files=frozen_files,
        aggregate_terminal_expected=True,
    )


def _publish_and_revalidate_production_seal(
    campaign_root: Path,
    *,
    expected_preregistration_sha256: str,
    config: CampaignConfig,
) -> str:
    bundle_before = _fresh_validate_nvidia_bundle(config)
    try:
        from tools.m2_formal_evidence import (  # noqa: PLC0415
            FormalEvidenceError,
            publish_formal_bundle_seal,
            validate_published_formal_bundle,
        )

        _, seal_sha256 = publish_formal_bundle_seal(
            campaign_root,
            expected_preregistration_sha256=expected_preregistration_sha256,
        )
        validate_published_formal_bundle(
            campaign_root / BUNDLE_SEAL_NAME,
            expected_seal_sha256=seal_sha256,
            expected_preregistration_sha256=expected_preregistration_sha256,
        )
        _revalidate_nvidia_boundary(
            config,
            bundle_before,
            label="formal seal publication and replay",
        )
    except (FormalEvidenceError, OSError) as exc:
        raise FormalCampaignError(
            f"formal bundle seal publication or replay failed: {exc}"
        ) from exc
    return seal_sha256


def _record_process_failure(
    *,
    submitted: Mapping[str, Any],
    outcome: ProcessOutcome | None,
    stdout_path: Path,
    stderr_path: Path,
    artifact_path: Path,
    attempts_path: Path,
    error: BaseException,
) -> None:
    terminal = _terminal_payload(
        submitted,
        status=(
            "orchestrator_interrupted"
            if isinstance(error, ProcessExecutionInterrupted)
            else "spawn_or_termination_failed"
        ),
        outcome=outcome,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        artifact_path=artifact_path,
        error=str(error),
    )
    _append_attempt(attempts_path, terminal)


def _revalidate_nvidia_boundary(
    config: CampaignConfig,
    before: BundleValidation,
    *,
    label: str,
) -> BundleValidation:
    after = _fresh_validate_nvidia_bundle(config)
    _require(after == before, f"NVIDIA userspace bundle changed during {label}")
    return after


def _verify_execution_freeze(
    *,
    config: CampaignConfig,
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
    preregistration_sha256: str,
    frozen_files: Mapping[str, Mapping[str, Any]],
    execution_binding: Mapping[str, Any],
    production: bool,
) -> BundleValidation:
    _verify_frozen_files(frozen_files)
    _require(
        _current_implementation_manifest_sha256()
        == config.expected_implementation_manifest_sha256,
        "current M2 implementation changed before formal submission",
    )
    _require(
        _sha256_file(preregistration_path) == preregistration_sha256,
        "formal preregistration changed after execution began",
    )
    if production:
        _revalidate_execution_binding(
            execution_binding,
            preregistration=preregistration,
            preregistration_sha256=preregistration_sha256,
        )
    return _fresh_validate_nvidia_bundle(config)


def _execute_prepared_campaign_locked(
    campaign_root: Path,
    expected_preregistration_sha256: str,
    *,
    _expected_runs: int = PRODUCTION_RUN_COUNT,
) -> dict[str, Any]:
    """Execute a pristine prepared formal campaign and stop at first failure."""

    config, preregistration, frozen_files, preregistration_sha256 = _prepared_campaign(
        campaign_root,
        expected_preregistration_sha256,
        expected_runs=_expected_runs,
    )
    root = config.campaign_root
    attempts_path = root / ATTEMPTS_NAME
    acceptance_path = root / ACCEPTANCE_NAME
    preregistration_path = root / PREREGISTRATION_NAME
    campaign_id = preregistration["campaign_id"]
    run_names = preregistration["run_names"]
    parent_binding = preregistration["parent_binding"]
    production = _expected_runs == PRODUCTION_RUN_COUNT
    execution_binding = (
        _establish_execution_binding(preregistration, preregistration_sha256)
        if production
        else _test_execution_binding(preregistration)
    )
    completed_validations: list[dict[str, Any]] = []

    for sequence, run_name in enumerate(run_names, start=1):
        bundle_before = _verify_execution_freeze(
            config=config,
            preregistration=preregistration,
            preregistration_path=preregistration_path,
            preregistration_sha256=preregistration_sha256,
            frozen_files=frozen_files,
            execution_binding=execution_binding,
            production=production,
        )
        environment = _runner_environment(
            config.cuda_device,
            nvidia_library_path=bundle_before.library_path,
        )
        run_dir, stdout_path, stderr_path = _campaign_paths(root, run_name)
        _require(
            not os.path.lexists(run_dir)
            and not os.path.lexists(stdout_path)
            and not os.path.lexists(stderr_path),
            f"fresh formal run paths already exist: {run_name}",
        )
        command = _run_command(config, run_dir)
        submitted = {
            "schema_version": ATTEMPT_SCHEMA,
            "campaign_id": campaign_id,
            "attempt_id": f"{campaign_id}:{run_name}",
            "kind": "formal_run",
            "event": "submitted",
            "timestamp_utc": _utc_now(),
            "sequence": sequence,
            "run_name": run_name,
            "command": command,
            "output_dir": run_name,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "preregistration_sha256": preregistration_sha256,
            "execution_binding": dict(execution_binding),
        }
        _append_attempt(attempts_path, submitted)

        outcome: ProcessOutcome | None = None
        try:
            outcome = _run_process(
                command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                environment=environment,
                timeout_s=config.process_timeout_s,
                terminate_grace_s=config.terminate_grace_s,
                kill_wait_s=config.kill_wait_s,
            )
        except (OSError, ProcessSupervisorError) as exc:
            if isinstance(exc, ProcessExecutionInterrupted):
                outcome = exc.outcome
            error: BaseException = exc
            try:
                _revalidate_nvidia_boundary(
                    config,
                    bundle_before,
                    label=run_name,
                )
            except FormalCampaignError as bundle_exc:
                error = FormalCampaignError(
                    f"{exc}; post-run bundle validation failed: {bundle_exc}"
                )
            _record_process_failure(
                submitted=submitted,
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                attempts_path=attempts_path,
                error=error,
            )
            raise FormalCampaignError(
                f"{run_name} could not reach a process terminal: {error}"
            ) from exc

        try:
            _revalidate_nvidia_boundary(
                config,
                bundle_before,
                label=run_name,
            )
        except FormalCampaignError as exc:
            terminal = _terminal_payload(
                submitted,
                status="validation_failed",
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error=f"post-run bundle validation failed: {exc}",
            )
            _append_attempt(attempts_path, terminal)
            raise

        if outcome.timed_out or outcome.exit_code != 0:
            status = "timed_out" if outcome.timed_out else "process_failed"
            terminal = _terminal_payload(
                submitted,
                status=status,
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error=(
                    f"process exceeded {config.process_timeout_s} seconds"
                    if outcome.timed_out
                    else f"runner exited with status {outcome.exit_code}"
                ),
            )
            _append_attempt(attempts_path, terminal)
            raise FormalCampaignError(
                f"{run_name} failed; formal campaign stopped without retry"
            )
        if outcome.sigterm_sent or outcome.sigkill_sent:
            terminal = _terminal_payload(
                submitted,
                status="lingering_process_group",
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error="runner exited but left live descendants in its process group",
            )
            _append_attempt(attempts_path, terminal)
            raise FormalCampaignError(
                f"{run_name} left a live process group; campaign stopped without retry"
            )
        try:
            _verify_frozen_files(frozen_files)
            validation = _validate_completed_run(
                run_dir,
                expected_implementation_manifest_sha256=(
                    config.expected_implementation_manifest_sha256
                ),
                expected_reproducibility_fingerprint=(
                    config.expected_reproducibility_fingerprint
                ),
                expected_frozen_tolerance_sha256=parent_binding[
                    "frozen_tolerance_sha256"
                ],
                expected_calibration_manifest_sha256=parent_binding[
                    "calibration_manifest_sha256"
                ],
                expected_execution_git_head=execution_binding["execution_git_head"],
            )
        except FormalCampaignError as exc:
            terminal = _terminal_payload(
                submitted,
                status="validation_failed",
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error=str(exc),
            )
            _append_attempt(attempts_path, terminal)
            raise
        if production:
            _require(
                validation.get("dagkv_git_head")
                == execution_binding["execution_git_head"],
                f"{run_name} provenance Git HEAD drifted",
            )
        completed_validations.append(validation)
        terminal = _terminal_payload(
            submitted,
            status="passed",
            outcome=outcome,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_path=run_dir,
            validation=validation,
        )
        _append_attempt(attempts_path, terminal)

    if production:
        _require(
            len({row.get("dagkv_snapshot_sha256") for row in completed_validations})
            == 1,
            "formal holdouts do not share one frozen DAGKV snapshot",
        )
    aggregate_bundle_before = _verify_execution_freeze(
        config=config,
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        preregistration_sha256=preregistration_sha256,
        frozen_files=frozen_files,
        execution_binding=execution_binding,
        production=production,
    )
    aggregate_environment = _runner_environment(
        config.cuda_device,
        nvidia_library_path=aggregate_bundle_before.library_path,
    )
    formal_prefix = _sealed_formal_prefix(
        attempts_path,
        campaign_id=campaign_id,
        run_names=run_names,
        preregistration_sha256=preregistration_sha256,
        execution_binding=execution_binding,
    )
    _verify_frozen_files(frozen_files)
    aggregate_stdout = root / "aggregate.stdout.log"
    aggregate_stderr = root / "aggregate.stderr.log"
    aggregate_command = _aggregate_command(config, acceptance_path)
    aggregate_submitted = {
        "schema_version": ATTEMPT_SCHEMA,
        "campaign_id": campaign_id,
        "attempt_id": f"{campaign_id}:aggregate",
        "kind": "aggregate",
        "event": "submitted",
        "timestamp_utc": _utc_now(),
        "command": aggregate_command,
        "output": ACCEPTANCE_NAME,
        "stdout": aggregate_stdout.name,
        "stderr": aggregate_stderr.name,
        "preregistration_sha256": preregistration_sha256,
        "execution_binding": dict(execution_binding),
        "formal_prefix": formal_prefix,
    }
    _append_attempt(attempts_path, aggregate_submitted)
    aggregate_outcome: ProcessOutcome | None = None
    try:
        aggregate_outcome = _run_process(
            aggregate_command,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            environment=aggregate_environment,
            timeout_s=config.aggregation_timeout_s,
            terminate_grace_s=config.terminate_grace_s,
            kill_wait_s=config.kill_wait_s,
        )
    except (OSError, ProcessSupervisorError) as exc:
        if isinstance(exc, ProcessExecutionInterrupted):
            aggregate_outcome = exc.outcome
        error: BaseException = exc
        try:
            _revalidate_nvidia_boundary(
                config,
                aggregate_bundle_before,
                label="formal aggregation",
            )
        except FormalCampaignError as bundle_exc:
            error = FormalCampaignError(
                f"{exc}; post-aggregate bundle validation failed: {bundle_exc}"
            )
        _record_process_failure(
            submitted=aggregate_submitted,
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=acceptance_path,
            attempts_path=attempts_path,
            error=error,
        )
        raise FormalCampaignError(
            f"formal aggregation could not terminate: {error}"
        ) from exc

    try:
        _revalidate_nvidia_boundary(
            config,
            aggregate_bundle_before,
            label="formal aggregation",
        )
    except FormalCampaignError as exc:
        terminal = _terminal_payload(
            aggregate_submitted,
            status="validation_failed",
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=acceptance_path,
            error=f"post-aggregate bundle validation failed: {exc}",
        )
        _append_attempt(attempts_path, terminal)
        raise

    if aggregate_outcome.timed_out or aggregate_outcome.exit_code != 0:
        status = "timed_out" if aggregate_outcome.timed_out else "process_failed"
        terminal = _terminal_payload(
            aggregate_submitted,
            status=status,
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=acceptance_path,
            error=(
                "formal aggregation timed out"
                if aggregate_outcome.timed_out
                else (
                    "formal aggregator exited with status "
                    f"{aggregate_outcome.exit_code}"
                )
            ),
        )
        _append_attempt(attempts_path, terminal)
        raise FormalCampaignError(
            f"formal aggregation failed with status {aggregate_outcome.exit_code}"
        )
    if aggregate_outcome.sigterm_sent or aggregate_outcome.sigkill_sent:
        terminal = _terminal_payload(
            aggregate_submitted,
            status="lingering_process_group",
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=acceptance_path,
            error="formal aggregator left live descendants in its process group",
        )
        _append_attempt(attempts_path, terminal)
        raise FormalCampaignError("formal aggregation left a live process group")

    try:
        run_validations = [
            row["validation"]
            for row in (
                json.loads(line)
                for line in attempts_path.read_text(encoding="utf-8").splitlines()[
                    1 : _expected_runs * 2 : 2
                ]
            )
        ]
        acceptance = _validate_acceptance_manifest(
            acceptance_path,
            expected_runs=_expected_runs,
            parent_binding=parent_binding,
            validated_runs=run_validations,
            data_plane_protocol_sha256=frozen_files["data_plane_protocol"]["sha256"],
        )
        if _expected_runs == PRODUCTION_RUN_COUNT:
            _revalidate_production_candidate(
                acceptance_path,
                config=config,
                preregistration=preregistration,
                preregistration_sha256=preregistration_sha256,
                frozen_files=frozen_files,
            )
    except (FormalCampaignError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if not isinstance(exc, FormalCampaignError):
            exc = FormalCampaignError(f"invalid formal attempt validation: {exc}")
        terminal = _terminal_payload(
            aggregate_submitted,
            status="validation_failed",
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=acceptance_path,
            error=str(exc),
        )
        _append_attempt(attempts_path, terminal)
        raise exc

    aggregate_validation = {
        "acceptance_sha256": _sha256_file(acceptance_path),
        "run_count": acceptance["run_count"],
        "gate_status": acceptance["gate_status"],
        "reproducibility_fingerprint": acceptance["reproducibility_fingerprint"],
    }
    terminal = _terminal_payload(
        aggregate_submitted,
        status="passed",
        outcome=aggregate_outcome,
        stdout_path=aggregate_stdout,
        stderr_path=aggregate_stderr,
        artifact_path=acceptance_path,
        validation=aggregate_validation,
    )
    _append_attempt(attempts_path, terminal)
    if _expected_runs == PRODUCTION_RUN_COUNT:
        _revalidate_production_bundle(
            acceptance_path,
            config=config,
            preregistration=preregistration,
            preregistration_sha256=preregistration_sha256,
            frozen_files=frozen_files,
        )
        _verify_execution_freeze(
            config=config,
            preregistration=preregistration,
            preregistration_path=preregistration_path,
            preregistration_sha256=preregistration_sha256,
            frozen_files=frozen_files,
            execution_binding=execution_binding,
            production=True,
        )
        _publish_and_revalidate_production_seal(
            root,
            expected_preregistration_sha256=preregistration_sha256,
            config=config,
        )
    return acceptance


def execute_prepared_campaign(
    campaign_root: Path,
    expected_preregistration_sha256: str,
    *,
    _expected_runs: int = PRODUCTION_RUN_COUNT,
) -> dict[str, Any]:
    """Lock and execute one pristine prepared formal campaign."""

    root = campaign_root.expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid campaign root: {root}")
    with _campaign_execution_lock(root):
        return _execute_prepared_campaign_locked(
            root,
            expected_preregistration_sha256,
            _expected_runs=_expected_runs,
        )


def run_campaign(
    config: CampaignConfig,
    *,
    _expected_runs: int = PRODUCTION_RUN_COUNT,
) -> dict[str, Any]:
    """Run a test-injected campaign; production requires the two-stage API."""

    _require(
        _expected_runs != PRODUCTION_RUN_COUNT,
        "production formal campaigns require separate prepare and execute stages",
    )

    preregistration_sha256 = prepare_campaign(config, _expected_runs=_expected_runs)
    return execute_prepared_campaign(
        config.campaign_root,
        preregistration_sha256,
        _expected_runs=_expected_runs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", action="store_true")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--frozen-tolerance", type=Path)
    parser.add_argument("--expected-implementation-manifest-sha256")
    parser.add_argument("--expected-reproducibility-fingerprint")
    parser.add_argument("--nvidia-userspace-bundle-root", type=Path)
    parser.add_argument("--expected-nvidia-userspace-bundle-manifest-sha256")
    parser.add_argument("--expected-nvidia-userspace-bundle-content-digest")
    parser.add_argument("--expected-nvidia-driver-version")
    parser.add_argument("--expected-preregistration-sha256")
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--aggregator", type=Path, default=DEFAULT_AGGREGATOR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--cpu-bytes", type=int, default=1 << 30)
    parser.add_argument("--runner-timeout-s", type=float, default=60.0)
    parser.add_argument("--process-timeout-s", type=float, default=1800.0)
    parser.add_argument("--aggregation-timeout-s", type=float, default=900.0)
    parser.add_argument("--terminate-grace-s", type=float, default=30.0)
    parser.add_argument("--kill-wait-s", type=float, default=30.0)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.execute_prepared:
        if args.expected_preregistration_sha256 is None:
            parser.error(
                "--execute-prepared requires --expected-preregistration-sha256"
            )
        try:
            acceptance = execute_prepared_campaign(
                args.campaign_root, args.expected_preregistration_sha256
            )
        except (FormalCampaignError, OSError) as exc:
            print(f"M2 formal campaign failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"M2 item-8 formal campaign completed: {args.campaign_root.resolve()} "
            f"({acceptance['passed_run_count']}/{acceptance['run_count']} holdouts)"
        )
        return 0

    required = (
        args.calibration_manifest,
        args.frozen_tolerance,
        args.expected_implementation_manifest_sha256,
        args.expected_reproducibility_fingerprint,
        args.nvidia_userspace_bundle_root,
        args.expected_nvidia_userspace_bundle_manifest_sha256,
        args.expected_nvidia_userspace_bundle_content_digest,
        args.expected_nvidia_driver_version,
    )
    if any(value is None for value in required):
        parser.error(
            "formal preparation requires calibration manifest, frozen tolerance, "
            "expected implementation/reproducibility values, and NVIDIA userspace "
            "bundle root/manifest/content/driver identities"
        )
    if args.expected_preregistration_sha256 is not None:
        parser.error(
            "--expected-preregistration-sha256 is only valid with --execute-prepared"
        )
    config = CampaignConfig(
        campaign_root=args.campaign_root,
        calibration_manifest=args.calibration_manifest,
        frozen_tolerance=args.frozen_tolerance,
        expected_implementation_manifest_sha256=(
            args.expected_implementation_manifest_sha256
        ),
        expected_reproducibility_fingerprint=(
            args.expected_reproducibility_fingerprint
        ),
        nvidia_userspace_bundle_root=args.nvidia_userspace_bundle_root,
        expected_nvidia_userspace_bundle_manifest_sha256=(
            args.expected_nvidia_userspace_bundle_manifest_sha256
        ),
        expected_nvidia_userspace_bundle_content_digest=(
            args.expected_nvidia_userspace_bundle_content_digest
        ),
        expected_nvidia_driver_version=args.expected_nvidia_driver_version,
        python_executable=args.python_executable,
        runner=args.runner,
        aggregator=args.aggregator,
        model=args.model,
        vllm_root=args.vllm_root,
        cpu_bytes=args.cpu_bytes,
        runner_timeout_s=args.runner_timeout_s,
        process_timeout_s=args.process_timeout_s,
        aggregation_timeout_s=args.aggregation_timeout_s,
        terminate_grace_s=args.terminate_grace_s,
        kill_wait_s=args.kill_wait_s,
        cuda_device=args.cuda_device,
    )
    try:
        preregistration_sha256 = prepare_campaign(config)
    except (FormalCampaignError, OSError) as exc:
        print(f"M2 formal preparation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"M2 formal campaign prepared: {config.campaign_root.resolve()} "
        f"preregistration_sha256={preregistration_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
