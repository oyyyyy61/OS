#!/usr/bin/env python3
"""Launch the frozen 59-process M2 calibration campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAMPAIGN_SCHEMA = "dagkv.m2.calibration_campaign_preregistration.v1"
ATTEMPT_SCHEMA = "dagkv.m2.calibration_campaign_attempt.v1"
COHORT_SCHEMA = "dagkv.m2.calibration_cohort.v2"
PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v2"
PRODUCTION_RUN_COUNT = 59
CALIBRATION_ATOL = 0.125
CALIBRATION_RTOL = 0.0
BASE_LD_LIBRARY_PATH = "/usr/local/cuda/lib64:"
NATURAL_DESCENDANT_GRACE_S = 2.0

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
DEFAULT_RUNNER = REPO_ROOT / "tools" / "run_m2_vllm_abba.py"
DEFAULT_AGGREGATOR = REPO_ROOT / "tools" / "aggregate_m2_calibration.py"
EVIDENCE_VALIDATOR_PATH = REPO_ROOT / "tools" / "m2_calibration_evidence.py"
RAW_REPLAY_PATH = REPO_ROOT / "tools" / "m2_raw_replay.py"
PROTOCOL_PATH = REPO_ROOT / "research" / "protocols" / "M2_VLLM_REPLAY_PROTOCOL.md"
INTEGRATION_ROOT = REPO_ROOT / "integrations" / "vllm_m2"
DEFAULT_PYTHON = Path("/home/data/25_oyzx/Agentrix/vllm/.venv/bin/python")
DEFAULT_MODEL = Path("/home/data/25_oyzx/moqae_runtime_gpu/modelscope/Qwen/Qwen3-8B")
DEFAULT_VLLM_ROOT = Path("/home/data/25_oyzx/Agentrix/vllm")

PREREGISTRATION_NAME = "CAMPAIGN_PREREGISTRATION.json"
ATTEMPTS_NAME = "ATTEMPTS.jsonl"
COHORT_NAME = "M2_CALIBRATION_MANIFEST.json"


class CalibrationCampaignError(RuntimeError):
    """Raised when the campaign cannot continue without violating its freeze."""


class _CampaignSignal(BaseException):
    """Convert a terminating shell signal into structured campaign cleanup."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received signal {signal_number}")
        self.signal_number = signal_number


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Immutable inputs for one non-resumable calibration campaign."""

    campaign_root: Path
    expected_implementation_manifest_sha256: str
    expected_reproducibility_fingerprint: str
    python_executable: Path = DEFAULT_PYTHON
    runner: Path = DEFAULT_RUNNER
    aggregator: Path = DEFAULT_AGGREGATOR
    model: Path = DEFAULT_MODEL
    vllm_root: Path = DEFAULT_VLLM_ROOT
    cpu_bytes: int = 1 << 30
    runner_timeout_s: float = 60.0
    process_timeout_s: float = 1800.0
    aggregation_timeout_s: float = 300.0
    terminate_grace_s: float = 30.0
    kill_wait_s: float = 30.0
    cuda_device: int = 0


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    pid: int
    exit_code: int
    duration_s: float
    started_at_utc: str
    ended_at_utc: str
    timed_out: bool
    sigterm_sent: bool
    sigkill_sent: bool


class ProcessExecutionInterrupted(CalibrationCampaignError):
    """Carry a terminal child outcome through orchestrator interruption."""

    def __init__(self, message: str, outcome: ProcessOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationCampaignError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_digest(value: str, *, label: str) -> str:
    _require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


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
        raise CalibrationCampaignError(f"invalid {label} at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _resolved_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
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
        path = Path(str(entry["path"]))
        _require(path.is_file(), f"frozen {label} file vanished: {path}")
        _require(
            path.stat().st_size == entry["size"]
            and _sha256_file(path) == entry["sha256"],
            f"frozen {label} file changed during campaign: {path}",
        )


def _runner_environment(cuda_device: int) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("VLLM_"):
            del environment[key]
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(cuda_device),
            "HF_HUB_OFFLINE": "1",
            "LD_LIBRARY_PATH": BASE_LD_LIBRARY_PATH,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(INTEGRATION_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    return environment


def _process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_group_signal(group_id: int, value: signal.Signals) -> None:
    try:
        os.killpg(group_id, value)
    except ProcessLookupError:
        return


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes], *, timeout_s: float
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        process.poll()
        if not _process_group_exists(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _stop_process_group(
    process: subprocess.Popen[bytes],
    *,
    terminate_grace_s: float,
    kill_wait_s: float,
) -> tuple[bool, bool]:
    sigterm_sent = _process_group_exists(process.pid)
    if sigterm_sent:
        _send_group_signal(process.pid, signal.SIGTERM)
    if _wait_for_process_group_exit(process, timeout_s=terminate_grace_s):
        return sigterm_sent, False

    sigkill_sent = _process_group_exists(process.pid)
    if sigkill_sent:
        _send_group_signal(process.pid, signal.SIGKILL)
    if not _wait_for_process_group_exit(process, timeout_s=kill_wait_s):
        raise CalibrationCampaignError(f"process group {process.pid} survived SIGKILL")
    return sigterm_sent, sigkill_sent


def _outcome(
    process: subprocess.Popen[bytes],
    *,
    started: float,
    started_at_utc: str,
    timed_out: bool,
    sigterm_sent: bool,
    sigkill_sent: bool,
) -> ProcessOutcome:
    process.poll()
    exit_code = process.returncode
    _require(exit_code is not None, "child process has no terminal exit code")
    return ProcessOutcome(
        pid=process.pid,
        exit_code=exit_code,
        duration_s=time.monotonic() - started,
        started_at_utc=started_at_utc,
        ended_at_utc=_utc_now(),
        timed_out=timed_out,
        sigterm_sent=sigterm_sent,
        sigkill_sent=sigkill_sent,
    )


def _campaign_signal_handler(signal_number: int, _frame: Any) -> None:
    raise _CampaignSignal(signal_number)


def _run_process(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    environment: Mapping[str, str],
    timeout_s: float,
    terminate_grace_s: float,
    kill_wait_s: float,
) -> ProcessOutcome:
    started = time.monotonic()
    with (
        stdout_path.open("xb", buffering=0) as stdout_handle,
        stderr_path.open("xb", buffering=0) as stderr_handle,
    ):
        process: subprocess.Popen[bytes] | None = None
        started_at_utc = _utc_now()
        old_signal_handlers: dict[signal.Signals, Any] = {}
        for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
            signal_value = getattr(signal, signal_name, None)
            if signal_value is not None:
                old_signal_handlers[signal_value] = signal.getsignal(signal_value)
                signal.signal(signal_value, _campaign_signal_handler)
        try:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            started_at_utc = _utc_now()
            timed_out = False
            sigterm_sent = False
            sigkill_sent = False
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                sigterm_sent, sigkill_sent = _stop_process_group(
                    process,
                    terminate_grace_s=terminate_grace_s,
                    kill_wait_s=kill_wait_s,
                )
            if (
                not timed_out
                and _process_group_exists(process.pid)
                and not _wait_for_process_group_exit(
                    process, timeout_s=NATURAL_DESCENDANT_GRACE_S
                )
            ):
                sigterm_sent, sigkill_sent = _stop_process_group(
                    process,
                    terminate_grace_s=terminate_grace_s,
                    kill_wait_s=kill_wait_s,
                )
            return _outcome(
                process,
                started=started,
                started_at_utc=started_at_utc,
                timed_out=timed_out,
                sigterm_sent=sigterm_sent,
                sigkill_sent=sigkill_sent,
            )
        except BaseException as exc:
            for signal_value in old_signal_handlers:
                signal.signal(signal_value, signal.SIG_IGN)
            if process is None:
                if isinstance(exc, OSError):
                    raise
                raise CalibrationCampaignError(
                    f"campaign orchestration interrupted before child spawn: {exc}"
                ) from exc
            sigterm_sent, sigkill_sent = _stop_process_group(
                process,
                terminate_grace_s=terminate_grace_s,
                kill_wait_s=kill_wait_s,
            )
            interrupted_outcome = _outcome(
                process,
                started=started,
                started_at_utc=started_at_utc,
                timed_out=False,
                sigterm_sent=sigterm_sent,
                sigkill_sent=sigkill_sent,
            )
            raise ProcessExecutionInterrupted(
                f"campaign orchestration interrupted: {exc}", interrupted_outcome
            ) from exc
        finally:
            for signal_value, old_handler in old_signal_handlers.items():
                signal.signal(signal_value, old_handler)
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())
    raise AssertionError("child process execution returned without an outcome")


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


def _validate_completed_run(
    run_dir: Path,
    *,
    expected_implementation_manifest_sha256: str,
    expected_reproducibility_fingerprint: str,
) -> dict[str, Any]:
    root_text = str(REPO_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from tools.aggregate_m2_calibration import (  # noqa: PLC0415
        CalibrationAggregationError,
        _validate_run,
    )

    try:
        validated = _validate_run(run_dir)
    except CalibrationAggregationError as exc:
        raise CalibrationCampaignError(
            f"run artifact validation failed for {run_dir.name}: {exc}"
        ) from exc

    result = _read_json_object(run_dir / "result.json", label="result.json")
    provenance = _read_json_object(run_dir / "provenance.json", label="provenance.json")
    _require(
        result.get("gate_status") == "CALIBRATED_NOT_ACCEPTED",
        f"{run_dir.name} gate status drifted",
    )
    _require(
        result.get("within_requested_tolerance") is True,
        f"{run_dir.name} exceeded the preregistered tolerance",
    )
    _require(
        result.get("tolerance") == {"atol": CALIBRATION_ATOL, "rtol": CALIBRATION_RTOL},
        f"{run_dir.name} calibration tolerance drifted",
    )
    comparisons = result.get("comparisons")
    _require(
        isinstance(comparisons, list)
        and comparisons
        and all(row.get("allclose") is True for row in comparisons),
        f"{run_dir.name} contains a non-passing logit comparison",
    )
    implementation = provenance.get("implementation")
    _require(
        isinstance(implementation, dict),
        f"{run_dir.name} lacks implementation provenance",
    )
    observed_implementation = implementation.get("manifest_sha256")
    _require(
        observed_implementation == expected_implementation_manifest_sha256,
        f"{run_dir.name} implementation manifest drifted: {observed_implementation}",
    )
    _require(
        validated.reproducibility_fingerprint == expected_reproducibility_fingerprint,
        f"{run_dir.name} reproducibility fingerprint drifted: "
        f"{validated.reproducibility_fingerprint}",
    )
    return {
        "run_id": validated.run_id,
        "result_sha256": validated.result_sha256,
        "provenance_sha256": validated.provenance_sha256,
        "sha256sums_sha256": validated.sha256sums_sha256,
        "implementation_manifest_sha256": observed_implementation,
        "reproducibility_fingerprint": validated.reproducibility_fingerprint,
        "observed_max_abs_error": validated.observed_max_abs_error,
    }


def _validate_cohort(
    path: Path,
    *,
    expected_runs: int,
    expected_reproducibility_fingerprint: str,
) -> dict[str, Any]:
    cohort = _read_json_object(path, label="calibration cohort")
    _require(cohort.get("schema_version") == COHORT_SCHEMA, "cohort schema drifted")
    _require(
        cohort.get("protocol_schema") == PROTOCOL_SCHEMA,
        "cohort protocol schema drifted",
    )
    _require(cohort.get("run_count") == expected_runs, "cohort run count drifted")
    _require(cohort.get("all_passed") is True, "cohort did not pass")
    _require(cohort.get("failures") == [], "cohort contains failures")
    _require(
        cohort.get("formal_atol") == CALIBRATION_ATOL
        and cohort.get("formal_rtol") == CALIBRATION_RTOL,
        "cohort tolerance drifted",
    )
    _require(
        cohort.get("reproducibility_fingerprint")
        == expected_reproducibility_fingerprint,
        "cohort reproducibility fingerprint drifted",
    )
    return cohort


def _revalidate_production_candidate(
    path: Path,
    *,
    expected_implementation_manifest_sha256: str,
) -> None:
    root_text = str(REPO_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from tools.aggregate_m2_calibration import _validate_run  # noqa: PLC0415
    from tools.m2_calibration_evidence import (  # noqa: PLC0415
        CalibrationEvidenceError,
        validate_published_calibration_candidate,
    )

    try:
        validate_published_calibration_candidate(
            path,
            run_validator=_validate_run,
            expected_manifest_sha256=_sha256_file(path),
            expected_implementation_manifest_sha256=(
                expected_implementation_manifest_sha256
            ),
        )
    except CalibrationEvidenceError as exc:
        raise CalibrationCampaignError(
            f"published calibration candidate replay failed: {exc}"
        ) from exc


def _revalidate_production_bundle(
    path: Path,
    *,
    expected_implementation_manifest_sha256: str,
) -> None:
    root_text = str(REPO_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from tools.aggregate_m2_calibration import _validate_run  # noqa: PLC0415
    from tools.m2_calibration_evidence import (  # noqa: PLC0415
        CalibrationEvidenceError,
        validate_published_calibration_bundle,
    )

    try:
        validate_published_calibration_bundle(
            path,
            run_validator=_validate_run,
            expected_manifest_sha256=_sha256_file(path),
            expected_implementation_manifest_sha256=(
                expected_implementation_manifest_sha256
            ),
        )
    except CalibrationEvidenceError as exc:
        raise CalibrationCampaignError(
            f"published calibration bundle self-check failed: {exc}"
        ) from exc


def _campaign_paths(root: Path, run_name: str) -> tuple[Path, Path, Path]:
    return (
        root / run_name,
        root / f"{run_name}.stdout.log",
        root / f"{run_name}.stderr.log",
    )


def _run_command(config: CampaignConfig, run_dir: Path) -> list[str]:
    return [
        str(config.python_executable),
        str(config.runner),
        "--output-dir",
        str(run_dir),
        "--mode",
        "calibration",
        "--atol",
        str(CALIBRATION_ATOL),
        "--rtol",
        str(CALIBRATION_RTOL),
        "--model",
        str(config.model),
        "--vllm-root",
        str(config.vllm_root),
        "--cpu-bytes",
        str(config.cpu_bytes),
        "--timeout-s",
        str(config.runner_timeout_s),
        "--cuda-device",
        str(config.cuda_device),
        "--full-provenance",
    ]


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
        "started_at_utc": (outcome.started_at_utc if outcome is not None else None),
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


def _validate_expected_runs(expected_runs: int) -> None:
    _require(
        type(expected_runs) is int and 0 < expected_runs <= 999,
        "expected run count must be in [1, 999]",
    )


def _selection_rule(run_names: Sequence[str]) -> dict[str, Any]:
    return {
        "attempts_per_run": 1,
        "eligible_kind": "calibration_run",
        "ordered_run_names": list(run_names),
        "required_events": ["submitted", "terminal"],
        "required_terminal_status": "passed",
        "retry_count": 0,
        "stop_on_first_failure": True,
    }


def _normalized_config(
    config: CampaignConfig,
) -> tuple[CampaignConfig, dict[str, dict[str, Any]]]:
    expected_implementation = _validate_digest(
        config.expected_implementation_manifest_sha256,
        label="expected implementation manifest",
    )
    expected_fingerprint = _validate_digest(
        config.expected_reproducibility_fingerprint,
        label="expected reproducibility fingerprint",
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
    aggregator = _resolved_file(config.aggregator, label="aggregator")
    evidence_validator = _resolved_file(
        EVIDENCE_VALIDATOR_PATH, label="calibration evidence validator"
    )
    raw_replay = _resolved_file(RAW_REPLAY_PATH, label="raw replay validator")
    protocol = _resolved_file(PROTOCOL_PATH, label="protocol")
    launcher = _resolved_file(LAUNCHER_PATH, label="launcher")
    # Preserve the venv entry-point path. Resolving its symlink would invoke the
    # base interpreter and silently drop the vLLM environment.
    python_executable = _command_path(
        config.python_executable, label="Python executable"
    )
    model = config.model.expanduser().resolve()
    vllm_root = config.vllm_root.expanduser().resolve()
    campaign_root = config.campaign_root.expanduser().resolve()
    _require(model.is_dir(), f"model directory is missing: {model}")
    _require(vllm_root.is_dir(), f"vLLM root is missing: {vllm_root}")
    for protected in (REPO_ROOT.resolve(), model, vllm_root):
        _require(
            campaign_root != protected and not campaign_root.is_relative_to(protected),
            f"campaign root is inside protected input root: {protected}",
        )

    normalized = CampaignConfig(
        campaign_root=campaign_root,
        expected_implementation_manifest_sha256=expected_implementation,
        expected_reproducibility_fingerprint=expected_fingerprint,
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
    frozen_files = {
        "protocol": _frozen_file_entry(protocol),
        "runner": _frozen_file_entry(runner),
        "launcher": _frozen_file_entry(launcher),
        "aggregator": _frozen_file_entry(aggregator),
        "evidence": _frozen_file_entry(evidence_validator),
        "raw_replay": _frozen_file_entry(raw_replay),
    }
    return normalized, frozen_files


def _preregistration_payload(
    config: CampaignConfig,
    frozen_files: Mapping[str, Mapping[str, Any]],
    *,
    expected_runs: int,
) -> dict[str, Any]:
    run_names = [f"run-{index:03d}" for index in range(1, expected_runs + 1)]
    environment = _runner_environment(config.cuda_device)
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": f"m2-calibration-{uuid.uuid4().hex}",
        "campaign_root": str(config.campaign_root),
        "created_at_utc": _utc_now(),
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": frozen_files["protocol"]["sha256"],
        "expected_runs": expected_runs,
        "production_run_count": PRODUCTION_RUN_COUNT,
        "test_injected_run_count": expected_runs != PRODUCTION_RUN_COUNT,
        "run_names": run_names,
        "calibration_attempt_prefix_record_count": expected_runs * 2,
        "selection_rule": _selection_rule(run_names),
        "atol": CALIBRATION_ATOL,
        "rtol": CALIBRATION_RTOL,
        "expected_implementation_manifest_sha256": (
            config.expected_implementation_manifest_sha256
        ),
        "expected_reproducibility_fingerprint": (
            config.expected_reproducibility_fingerprint
        ),
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
        "cohort_output": COHORT_NAME,
        "runner_command_template": _run_command(config, Path("<RUN_DIR>")),
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
    """Create and fsync the sole preregistration artifact for a fresh root."""

    _validate_expected_runs(_expected_runs)
    frozen_config, frozen_files = _normalized_config(config)
    campaign_root = frozen_config.campaign_root
    _require(
        not os.path.lexists(campaign_root),
        f"campaign root must be brand new: {campaign_root}",
    )
    _require(
        campaign_root.parent.is_dir(),
        f"campaign parent is missing: {campaign_root.parent}",
    )
    preregistration = _preregistration_payload(
        frozen_config,
        frozen_files,
        expected_runs=_expected_runs,
    )

    campaign_root.mkdir(mode=0o750)
    _fsync_directory(campaign_root.parent)
    preregistration_path = campaign_root / PREREGISTRATION_NAME
    _write_json_atomic(preregistration_path, preregistration)
    return _sha256_file(preregistration_path)


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
    expected_preregistration_sha256 = _validate_digest(
        expected_preregistration_sha256,
        label="expected preregistration",
    )
    root = campaign_root.expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid campaign root: {root}")
    entries = sorted(entry.name for entry in root.iterdir())
    _require(
        entries == [PREREGISTRATION_NAME],
        "prepared campaign root must contain only its preregistration",
    )
    preregistration_path = root / PREREGISTRATION_NAME
    _require(
        _sha256_file(preregistration_path) == expected_preregistration_sha256,
        "prepared campaign preregistration SHA-256 does not match",
    )
    preregistration = _read_json_object(
        preregistration_path,
        label="campaign preregistration",
    )
    _require(
        preregistration.get("schema_version") == CAMPAIGN_SCHEMA,
        "campaign preregistration schema drifted",
    )
    _require(
        preregistration.get("campaign_root") == str(root),
        "prepared campaign root binding drifted",
    )
    _require(
        preregistration.get("protocol_schema") == PROTOCOL_SCHEMA,
        "prepared protocol schema drifted",
    )
    run_names = [f"run-{index:03d}" for index in range(1, expected_runs + 1)]
    _require(
        preregistration.get("expected_runs") == expected_runs
        and preregistration.get("production_run_count") == PRODUCTION_RUN_COUNT
        and preregistration.get("test_injected_run_count")
        is (expected_runs != PRODUCTION_RUN_COUNT),
        "prepared run-count freeze drifted",
    )
    _require(preregistration.get("run_names") == run_names, "run names drifted")
    _require(
        preregistration.get("calibration_attempt_prefix_record_count")
        == expected_runs * 2,
        "attempt prefix size drifted",
    )
    _require(
        preregistration.get("selection_rule") == _selection_rule(run_names),
        "selection rule drifted",
    )
    _require(
        preregistration.get("atol") == CALIBRATION_ATOL
        and preregistration.get("rtol") == CALIBRATION_RTOL,
        "calibration tolerance drifted",
    )
    _require(
        preregistration.get("retry_policy") == "none_stop_on_first_failure",
        "retry policy drifted",
    )
    _require(
        preregistration.get("attempts_file") == ATTEMPTS_NAME
        and preregistration.get("cohort_output") == COHORT_NAME,
        "prepared output names drifted",
    )

    try:
        prepared_config = CampaignConfig(
            campaign_root=root,
            expected_implementation_manifest_sha256=preregistration[
                "expected_implementation_manifest_sha256"
            ],
            expected_reproducibility_fingerprint=preregistration[
                "expected_reproducibility_fingerprint"
            ],
            python_executable=Path(preregistration["python_executable"]["path"]),
            runner=Path(preregistration["frozen_files"]["runner"]["path"]),
            aggregator=Path(preregistration["frozen_files"]["aggregator"]["path"]),
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
        raise CalibrationCampaignError(
            f"prepared campaign configuration is malformed: {exc}"
        ) from exc
    frozen_config, frozen_files = _normalized_config(prepared_config)
    _require(
        preregistration.get("frozen_files") == frozen_files,
        "prepared frozen-file inventory drifted",
    )
    _require(
        preregistration.get("protocol_sha256") == frozen_files["protocol"]["sha256"],
        "prepared protocol SHA-256 drifted",
    )
    _require(
        preregistration.get("python_executable")
        == _frozen_file_entry(frozen_config.python_executable),
        "prepared Python executable drifted",
    )
    _require(
        preregistration.get("runner_command_template")
        == _run_command(frozen_config, Path("<RUN_DIR>")),
        "prepared runner command drifted",
    )
    environment = _runner_environment(frozen_config.cuda_device)
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
        "prepared runner environment drifted",
    )
    campaign_id = preregistration.get("campaign_id")
    _require(
        isinstance(campaign_id, str)
        and campaign_id.startswith("m2-calibration-")
        and len(campaign_id.removeprefix("m2-calibration-")) == 32,
        "prepared campaign ID is malformed",
    )
    _verify_frozen_files(frozen_files)
    return (
        frozen_config,
        preregistration,
        frozen_files,
        expected_preregistration_sha256,
    )


def _sealed_calibration_prefix(
    attempts_path: Path,
    *,
    campaign_id: str,
    run_names: Sequence[str],
    preregistration_sha256: str,
) -> dict[str, Any]:
    _require(
        attempts_path.is_file() and not attempts_path.is_symlink(),
        "attempt journal is missing before aggregation",
    )
    raw = attempts_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    expected_records = len(run_names) * 2
    _require(
        len(lines) == expected_records and all(line.endswith(b"\n") for line in lines),
        "calibration attempt prefix has an invalid record boundary",
    )
    for sequence, run_name in enumerate(run_names, start=1):
        try:
            submitted = json.loads(lines[(sequence - 1) * 2])
            terminal = json.loads(lines[(sequence - 1) * 2 + 1])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationCampaignError(
                f"invalid attempt record for {run_name}: {exc}"
            ) from exc
        attempt_id = f"{campaign_id}:{run_name}"
        for record in (submitted, terminal):
            _require(isinstance(record, dict), f"invalid attempt row for {run_name}")
            _require(
                record.get("schema_version") == ATTEMPT_SCHEMA
                and record.get("campaign_id") == campaign_id
                and record.get("attempt_id") == attempt_id
                and record.get("kind") == "calibration_run"
                and record.get("sequence") == sequence
                and record.get("run_name") == run_name,
                f"attempt identity drifted for {run_name}",
            )
        _require(
            submitted.get("event") == "submitted"
            and submitted.get("preregistration_sha256") == preregistration_sha256,
            f"submitted record drifted for {run_name}",
        )
        _require(
            terminal.get("event") == "terminal"
            and terminal.get("status") == "passed"
            and type(terminal.get("pid")) is int
            and terminal["pid"] > 0,
            f"terminal record drifted for {run_name}",
        )
    return {
        "prefix_bytes": len(raw),
        "prefix_record_count": expected_records,
        "prefix_sha256": hashlib.sha256(raw).hexdigest(),
    }


def execute_prepared_campaign(
    campaign_root: Path,
    expected_preregistration_sha256: str,
    *,
    _expected_runs: int = PRODUCTION_RUN_COUNT,
) -> dict[str, Any]:
    """Execute a pristine prepared campaign whose preregistration is frozen."""

    config, preregistration, frozen_files, preregistration_sha256 = _prepared_campaign(
        campaign_root,
        expected_preregistration_sha256,
        expected_runs=_expected_runs,
    )
    root = config.campaign_root
    attempts_path = root / ATTEMPTS_NAME
    cohort_path = root / COHORT_NAME
    preregistration_path = root / PREREGISTRATION_NAME
    campaign_id = preregistration["campaign_id"]
    run_names = preregistration["run_names"]
    environment = _runner_environment(config.cuda_device)

    for sequence, run_name in enumerate(run_names, start=1):
        _verify_frozen_files(frozen_files)
        _require(
            _sha256_file(preregistration_path) == preregistration_sha256,
            "campaign preregistration changed after execution began",
        )
        run_dir, stdout_path, stderr_path = _campaign_paths(root, run_name)
        _require(
            not os.path.lexists(run_dir)
            and not os.path.lexists(stdout_path)
            and not os.path.lexists(stderr_path),
            f"fresh run paths already exist: {run_name}",
        )
        command = _run_command(config, run_dir)
        submitted = {
            "schema_version": ATTEMPT_SCHEMA,
            "campaign_id": campaign_id,
            "attempt_id": f"{campaign_id}:{run_name}",
            "kind": "calibration_run",
            "event": "submitted",
            "timestamp_utc": _utc_now(),
            "sequence": sequence,
            "run_name": run_name,
            "command": command,
            "output_dir": run_name,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "preregistration_sha256": preregistration_sha256,
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
        except (OSError, CalibrationCampaignError) as exc:
            if isinstance(exc, ProcessExecutionInterrupted):
                outcome = exc.outcome
            terminal = _terminal_payload(
                submitted,
                status=(
                    "orchestrator_interrupted"
                    if isinstance(exc, ProcessExecutionInterrupted)
                    else "spawn_or_termination_failed"
                ),
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error=str(exc),
            )
            _append_attempt(attempts_path, terminal)
            raise CalibrationCampaignError(
                f"{run_name} could not reach a process terminal: {exc}"
            ) from exc

        if outcome.timed_out:
            terminal = _terminal_payload(
                submitted,
                status="timed_out",
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error=f"process exceeded {config.process_timeout_s} seconds",
            )
            _append_attempt(attempts_path, terminal)
            raise CalibrationCampaignError(
                f"{run_name} timed out; campaign stopped without retry"
            )
        if outcome.exit_code != 0:
            terminal = _terminal_payload(
                submitted,
                status="process_failed",
                outcome=outcome,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_path=run_dir,
                error=f"runner exited with status {outcome.exit_code}",
            )
            _append_attempt(attempts_path, terminal)
            raise CalibrationCampaignError(
                f"{run_name} exited {outcome.exit_code}; campaign stopped without retry"
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
            raise CalibrationCampaignError(
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
            )
        except CalibrationCampaignError as exc:
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

    calibration_prefix = _sealed_calibration_prefix(
        attempts_path,
        campaign_id=campaign_id,
        run_names=run_names,
        preregistration_sha256=preregistration_sha256,
    )
    _verify_frozen_files(frozen_files)
    aggregate_stdout = root / "aggregate.stdout.log"
    aggregate_stderr = root / "aggregate.stderr.log"
    aggregate_command = [
        str(config.python_executable),
        str(config.aggregator),
        "--campaign-dir",
        str(root),
        "--output",
        str(cohort_path),
    ]
    aggregate_submitted = {
        "schema_version": ATTEMPT_SCHEMA,
        "campaign_id": campaign_id,
        "attempt_id": f"{campaign_id}:aggregate",
        "kind": "aggregate",
        "event": "submitted",
        "timestamp_utc": _utc_now(),
        "command": aggregate_command,
        "output": COHORT_NAME,
        "stdout": aggregate_stdout.name,
        "stderr": aggregate_stderr.name,
        "preregistration_sha256": preregistration_sha256,
        "calibration_prefix": calibration_prefix,
    }
    _append_attempt(attempts_path, aggregate_submitted)
    aggregate_outcome: ProcessOutcome | None = None
    try:
        aggregate_outcome = _run_process(
            aggregate_command,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            environment=environment,
            timeout_s=config.aggregation_timeout_s,
            terminate_grace_s=config.terminate_grace_s,
            kill_wait_s=config.kill_wait_s,
        )
    except (OSError, CalibrationCampaignError) as exc:
        if isinstance(exc, ProcessExecutionInterrupted):
            aggregate_outcome = exc.outcome
        terminal = _terminal_payload(
            aggregate_submitted,
            status=(
                "orchestrator_interrupted"
                if isinstance(exc, ProcessExecutionInterrupted)
                else "spawn_or_termination_failed"
            ),
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=cohort_path,
            error=str(exc),
        )
        _append_attempt(attempts_path, terminal)
        raise CalibrationCampaignError(
            f"aggregation could not terminate: {exc}"
        ) from exc

    if aggregate_outcome.timed_out or aggregate_outcome.exit_code != 0:
        status = "timed_out" if aggregate_outcome.timed_out else "process_failed"
        terminal = _terminal_payload(
            aggregate_submitted,
            status=status,
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=cohort_path,
            error=(
                "aggregation timed out"
                if aggregate_outcome.timed_out
                else f"aggregator exited with status {aggregate_outcome.exit_code}"
            ),
        )
        _append_attempt(attempts_path, terminal)
        raise CalibrationCampaignError(
            f"aggregation failed with status {aggregate_outcome.exit_code}"
        )
    if aggregate_outcome.sigterm_sent or aggregate_outcome.sigkill_sent:
        terminal = _terminal_payload(
            aggregate_submitted,
            status="lingering_process_group",
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=cohort_path,
            error="aggregator exited but left live descendants in its process group",
        )
        _append_attempt(attempts_path, terminal)
        raise CalibrationCampaignError(
            "aggregation left a live process group; campaign stopped"
        )

    try:
        cohort = _validate_cohort(
            cohort_path,
            expected_runs=_expected_runs,
            expected_reproducibility_fingerprint=(
                config.expected_reproducibility_fingerprint
            ),
        )
        if _expected_runs == PRODUCTION_RUN_COUNT:
            _revalidate_production_candidate(
                cohort_path,
                expected_implementation_manifest_sha256=(
                    config.expected_implementation_manifest_sha256
                ),
            )
    except CalibrationCampaignError as exc:
        terminal = _terminal_payload(
            aggregate_submitted,
            status="validation_failed",
            outcome=aggregate_outcome,
            stdout_path=aggregate_stdout,
            stderr_path=aggregate_stderr,
            artifact_path=cohort_path,
            error=str(exc),
        )
        _append_attempt(attempts_path, terminal)
        raise

    aggregate_validation = {
        "cohort_sha256": _sha256_file(cohort_path),
        "run_count": cohort["run_count"],
        "reproducibility_fingerprint": cohort["reproducibility_fingerprint"],
    }
    terminal = _terminal_payload(
        aggregate_submitted,
        status="passed",
        outcome=aggregate_outcome,
        stdout_path=aggregate_stdout,
        stderr_path=aggregate_stderr,
        artifact_path=cohort_path,
        validation=aggregate_validation,
    )
    _append_attempt(attempts_path, terminal)
    if _expected_runs == PRODUCTION_RUN_COUNT:
        _revalidate_production_bundle(
            cohort_path,
            expected_implementation_manifest_sha256=(
                config.expected_implementation_manifest_sha256
            ),
        )
    return cohort


def run_campaign(
    config: CampaignConfig,
    *,
    _expected_runs: int = PRODUCTION_RUN_COUNT,
) -> dict[str, Any]:
    """Prepare and execute one non-resumable campaign."""

    preregistration_sha256 = prepare_campaign(
        config,
        _expected_runs=_expected_runs,
    )
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
    parser.add_argument(
        "--expected-implementation-manifest-sha256",
    )
    parser.add_argument(
        "--expected-reproducibility-fingerprint",
    )
    parser.add_argument(
        "--expected-preregistration-sha256",
    )
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--aggregator", type=Path, default=DEFAULT_AGGREGATOR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--cpu-bytes", type=int, default=1 << 30)
    parser.add_argument("--runner-timeout-s", type=float, default=60.0)
    parser.add_argument("--process-timeout-s", type=float, default=1800.0)
    parser.add_argument("--aggregation-timeout-s", type=float, default=300.0)
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
            cohort = execute_prepared_campaign(
                args.campaign_root,
                args.expected_preregistration_sha256,
            )
        except (CalibrationCampaignError, OSError) as exc:
            print(f"M2 calibration campaign failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"M2 calibration campaign completed: {args.campaign_root.resolve()} "
            f"({cohort['run_count']} runs)"
        )
        return 0

    if (
        args.expected_implementation_manifest_sha256 is None
        or args.expected_reproducibility_fingerprint is None
    ):
        parser.error(
            "campaign preparation requires both expected implementation and "
            "reproducibility SHA-256 values"
        )
    if args.expected_preregistration_sha256 is not None:
        parser.error(
            "--expected-preregistration-sha256 is only valid with --execute-prepared"
        )
    config = CampaignConfig(
        campaign_root=args.campaign_root,
        expected_implementation_manifest_sha256=(
            args.expected_implementation_manifest_sha256
        ),
        expected_reproducibility_fingerprint=(
            args.expected_reproducibility_fingerprint
        ),
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
    if args.prepare_only:
        try:
            preregistration_sha256 = prepare_campaign(config)
        except (CalibrationCampaignError, OSError) as exc:
            print(f"M2 calibration preparation failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"M2 calibration campaign prepared: {config.campaign_root.resolve()} "
            f"preregistration_sha256={preregistration_sha256}"
        )
        return 0

    raise AssertionError("campaign mode dispatch is incomplete")


if __name__ == "__main__":
    raise SystemExit(main())
