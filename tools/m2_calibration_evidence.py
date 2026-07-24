"""Validate the closed evidence chain for an M2 calibration campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v2"
CAMPAIGN_SCHEMA = "dagkv.m2.calibration_campaign_preregistration.v1"
ATTEMPT_SCHEMA = "dagkv.m2.calibration_campaign_attempt.v1"
CALIBRATION_COHORT_SCHEMA = "dagkv.m2.calibration_cohort.v2"
CALIBRATION_RUN_COUNT = 59
MAX_FORMAL_ATOL = 0.125
FORMAL_RTOL = 0.0

PREREGISTRATION_NAME = "CAMPAIGN_PREREGISTRATION.json"
ATTEMPTS_NAME = "ATTEMPTS.jsonl"
MANIFEST_NAME = "M2_CALIBRATION_MANIFEST.json"
AGGREGATE_STDOUT_NAME = "aggregate.stdout.log"
AGGREGATE_STDERR_NAME = "aggregate.stderr.log"

PREREGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_root",
        "created_at_utc",
        "protocol_schema",
        "protocol_sha256",
        "expected_runs",
        "production_run_count",
        "test_injected_run_count",
        "run_names",
        "calibration_attempt_prefix_record_count",
        "selection_rule",
        "atol",
        "rtol",
        "expected_implementation_manifest_sha256",
        "expected_reproducibility_fingerprint",
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
        "cohort_output",
        "runner_command_template",
        "environment_overrides",
    }
)
SELECTION_RULE_FIELDS = frozenset(
    {
        "attempts_per_run",
        "eligible_kind",
        "ordered_run_names",
        "required_events",
        "required_terminal_status",
        "retry_count",
        "stop_on_first_failure",
    }
)
FROZEN_FILE_FIELDS = frozenset({"path", "size", "sha256"})
SUBMITTED_FIELDS = frozenset(
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
    }
)
TERMINAL_FIELDS = frozenset(
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
        "sequence",
        "run_name",
        "validation",
    }
)
TERMINAL_VALIDATION_FIELDS = frozenset(
    {
        "run_id",
        "result_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
        "implementation_manifest_sha256",
        "reproducibility_fingerprint",
        "observed_max_abs_error",
    }
)
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
        "calibration_prefix",
    }
)
AGGREGATE_TERMINAL_FIELDS = TERMINAL_FIELDS - {"sequence", "run_name"}
AGGREGATE_VALIDATION_FIELDS = frozenset(
    {"cohort_sha256", "run_count", "reproducibility_fingerprint"}
)
PREFIX_FIELDS = frozenset({"prefix_bytes", "prefix_record_count", "prefix_sha256"})
LOG_FIELDS = frozenset({"path", "size", "sha256"})
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_schema",
        "campaign_id",
        "campaign_preregistration_file",
        "campaign_preregistration_sha256",
        "attempt_file",
        "attempt_prefix_bytes",
        "attempt_prefix_record_count",
        "attempt_prefix_sha256",
        "protocol_sha256",
        "implementation_manifest_sha256",
        "selection_rule",
        "pilot_excluded",
        "attempt_count",
        "run_count",
        "all_passed",
        "failures",
        "observed_max_abs_error",
        "formal_atol",
        "formal_rtol",
        "reproducibility_fingerprint",
        "runs",
    }
)
MANIFEST_RUN_FIELDS = frozenset(
    {
        "sequence",
        "run_name",
        "attempt_id",
        "run_id",
        "result_sha256",
        "provenance_sha256",
        "sha256sums_sha256",
        "observed_max_abs_error",
    }
)


class CalibrationEvidenceError(RuntimeError):
    """Raised when a calibration campaign has an open or altered evidence set."""


class ValidatedRunLike(Protocol):
    run_id: str
    result_sha256: str
    provenance_sha256: str
    sha256sums_sha256: str
    reproducibility_fingerprint: str
    implementation_manifest_sha256: str
    observed_max_abs_error: float


@dataclass(frozen=True, slots=True)
class EvidenceRun:
    sequence: int
    run_name: str
    attempt_id: str
    run_id: str
    result_sha256: str
    provenance_sha256: str
    sha256sums_sha256: str
    reproducibility_fingerprint: str
    implementation_manifest_sha256: str
    observed_max_abs_error: float

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "run_name": self.run_name,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "result_sha256": self.result_sha256,
            "provenance_sha256": self.provenance_sha256,
            "sha256sums_sha256": self.sha256sums_sha256,
            "observed_max_abs_error": self.observed_max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class CampaignEvidence:
    campaign_root: Path
    campaign_id: str
    preregistration_sha256: str
    attempt_prefix_bytes: int
    attempt_prefix_record_count: int
    attempt_prefix_sha256: str
    protocol_sha256: str
    implementation_manifest_sha256: str
    reproducibility_fingerprint: str
    selection_rule: dict[str, Any]
    runs: tuple[EvidenceRun, ...]

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_COHORT_SCHEMA,
            "protocol_schema": PROTOCOL_SCHEMA,
            "campaign_id": self.campaign_id,
            "campaign_preregistration_file": PREREGISTRATION_NAME,
            "campaign_preregistration_sha256": self.preregistration_sha256,
            "attempt_file": ATTEMPTS_NAME,
            "attempt_prefix_bytes": self.attempt_prefix_bytes,
            "attempt_prefix_record_count": self.attempt_prefix_record_count,
            "attempt_prefix_sha256": self.attempt_prefix_sha256,
            "protocol_sha256": self.protocol_sha256,
            "implementation_manifest_sha256": (self.implementation_manifest_sha256),
            "selection_rule": self.selection_rule,
            "pilot_excluded": True,
            "attempt_count": len(self.runs),
            "run_count": len(self.runs),
            "all_passed": True,
            "failures": [],
            "observed_max_abs_error": max(
                run.observed_max_abs_error for run in self.runs
            ),
            "formal_atol": MAX_FORMAL_ATOL,
            "formal_rtol": FORMAL_RTOL,
            "reproducibility_fingerprint": self.reproducibility_fingerprint,
            "runs": [run.manifest_entry() for run in self.runs],
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationEvidenceError(message)


def lower_sha256(value: Any, *, label: str) -> str:
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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in payload, f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> None:
    raise CalibrationEvidenceError(f"non-finite JSON constant: {value}")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _campaign_tree_identity(campaign_root: Path) -> tuple[tuple[Any, ...], ...]:
    entries: list[tuple[Any, ...]] = []
    try:
        paths = sorted((campaign_root, *campaign_root.rglob("*")))
        for path in paths:
            value = path.lstat()
            if stat.S_ISREG(value.st_mode):
                require(
                    value.st_nlink == 1,
                    f"campaign file has an external hard link: {path}",
                )
            relative = (
                "."
                if path == campaign_root
                else path.relative_to(campaign_root).as_posix()
            )
            entries.append(
                (
                    relative,
                    value.st_mode,
                    value.st_dev,
                    value.st_ino,
                    value.st_nlink,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )
            )
    except OSError as exc:
        raise CalibrationEvidenceError(
            f"cannot snapshot calibration campaign: {exc}"
        ) from exc
    return tuple(entries)


def read_stable_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode), f"{label} must be regular: {path}")
        payload = path.read_bytes()
        after = path.lstat()
    except CalibrationEvidenceError:
        raise
    except OSError as exc:
        raise CalibrationEvidenceError(f"cannot read {label} at {path}: {exc}") from exc
    require(
        _stat_identity(before) == _stat_identity(after),
        f"{label} changed while read: {path}",
    )
    require(len(payload) == before.st_size, f"short read from {label}: {path}")
    return payload


def _read_stable_executable(path: Path, *, label: str) -> bytes:
    """Read a regular executable or one stable venv entry-point symlink."""

    try:
        link_before = path.lstat()
    except OSError as exc:
        raise CalibrationEvidenceError(f"cannot stat {label} at {path}: {exc}") from exc
    if stat.S_ISREG(link_before.st_mode):
        return read_stable_bytes(path, label=label)
    require(stat.S_ISLNK(link_before.st_mode), f"{label} has an invalid file type")
    try:
        target_before = os.readlink(path)
        resolved_before = path.resolve(strict=True)
        payload = read_stable_bytes(resolved_before, label=f"{label} target")
        target_after = os.readlink(path)
        resolved_after = path.resolve(strict=True)
        link_after = path.lstat()
    except CalibrationEvidenceError:
        raise
    except OSError as exc:
        raise CalibrationEvidenceError(
            f"cannot resolve {label} symlink at {path}: {exc}"
        ) from exc
    require(
        _stat_identity(link_before) == _stat_identity(link_after)
        and target_before == target_after
        and resolved_before == resolved_after,
        f"{label} symlink changed while read",
    )
    return payload


def sha256_file(path: Path, *, label: str = "file") -> str:
    return hashlib.sha256(read_stable_bytes(path, label=label)).hexdigest()


def read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = read_stable_bytes(path, label=label)
    require(raw.endswith(b"\n"), f"{label} is unterminated: {path}")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CalibrationEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationEvidenceError(f"invalid {label} at {path}: {exc}") from exc
    require(isinstance(payload, dict), f"{label} must be an object: {path}")
    return payload, raw


def _timestamp(value: Any, *, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} must be an ISO timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalibrationEvidenceError(f"{label} must be an ISO timestamp") from exc
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        f"{label} must include a timezone",
    )
    return timestamp.astimezone(UTC)


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


def _validate_frozen_entry(
    value: Any,
    *,
    label: str,
    allow_executable_symlink: bool = False,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} entry must be an object")
    require(set(value) == FROZEN_FILE_FIELDS, f"{label} fields differ")
    path_value = value.get("path")
    size = value.get("size")
    require(isinstance(path_value, str) and path_value, f"{label} path is invalid")
    require(type(size) is int and size >= 0, f"{label} size is invalid")
    expected = lower_sha256(value.get("sha256"), label=f"{label} SHA-256")
    path = Path(path_value)
    require(path.is_absolute(), f"{label} path must be absolute")
    raw = (
        _read_stable_executable(path, label=label)
        if allow_executable_symlink
        else read_stable_bytes(path, label=label)
    )
    require(len(raw) == size, f"{label} size drifted")
    require(hashlib.sha256(raw).hexdigest() == expected, f"{label} SHA-256 drifted")
    return value


def _runner_command_template(preregistration: Mapping[str, Any]) -> list[str]:
    frozen_files = preregistration["frozen_files"]
    python = preregistration["python_executable"]
    return [
        python["path"],
        frozen_files["runner"]["path"],
        "--output-dir",
        "<RUN_DIR>",
        "--mode",
        "calibration",
        "--atol",
        str(MAX_FORMAL_ATOL),
        "--rtol",
        str(FORMAL_RTOL),
        "--model",
        preregistration["model"],
        "--vllm-root",
        preregistration["vllm_root"],
        "--cpu-bytes",
        str(preregistration["cpu_bytes"]),
        "--timeout-s",
        str(preregistration["runner_timeout_s"]),
        "--cuda-device",
        str(preregistration["cuda_device"]),
        "--full-provenance",
    ]


def _validate_preregistration(
    campaign_root: Path,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    path = campaign_root / PREREGISTRATION_NAME
    payload, raw = read_json_object(path, label="campaign preregistration")
    require(set(payload) == PREREGISTRATION_FIELDS, "preregistration fields differ")
    require(payload.get("schema_version") == CAMPAIGN_SCHEMA, "wrong campaign schema")
    campaign_id = payload.get("campaign_id")
    require(
        isinstance(campaign_id, str)
        and campaign_id.startswith("m2-calibration-")
        and len(campaign_id.removeprefix("m2-calibration-")) == 32,
        "campaign_id is malformed",
    )
    _timestamp(payload.get("created_at_utc"), label="campaign created_at_utc")
    require(payload.get("campaign_root") == str(campaign_root), "campaign root drifted")
    require(
        payload.get("protocol_schema") == PROTOCOL_SCHEMA, "protocol schema drifted"
    )
    require(
        payload.get("expected_runs") == CALIBRATION_RUN_COUNT
        and payload.get("production_run_count") == CALIBRATION_RUN_COUNT
        and payload.get("test_injected_run_count") is False,
        f"campaign must preregister exactly {CALIBRATION_RUN_COUNT} production runs",
    )
    run_names = tuple(
        f"run-{index:03d}" for index in range(1, CALIBRATION_RUN_COUNT + 1)
    )
    require(payload.get("run_names") == list(run_names), "run-name freeze drifted")
    require(
        payload.get("calibration_attempt_prefix_record_count")
        == CALIBRATION_RUN_COUNT * 2,
        "attempt-prefix record count drifted",
    )
    selection_rule = payload.get("selection_rule")
    require(isinstance(selection_rule, dict), "selection_rule must be an object")
    require(
        set(selection_rule) == SELECTION_RULE_FIELDS, "selection_rule fields differ"
    )
    require(selection_rule == _selection_rule(run_names), "selection_rule drifted")
    require(
        payload.get("atol") == MAX_FORMAL_ATOL and payload.get("rtol") == FORMAL_RTOL,
        "preregistered tolerance drifted",
    )
    lower_sha256(
        payload.get("expected_implementation_manifest_sha256"),
        label="expected implementation manifest",
    )
    lower_sha256(
        payload.get("expected_reproducibility_fingerprint"),
        label="expected reproducibility fingerprint",
    )
    frozen_files = payload.get("frozen_files")
    require(isinstance(frozen_files, dict), "frozen_files must be an object")
    require(
        set(frozen_files)
        == {
            "protocol",
            "runner",
            "launcher",
            "aggregator",
            "evidence",
            "raw_replay",
        },
        "frozen-file labels differ",
    )
    for label, entry in frozen_files.items():
        _validate_frozen_entry(entry, label=f"frozen {label}")
    _validate_frozen_entry(
        payload.get("python_executable"),
        label="Python executable",
        allow_executable_symlink=True,
    )
    require(
        payload.get("protocol_sha256") == frozen_files["protocol"]["sha256"],
        "protocol SHA-256 binding drifted",
    )
    for label in ("model", "vllm_root"):
        value = payload.get(label)
        require(isinstance(value, str) and Path(value).is_dir(), f"{label} is invalid")
    require(
        type(payload.get("cpu_bytes")) is int and payload["cpu_bytes"] > 0,
        "cpu_bytes is invalid",
    )
    require(
        type(payload.get("cuda_device")) is int and payload["cuda_device"] >= 0,
        "cuda_device is invalid",
    )
    for label in (
        "runner_timeout_s",
        "process_timeout_s",
        "aggregation_timeout_s",
        "terminate_grace_s",
        "kill_wait_s",
    ):
        require(
            _finite_number(payload.get(label), label=label) > 0,
            f"{label} must be positive",
        )
    require(
        payload.get("retry_policy") == "none_stop_on_first_failure",
        "retry policy drifted",
    )
    require(payload.get("attempts_file") == ATTEMPTS_NAME, "attempt filename drifted")
    require(payload.get("cohort_output") == MANIFEST_NAME, "manifest filename drifted")
    require(
        payload.get("runner_command_template") == _runner_command_template(payload),
        "runner command template drifted",
    )
    environment = payload.get("environment_overrides")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": str(payload["cuda_device"]),
        "HF_HUB_OFFLINE": "1",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(
            Path(__file__).resolve().parents[1] / "integrations" / "vllm_m2"
        ),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    require(environment == expected_environment, "runner environment drifted")
    return payload, hashlib.sha256(raw).hexdigest(), run_names


def _actual_artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for artifact in sorted(run_dir.rglob("*")):
        relative = artifact.relative_to(run_dir).as_posix()
        require(
            not artifact.is_symlink(), f"run artifact cannot be a symlink: {artifact}"
        )
        if artifact.is_file():
            inventory.append(
                {"path": relative, "type": "file", "size": artifact.stat().st_size}
            )
        elif artifact.is_dir():
            inventory.append({"path": relative, "type": "directory"})
    return inventory


def _validate_log(value: Any, *, root: Path, expected_name: str, label: str) -> None:
    require(
        isinstance(value, dict) and set(value) == LOG_FIELDS, f"{label} fields differ"
    )
    require(value.get("path") == expected_name, f"{label} path drifted")
    size = value.get("size")
    require(type(size) is int and size >= 0, f"{label} size is invalid")
    digest = lower_sha256(value.get("sha256"), label=f"{label} SHA-256")
    raw = read_stable_bytes(root / expected_name, label=label)
    require(len(raw) == size, f"{label} size drifted")
    require(hashlib.sha256(raw).hexdigest() == digest, f"{label} SHA-256 drifted")


def _validate_run_records(
    *,
    campaign_root: Path,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    run_name: str,
    sequence: int,
    submitted: Any,
    terminal: Any,
    validated: ValidatedRunLike,
    not_before: datetime,
) -> tuple[EvidenceRun, datetime]:
    campaign_id = preregistration["campaign_id"]
    attempt_id = f"{campaign_id}:{run_name}"
    require(isinstance(submitted, dict), f"{run_name} submitted row is invalid")
    require(set(submitted) == SUBMITTED_FIELDS, f"{run_name} submitted fields differ")
    require(isinstance(terminal, dict), f"{run_name} terminal row is invalid")
    require(set(terminal) == TERMINAL_FIELDS, f"{run_name} terminal fields differ")
    for record in (submitted, terminal):
        require(
            record.get("schema_version") == ATTEMPT_SCHEMA, "attempt schema drifted"
        )
        require(record.get("campaign_id") == campaign_id, "attempt campaign drifted")
        require(record.get("attempt_id") == attempt_id, "attempt identity drifted")
        require(record.get("kind") == "calibration_run", "attempt kind drifted")
        require(record.get("sequence") == sequence, "attempt sequence drifted")
        require(record.get("run_name") == run_name, "attempt run_name drifted")
    require(submitted.get("event") == "submitted", "submitted event drifted")
    require(terminal.get("event") == "terminal", "terminal event drifted")
    submitted_at = _timestamp(
        submitted.get("timestamp_utc"), label=f"{run_name} submitted timestamp"
    )
    started_at = _timestamp(
        terminal.get("started_at_utc"), label=f"{run_name} process start"
    )
    ended_at = _timestamp(terminal.get("ended_at_utc"), label=f"{run_name} process end")
    terminal_at = _timestamp(
        terminal.get("timestamp_utc"), label=f"{run_name} terminal timestamp"
    )
    require(
        submitted_at <= started_at <= ended_at <= terminal_at,
        f"{run_name} process timestamps are out of order",
    )
    require(
        submitted_at >= not_before,
        f"{run_name} was submitted before its frozen predecessor",
    )
    require(
        terminal.get("status") == "passed"
        and type(terminal.get("pid")) is int
        and terminal["pid"] > 0
        and terminal.get("exit_code") == 0
        and _finite_number(terminal.get("duration_s"), label="duration_s") >= 0
        and terminal.get("timed_out") is False
        and terminal.get("sigterm_sent") is False
        and terminal.get("sigkill_sent") is False
        and terminal.get("error") is None,
        f"{run_name} did not have one clean process terminal",
    )
    run_dir = campaign_root / run_name
    expected_command = list(preregistration["runner_command_template"])
    expected_command[3] = str(run_dir)
    require(submitted.get("command") == expected_command, f"{run_name} command drifted")
    require(submitted.get("output_dir") == run_name, f"{run_name} output drifted")
    stdout_name = f"{run_name}.stdout.log"
    stderr_name = f"{run_name}.stderr.log"
    require(submitted.get("stdout") == stdout_name, f"{run_name} stdout drifted")
    require(submitted.get("stderr") == stderr_name, f"{run_name} stderr drifted")
    require(
        submitted.get("preregistration_sha256") == preregistration_sha256,
        f"{run_name} preregistration binding drifted",
    )
    _validate_log(
        terminal.get("stdout"),
        root=campaign_root,
        expected_name=stdout_name,
        label=f"{run_name} stdout",
    )
    _validate_log(
        terminal.get("stderr"),
        root=campaign_root,
        expected_name=stderr_name,
        label=f"{run_name} stderr",
    )
    require(
        terminal.get("artifact_inventory") == _actual_artifact_inventory(run_dir),
        f"{run_name} artifact inventory drifted",
    )
    validation = terminal.get("validation")
    require(isinstance(validation, dict), f"{run_name} validation is missing")
    require(
        set(validation) == TERMINAL_VALIDATION_FIELDS,
        f"{run_name} validation fields differ",
    )
    expected_validation = {
        "run_id": validated.run_id,
        "result_sha256": validated.result_sha256,
        "provenance_sha256": validated.provenance_sha256,
        "sha256sums_sha256": validated.sha256sums_sha256,
        "implementation_manifest_sha256": (validated.implementation_manifest_sha256),
        "reproducibility_fingerprint": validated.reproducibility_fingerprint,
        "observed_max_abs_error": validated.observed_max_abs_error,
    }
    require(validation == expected_validation, f"{run_name} validation mapping drifted")
    protocol_copy = run_dir / "protocol.md"
    require(
        sha256_file(protocol_copy, label=f"{run_name} protocol")
        == preregistration["protocol_sha256"],
        f"{run_name} protocol copy drifted",
    )
    return (
        EvidenceRun(
            sequence=sequence,
            run_name=run_name,
            attempt_id=attempt_id,
            run_id=validated.run_id,
            result_sha256=validated.result_sha256,
            provenance_sha256=validated.provenance_sha256,
            sha256sums_sha256=validated.sha256sums_sha256,
            reproducibility_fingerprint=validated.reproducibility_fingerprint,
            implementation_manifest_sha256=(validated.implementation_manifest_sha256),
            observed_max_abs_error=validated.observed_max_abs_error,
        ),
        terminal_at,
    )


def _parse_attempt_lines(raw: bytes) -> list[Any]:
    require(raw.endswith(b"\n"), "attempt journal is unterminated")
    require(b"\r" not in raw, "attempt journal must use LF newlines")
    lines = raw.splitlines(keepends=True)
    require(
        lines and all(line.strip() for line in lines), "attempt journal has blank rows"
    )
    records: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except CalibrationEvidenceError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationEvidenceError(
                f"invalid attempt row {line_number}: {exc}"
            ) from exc
        require(isinstance(record, dict), f"attempt row {line_number} is not an object")
        records.append(record)
    return records


def _validate_prefix(value: Any, *, prefix_raw: bytes) -> dict[str, Any]:
    require(isinstance(value, dict), "aggregate calibration_prefix is invalid")
    require(set(value) == PREFIX_FIELDS, "aggregate calibration_prefix fields differ")
    expected = {
        "prefix_bytes": len(prefix_raw),
        "prefix_record_count": CALIBRATION_RUN_COUNT * 2,
        "prefix_sha256": hashlib.sha256(prefix_raw).hexdigest(),
    }
    require(value == expected, "aggregate calibration_prefix drifted")
    return expected


def _validate_aggregate_submitted(
    record: Any,
    *,
    campaign_root: Path,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    prefix_raw: bytes,
    last_calibration_terminal_at: datetime,
) -> tuple[dict[str, Any], datetime]:
    require(isinstance(record, dict), "aggregate submitted row is invalid")
    require(
        set(record) == AGGREGATE_SUBMITTED_FIELDS, "aggregate submitted fields differ"
    )
    campaign_id = preregistration["campaign_id"]
    require(record.get("schema_version") == ATTEMPT_SCHEMA, "aggregate schema drifted")
    require(record.get("campaign_id") == campaign_id, "aggregate campaign drifted")
    require(
        record.get("attempt_id") == f"{campaign_id}:aggregate", "aggregate ID drifted"
    )
    require(record.get("kind") == "aggregate", "aggregate kind drifted")
    require(record.get("event") == "submitted", "aggregate event drifted")
    timestamp = _timestamp(record.get("timestamp_utc"), label="aggregate submitted")
    require(
        timestamp >= last_calibration_terminal_at,
        "aggregate started before run-059 terminal",
    )
    manifest_path = campaign_root / MANIFEST_NAME
    expected_command = [
        preregistration["python_executable"]["path"],
        preregistration["frozen_files"]["aggregator"]["path"],
        "--campaign-dir",
        str(campaign_root),
        "--output",
        str(manifest_path),
    ]
    require(record.get("command") == expected_command, "aggregate command drifted")
    require(record.get("output") == MANIFEST_NAME, "aggregate output drifted")
    require(record.get("stdout") == AGGREGATE_STDOUT_NAME, "aggregate stdout drifted")
    require(record.get("stderr") == AGGREGATE_STDERR_NAME, "aggregate stderr drifted")
    require(
        record.get("preregistration_sha256") == preregistration_sha256,
        "aggregate preregistration binding drifted",
    )
    prefix = _validate_prefix(record.get("calibration_prefix"), prefix_raw=prefix_raw)
    return prefix, timestamp


def _validate_aggregate_terminal(
    record: Any,
    *,
    campaign_root: Path,
    campaign_id: str,
    submitted_at: datetime,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> None:
    require(isinstance(record, dict), "aggregate terminal row is invalid")
    require(
        set(record) == AGGREGATE_TERMINAL_FIELDS, "aggregate terminal fields differ"
    )
    require(record.get("schema_version") == ATTEMPT_SCHEMA, "aggregate schema drifted")
    require(record.get("campaign_id") == campaign_id, "aggregate campaign drifted")
    require(
        record.get("attempt_id") == f"{campaign_id}:aggregate", "aggregate ID drifted"
    )
    require(record.get("kind") == "aggregate", "aggregate kind drifted")
    require(record.get("event") == "terminal", "aggregate terminal event drifted")
    started_at = _timestamp(
        record.get("started_at_utc"), label="aggregate process start"
    )
    ended_at = _timestamp(record.get("ended_at_utc"), label="aggregate process end")
    terminal_at = _timestamp(record.get("timestamp_utc"), label="aggregate terminal")
    require(
        submitted_at <= started_at <= ended_at <= terminal_at,
        "aggregate timestamps are out of order",
    )
    require(
        record.get("status") == "passed"
        and type(record.get("pid")) is int
        and record["pid"] > 0
        and record.get("exit_code") == 0
        and _finite_number(record.get("duration_s"), label="aggregate duration") >= 0
        and record.get("timed_out") is False
        and record.get("sigterm_sent") is False
        and record.get("sigkill_sent") is False
        and record.get("error") is None,
        "aggregate process did not have one clean terminal",
    )
    _validate_log(
        record.get("stdout"),
        root=campaign_root,
        expected_name=AGGREGATE_STDOUT_NAME,
        label="aggregate stdout",
    )
    _validate_log(
        record.get("stderr"),
        root=campaign_root,
        expected_name=AGGREGATE_STDERR_NAME,
        label="aggregate stderr",
    )
    manifest_path = campaign_root / MANIFEST_NAME
    expected_inventory = [
        {"path": MANIFEST_NAME, "type": "file", "size": manifest_path.stat().st_size}
    ]
    require(
        record.get("artifact_inventory") == expected_inventory,
        "aggregate artifact inventory drifted",
    )
    validation = record.get("validation")
    require(isinstance(validation, dict), "aggregate validation is missing")
    require(
        set(validation) == AGGREGATE_VALIDATION_FIELDS,
        "aggregate validation fields differ",
    )
    require(
        validation
        == {
            "cohort_sha256": manifest_sha256,
            "run_count": CALIBRATION_RUN_COUNT,
            "reproducibility_fingerprint": manifest["reproducibility_fingerprint"],
        },
        "aggregate validation mapping drifted",
    )


def _validate_root_entries(campaign_root: Path, *, published: bool) -> None:
    run_names = {f"run-{index:03d}" for index in range(1, CALIBRATION_RUN_COUNT + 1)}
    files = {
        PREREGISTRATION_NAME,
        ATTEMPTS_NAME,
        AGGREGATE_STDOUT_NAME,
        AGGREGATE_STDERR_NAME,
    }
    for run_name in run_names:
        files.add(f"{run_name}.stdout.log")
        files.add(f"{run_name}.stderr.log")
    if published:
        files.add(MANIFEST_NAME)
    expected = files | run_names
    actual = {path.name for path in campaign_root.iterdir()}
    require(actual == expected, "campaign root contains undeclared entries")
    for path in campaign_root.rglob("*"):
        require(not path.is_symlink(), f"campaign entry cannot be a symlink: {path}")


def validate_campaign_for_aggregation(
    campaign_root: Path,
    *,
    run_validator: Callable[[Path], ValidatedRunLike],
) -> CampaignEvidence:
    """Validate the exact pre-publication state seen by the aggregator."""

    return _validate_campaign(
        campaign_root,
        run_validator=run_validator,
        manifest=None,
        manifest_sha256=None,
        require_aggregate_terminal=False,
    )


def _validate_campaign(
    campaign_root: Path,
    *,
    run_validator: Callable[[Path], ValidatedRunLike],
    manifest: Mapping[str, Any] | None,
    manifest_sha256: str | None,
    require_aggregate_terminal: bool,
) -> CampaignEvidence:
    campaign_root = campaign_root.resolve()
    require(campaign_root.is_dir(), f"campaign directory is missing: {campaign_root}")
    published = manifest is not None
    _validate_root_entries(campaign_root, published=published)
    preregistration, preregistration_sha256, run_names = _validate_preregistration(
        campaign_root
    )
    attempts_raw = read_stable_bytes(
        campaign_root / ATTEMPTS_NAME,
        label="attempt journal",
    )
    records = _parse_attempt_lines(attempts_raw)
    expected_record_count = CALIBRATION_RUN_COUNT * 2 + 1
    if require_aggregate_terminal:
        require(published, "aggregate terminal requires a published manifest")
        expected_record_count += 1
    require(
        len(records) == expected_record_count,
        "attempt journal has an unexpected record count",
    )
    lines = attempts_raw.splitlines(keepends=True)
    prefix_raw = b"".join(lines[: CALIBRATION_RUN_COUNT * 2])
    evidence_runs: list[EvidenceRun] = []
    previous_terminal_at = _timestamp(
        preregistration["created_at_utc"],
        label="campaign created_at_utc",
    )
    for sequence, run_name in enumerate(run_names, start=1):
        validated = run_validator(campaign_root / run_name)
        evidence_run, previous_terminal_at = _validate_run_records(
            campaign_root=campaign_root,
            preregistration=preregistration,
            preregistration_sha256=preregistration_sha256,
            run_name=run_name,
            sequence=sequence,
            submitted=records[(sequence - 1) * 2],
            terminal=records[(sequence - 1) * 2 + 1],
            validated=validated,
            not_before=previous_terminal_at,
        )
        evidence_runs.append(evidence_run)
    prefix, aggregate_submitted_at = _validate_aggregate_submitted(
        records[CALIBRATION_RUN_COUNT * 2],
        campaign_root=campaign_root,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
        prefix_raw=prefix_raw,
        last_calibration_terminal_at=previous_terminal_at,
    )
    require(
        all(
            record.get("kind") != "calibration_run"
            for record in records[CALIBRATION_RUN_COUNT * 2 :]
        ),
        "calibration attempt appears after the sealed prefix",
    )

    run_ids = {run.run_id for run in evidence_runs}
    result_hashes = {run.result_sha256 for run in evidence_runs}
    fingerprints = {run.reproducibility_fingerprint for run in evidence_runs}
    implementations = {run.implementation_manifest_sha256 for run in evidence_runs}
    require(len(run_ids) == CALIBRATION_RUN_COUNT, "calibration run IDs are not unique")
    require(
        len(result_hashes) == CALIBRATION_RUN_COUNT,
        "calibration result hashes are not unique",
    )
    require(len(fingerprints) == 1, "calibration fingerprints differ")
    require(len(implementations) == 1, "calibration implementations differ")
    fingerprint = next(iter(fingerprints))
    implementation = next(iter(implementations))
    require(
        fingerprint == preregistration["expected_reproducibility_fingerprint"],
        "campaign fingerprint differs from preregistration",
    )
    require(
        implementation == preregistration["expected_implementation_manifest_sha256"],
        "campaign implementation differs from preregistration",
    )
    evidence = CampaignEvidence(
        campaign_root=campaign_root,
        campaign_id=preregistration["campaign_id"],
        preregistration_sha256=preregistration_sha256,
        attempt_prefix_bytes=prefix["prefix_bytes"],
        attempt_prefix_record_count=prefix["prefix_record_count"],
        attempt_prefix_sha256=prefix["prefix_sha256"],
        protocol_sha256=preregistration["protocol_sha256"],
        implementation_manifest_sha256=implementation,
        reproducibility_fingerprint=fingerprint,
        selection_rule=dict(preregistration["selection_rule"]),
        runs=tuple(evidence_runs),
    )
    if published:
        assert manifest is not None
        assert manifest_sha256 is not None
        _validate_manifest_mapping(manifest, evidence=evidence)
    if require_aggregate_terminal:
        assert manifest is not None
        assert manifest_sha256 is not None
        _validate_aggregate_terminal(
            records[-1],
            campaign_root=campaign_root,
            campaign_id=evidence.campaign_id,
            submitted_at=aggregate_submitted_at,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
    return evidence


def _validate_manifest_mapping(
    manifest: Mapping[str, Any], *, evidence: CampaignEvidence
) -> None:
    require(set(manifest) == MANIFEST_FIELDS, "calibration manifest fields differ")
    require(
        manifest.get("schema_version") == CALIBRATION_COHORT_SCHEMA,
        "wrong calibration manifest schema",
    )
    require(
        manifest.get("protocol_schema") == PROTOCOL_SCHEMA, "protocol schema drifted"
    )
    expected = evidence.manifest_payload()
    require(dict(manifest) == expected, "calibration manifest evidence mapping drifted")
    runs = manifest.get("runs")
    assert isinstance(runs, list)
    for index, run in enumerate(runs, start=1):
        require(isinstance(run, dict), f"manifest run {index} is invalid")
        require(set(run) == MANIFEST_RUN_FIELDS, f"manifest run {index} fields differ")
        require(run.get("sequence") == index, f"manifest run {index} sequence drifted")


def _validate_published(
    manifest_path: Path,
    *,
    run_validator: Callable[[Path], ValidatedRunLike],
    expected_manifest_sha256: str | None = None,
    expected_implementation_manifest_sha256: str | None = None,
    require_aggregate_terminal: bool,
) -> tuple[dict[str, Any], str, CampaignEvidence]:
    manifest_path = manifest_path.absolute()
    require(manifest_path.name == MANIFEST_NAME, "calibration manifest name drifted")
    campaign_root = manifest_path.parent.resolve()
    tree_before = _campaign_tree_identity(campaign_root)
    manifest, raw = read_json_object(manifest_path, label="calibration manifest")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None:
        require(
            manifest_sha256
            == lower_sha256(
                expected_manifest_sha256,
                label="expected calibration manifest SHA-256",
            ),
            "calibration manifest SHA-256 differs from the expected digest",
        )
    evidence = _validate_campaign(
        campaign_root,
        run_validator=run_validator,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        require_aggregate_terminal=require_aggregate_terminal,
    )
    if expected_implementation_manifest_sha256 is not None:
        require(
            evidence.implementation_manifest_sha256
            == lower_sha256(
                expected_implementation_manifest_sha256,
                label="current implementation manifest SHA-256",
            ),
            "current implementation differs from calibration implementation",
        )
    require(
        hashlib.sha256(
            read_stable_bytes(manifest_path, label="calibration manifest")
        ).hexdigest()
        == manifest_sha256,
        "calibration manifest changed during validation",
    )
    require(
        _campaign_tree_identity(campaign_root) == tree_before,
        "calibration campaign changed during validation",
    )
    return manifest, manifest_sha256, evidence


def validate_published_calibration_candidate(
    manifest_path: Path,
    *,
    run_validator: Callable[[Path], ValidatedRunLike],
    expected_manifest_sha256: str | None = None,
    expected_implementation_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], str, CampaignEvidence]:
    """Revalidate a published manifest before its aggregate terminal exists."""

    return _validate_published(
        manifest_path,
        run_validator=run_validator,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_implementation_manifest_sha256=(
            expected_implementation_manifest_sha256
        ),
        require_aggregate_terminal=False,
    )


def validate_published_calibration_bundle(
    manifest_path: Path,
    *,
    run_validator: Callable[[Path], ValidatedRunLike],
    expected_manifest_sha256: str | None = None,
    expected_implementation_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], str, CampaignEvidence]:
    """Revalidate a published manifest and every upstream campaign artifact."""

    return _validate_published(
        manifest_path,
        run_validator=run_validator,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_implementation_manifest_sha256=(
            expected_implementation_manifest_sha256
        ),
        require_aggregate_terminal=True,
    )
