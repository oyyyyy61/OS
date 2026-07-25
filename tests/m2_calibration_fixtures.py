"""Complete CPU-only fixtures for the M2 calibration evidence chain."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import tools.m2_calibration_evidence as calibration_evidence_module
import tools.run_m2_calibration_campaign as calibration_campaign_module
from tests.m2_raw_replay_fixtures import (
    NVIDIA_KERNEL_VERSION,
    build_raw_run,
)
from tools.aggregate_m2_calibration import _validate_run, aggregate_campaign
from tools.nvidia_driver_userspace_bundle import (
    BundleValidation,
    NvidiaUserspaceBundleError,
    RuntimeMapping,
)
from tools.run_m2_calibration_campaign import (
    ATTEMPT_SCHEMA,
    ATTEMPTS_NAME,
    COHORT_NAME,
    PREREGISTRATION_NAME,
    CampaignConfig,
    prepare_campaign,
)
from tools.run_m2_vllm_abba import _implementation_capture

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "research" / "protocols" / "M2_VLLM_REPLAY_PROTOCOL.md"
_REAL_VALIDATE_BUNDLE = calibration_campaign_module.validate_bundle
_FAKE_BUNDLES: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class CalibrationFixture:
    campaign_root: Path
    manifest_path: Path
    preregistration_sha256: str
    implementation_manifest_sha256: str
    reproducibility_fingerprint: str


def _fixture_bundle_validator(
    bundle_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    **kwargs: object,
) -> BundleValidation:
    root = Path(os.path.abspath(bundle_root))
    fixture = _FAKE_BUNDLES.get(str(root))
    if fixture is None:
        return _REAL_VALIDATE_BUNDLE(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
            **kwargs,
        )
    try:
        manifest_raw = (root / "NVIDIA_USERSPACE_BUNDLE_MANIFEST.json").read_bytes()
        sentinel = (root / "fixture-content.bin").read_bytes()
    except OSError as exc:
        raise NvidiaUserspaceBundleError("fixture bundle is unavailable") from exc
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if expected_manifest_sha256 is not None and (
        manifest_sha256 != expected_manifest_sha256
    ):
        raise NvidiaUserspaceBundleError("bundle manifest digest differs")
    if manifest_sha256 != fixture["manifest_sha256"]:
        raise NvidiaUserspaceBundleError("fixture bundle manifest changed")
    expected_sentinel = f"{fixture['content_digest']}\n".encode("ascii")
    if sentinel != expected_sentinel:
        raise NvidiaUserspaceBundleError("fixture bundle content changed")
    return BundleValidation(
        manifest=fixture["manifest"],
        manifest_sha256=manifest_sha256,
        content_digest=fixture["content_digest"],
        kernel_module_version=fixture["driver_version"],
        stat_snapshot=(),
        runtime=fixture["runtime"],
    )


def _register_fake_bundle(root: Path, capture: dict[str, Any]) -> None:
    rootfs = root / "rootfs"
    library = rootfs / "usr/lib/x86_64-linux-gnu"
    binary = rootfs / "usr/bin"
    packages = root / "packages"
    library.mkdir(parents=True)
    binary.mkdir(parents=True)
    packages.mkdir()
    manifest = capture["manifest"]
    manifest_raw = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (root / "NVIDIA_USERSPACE_BUNDLE_MANIFEST.json").write_bytes(manifest_raw)
    (root / "fixture-content.bin").write_text(
        f"{capture['content_digest']}\n",
        encoding="ascii",
    )
    driver_version = capture["kernel_module_version"]
    runtime = RuntimeMapping(
        rootfs=rootfs,
        library_directory=library,
        nvidia_smi=binary / "nvidia-smi",
        libcuda=library / f"libcuda.so.{driver_version}",
        libnvidia_ml=library / f"libnvidia-ml.so.{driver_version}",
    )
    _FAKE_BUNDLES[str(root)] = {
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "content_digest": capture["content_digest"],
        "driver_version": driver_version,
        "runtime": runtime,
    }
    calibration_campaign_module.validate_bundle = _fixture_bundle_validator
    calibration_evidence_module.validate_bundle = _fixture_bundle_validator


def artifact_inventory(run_dir: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for artifact in sorted(run_dir.rglob("*")):
        relative = artifact.relative_to(run_dir).as_posix()
        if artifact.is_file():
            inventory.append(
                {"path": relative, "type": "file", "size": artifact.stat().st_size}
            )
        elif artifact.is_dir():
            inventory.append({"path": relative, "type": "directory"})
    return inventory


def make_run(
    campaign: Path,
    index: int,
    *,
    implementation_manifest_sha256: str,
    fingerprint_seed: str = "shared",
    implementation_capture: dict[str, object] | None = None,
    model_root: Path | None = None,
    cpu_bytes: int = 1024,
    nvidia_bundle_root: Path | None = None,
    nvidia_driver_version: str = NVIDIA_KERNEL_VERSION,
    protocol_payload: bytes | None = None,
) -> Path:
    run_dir = campaign / f"run-{index:03d}"
    run_id = f"m2-run-{index:03d}"
    capture = implementation_capture or _implementation_capture()
    build_raw_run(
        run_dir,
        run_id=run_id,
        implementation=capture,
        expected_implementation_manifest_sha256=implementation_manifest_sha256,
        fingerprint_seed=fingerprint_seed,
        model_root=model_root,
        cpu_bytes=cpu_bytes,
        protocol_payload=protocol_payload or PROTOCOL_PATH.read_bytes(),
        nvidia_bundle_root=nvidia_bundle_root,
        nvidia_driver_version=nvidia_driver_version,
    )
    return run_dir


def _compact_row(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _log_entry(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _timestamp(base: datetime, offset: int) -> str:
    return (base + timedelta(seconds=offset)).isoformat()


def _make_python_symlink(root: Path) -> Path:
    target = root / "python-target"
    target.write_bytes(b"fixture interpreter bytes\n")
    target.chmod(0o755)
    venv_bin = root / "fixture-venv" / "bin"
    venv_bin.mkdir(parents=True)
    entry = venv_bin / "python"
    entry.symlink_to(target)
    return entry


def build_calibration_campaign(
    root: Path,
    *,
    publish: bool,
) -> CalibrationFixture:
    model = root / "model"
    vllm = root / "vllm"
    model.mkdir()
    vllm.mkdir()
    python_entry = _make_python_symlink(root)
    nvidia_bundle_root = (root / "nvidia-bundle").absolute()
    for _ in range(8):
        protocol_payload = PROTOCOL_PATH.read_bytes()
        implementation_capture = _implementation_capture()
        protocol_after = PROTOCOL_PATH.read_bytes()
        implementation_after = _implementation_capture()
        protocol_entry = next(
            entry
            for entry in implementation_capture["files"]
            if entry["path"] == "research/protocols/M2_VLLM_REPLAY_PROTOCOL.md"
        )
        if (
            protocol_payload == protocol_after
            and implementation_capture == implementation_after
            and protocol_entry["sha256"] == hashlib.sha256(protocol_payload).hexdigest()
        ):
            break
    else:
        raise RuntimeError("protocol source did not stabilize during fixture capture")
    implementation = implementation_capture["manifest_sha256"]
    probe = build_raw_run(
        root / "fingerprint-probe",
        run_id="m2-fingerprint-probe",
        implementation=implementation_capture,
        expected_implementation_manifest_sha256=implementation,
        fingerprint_seed="shared",
        model_root=model,
        cpu_bytes=1024,
        protocol_payload=protocol_payload,
        nvidia_bundle_root=nvidia_bundle_root,
        nvidia_driver_version=NVIDIA_KERNEL_VERSION,
    )
    provenance = json.loads(
        (probe.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    nvidia_capture = provenance["nvidia_driver_userspace"]
    _register_fake_bundle(nvidia_bundle_root, nvidia_capture)
    fingerprint = probe.reproducibility_fingerprint
    campaign = root / "campaign"
    config = CampaignConfig(
        campaign_root=campaign,
        expected_implementation_manifest_sha256=implementation,
        expected_reproducibility_fingerprint=fingerprint,
        nvidia_userspace_bundle_root=nvidia_bundle_root,
        expected_nvidia_driver_version=probe.nvidia_driver_version,
        expected_nvidia_userspace_bundle_manifest_sha256=(probe.nvidia_manifest_sha256),
        expected_nvidia_userspace_bundle_content_digest=probe.nvidia_content_digest,
        python_executable=python_entry,
        model=model,
        vllm_root=vllm,
        cpu_bytes=1024,
        runner_timeout_s=1.0,
        process_timeout_s=2.0,
        aggregation_timeout_s=2.0,
        terminate_grace_s=1.0,
        kill_wait_s=1.0,
    )
    synthetic_binding = {
        "preparation_git_head": "b" * 40,
        "execution_git_head": provenance["dagkv_git"]["head"],
        "launch_marker_repository_path": (
            "evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json"
        ),
        "launch_marker_sha256": "c" * 64,
    }
    real_clean_git_head = calibration_campaign_module._clean_git_head
    calibration_campaign_module._clean_git_head = lambda **_: "b" * 40
    try:
        preregistration_sha256 = prepare_campaign(config)
    finally:
        calibration_campaign_module._clean_git_head = real_clean_git_head
    calibration_evidence_module._marker_and_execution_binding = (
        lambda preregistration, preregistration_sha256, candidate: dict(
            synthetic_binding
        )
    )
    preregistration_path = campaign / PREREGISTRATION_NAME
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    campaign_id = preregistration["campaign_id"]
    base = datetime.fromisoformat(preregistration["created_at_utc"]).astimezone(UTC)
    rows: list[dict[str, object]] = []
    for sequence in range(1, 60):
        run_name = f"run-{sequence:03d}"
        run_dir = make_run(
            campaign,
            sequence,
            implementation_manifest_sha256=implementation,
            implementation_capture=implementation_capture,
            model_root=model,
            nvidia_bundle_root=nvidia_bundle_root,
            nvidia_driver_version=probe.nvidia_driver_version,
            protocol_payload=protocol_payload,
        )
        stdout = campaign / f"{run_name}.stdout.log"
        stderr = campaign / f"{run_name}.stderr.log"
        stdout.write_text(f"completed {run_name}\n", encoding="utf-8")
        stderr.write_bytes(b"")
        command = list(preregistration["runner_command_template"])
        command[3] = str(run_dir)
        attempt_id = f"{campaign_id}:{run_name}"
        submitted = {
            "schema_version": ATTEMPT_SCHEMA,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "kind": "calibration_run",
            "event": "submitted",
            "timestamp_utc": _timestamp(base, sequence * 4),
            "sequence": sequence,
            "run_name": run_name,
            "command": command,
            "output_dir": run_name,
            "stdout": stdout.name,
            "stderr": stderr.name,
            "preregistration_sha256": preregistration_sha256,
            "execution_binding": dict(synthetic_binding),
        }
        validated = _validate_run(run_dir)
        terminal = {
            "schema_version": ATTEMPT_SCHEMA,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "kind": "calibration_run",
            "event": "terminal",
            "timestamp_utc": _timestamp(base, sequence * 4 + 3),
            "status": "passed",
            "pid": 10_000 + sequence,
            "exit_code": 0,
            "duration_s": 1.0,
            "started_at_utc": _timestamp(base, sequence * 4 + 1),
            "ended_at_utc": _timestamp(base, sequence * 4 + 2),
            "timed_out": False,
            "sigterm_sent": False,
            "sigkill_sent": False,
            "stdout": _log_entry(stdout),
            "stderr": _log_entry(stderr),
            "artifact_inventory": artifact_inventory(run_dir),
            "error": None,
            "sequence": sequence,
            "run_name": run_name,
            "validation": {
                "run_id": validated.run_id,
                "result_sha256": validated.result_sha256,
                "provenance_sha256": validated.provenance_sha256,
                "sha256sums_sha256": validated.sha256sums_sha256,
                "implementation_manifest_sha256": (
                    validated.implementation_manifest_sha256
                ),
                "reproducibility_fingerprint": (validated.reproducibility_fingerprint),
                "observed_max_abs_error": validated.observed_max_abs_error,
                "dagkv_git_head": validated.dagkv_git_head,
                "dagkv_snapshot_sha256": validated.dagkv_snapshot_sha256,
            },
        }
        rows.extend((submitted, terminal))

    prefix = b"".join(_compact_row(row) for row in rows)
    aggregate_stdout = campaign / "aggregate.stdout.log"
    aggregate_stderr = campaign / "aggregate.stderr.log"
    aggregate_stdout.write_bytes(b"")
    aggregate_stderr.write_bytes(b"")
    manifest_path = campaign / COHORT_NAME
    aggregate_submitted = {
        "schema_version": ATTEMPT_SCHEMA,
        "campaign_id": campaign_id,
        "attempt_id": f"{campaign_id}:aggregate",
        "kind": "aggregate",
        "event": "submitted",
        "timestamp_utc": _timestamp(base, 60 * 4),
        "command": [
            preregistration["python_executable"]["path"],
            preregistration["frozen_files"]["aggregator"]["path"],
            "--campaign-dir",
            str(campaign),
            "--output",
            str(manifest_path),
        ],
        "output": COHORT_NAME,
        "stdout": aggregate_stdout.name,
        "stderr": aggregate_stderr.name,
        "preregistration_sha256": preregistration_sha256,
        "calibration_prefix": {
            "prefix_bytes": len(prefix),
            "prefix_record_count": 118,
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        },
        "execution_binding": dict(synthetic_binding),
    }
    attempts_path = campaign / ATTEMPTS_NAME
    attempts_path.write_bytes(prefix + _compact_row(aggregate_submitted))

    if publish:
        manifest = aggregate_campaign(campaign)
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        aggregate_terminal = {
            "schema_version": ATTEMPT_SCHEMA,
            "campaign_id": campaign_id,
            "attempt_id": f"{campaign_id}:aggregate",
            "kind": "aggregate",
            "event": "terminal",
            "timestamp_utc": _timestamp(base, 60 * 4 + 3),
            "status": "passed",
            "pid": 20_000,
            "exit_code": 0,
            "duration_s": 1.0,
            "started_at_utc": _timestamp(base, 60 * 4 + 1),
            "ended_at_utc": _timestamp(base, 60 * 4 + 2),
            "timed_out": False,
            "sigterm_sent": False,
            "sigkill_sent": False,
            "stdout": _log_entry(aggregate_stdout),
            "stderr": _log_entry(aggregate_stderr),
            "artifact_inventory": [
                {
                    "path": COHORT_NAME,
                    "type": "file",
                    "size": manifest_path.stat().st_size,
                }
            ],
            "error": None,
            "validation": {
                "cohort_sha256": manifest_sha256,
                "run_count": 59,
                "reproducibility_fingerprint": manifest["reproducibility_fingerprint"],
            },
        }
        with attempts_path.open("ab") as handle:
            handle.write(_compact_row(aggregate_terminal))
            handle.flush()
            os.fsync(handle.fileno())
    return CalibrationFixture(
        campaign_root=campaign,
        manifest_path=manifest_path,
        preregistration_sha256=preregistration_sha256,
        implementation_manifest_sha256=implementation,
        reproducibility_fingerprint=fingerprint,
    )
