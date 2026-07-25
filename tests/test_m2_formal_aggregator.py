from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.aggregate_m2_formal as formal_aggregator
from tools.aggregate_m2_formal import (
    ACCEPTANCE_MANIFEST,
    ACCEPTANCE_STATEMENT,
    EXACT_PAIRS,
    FORMAL_RUN_COUNT,
    FORMAL_RUN_MANIFEST,
    FORMAL_RUN_SCHEMA,
    FORMAL_RUN_STATEMENT,
    ITEM8_ACCEPTANCE_SCHEMA,
    PROTOCOL_SCHEMA,
    REQUIRED_INTERNAL_ARTIFACTS,
    TOLERANCE_DERIVATION,
    TOLERANT_PAIRS,
    FormalAggregationError,
    _parser,
    aggregate_campaign,
)
from tools.freeze_m2_tolerance import TOLERANCE_SCHEMA
from tools.m2_calibration_evidence import CalibrationEvidenceError
from tools.m2_raw_replay import M2RawReplayError, RawReplayValidation
from tools.run_m2_vllm_abba import (
    ITEM8_FORMAL_RUN_SCHEMA as RUNNER_FORMAL_RUN_SCHEMA,
)
from tools.run_m2_vllm_abba import PROTOCOL_SCHEMA as RUNNER_PROTOCOL_SCHEMA
from tools.run_m2_vllm_abba import _git_capture as _runner_git_capture


@dataclass(frozen=True, slots=True)
class ParentEvidenceFixture:
    calibration_manifest: Path
    frozen_tolerance: Path
    calibration_sha256: str
    tolerance_sha256: str
    reproducibility_fingerprint: str


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _digest_bytes(encoded)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_checksums(run_dir: Path) -> None:
    paths = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and path.name != FORMAL_RUN_MANIFEST
    )
    lines = [
        f"{_digest_bytes(path.read_bytes())}  {path.relative_to(run_dir).as_posix()}"
        for path in paths
    ]
    (run_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _fixture_git_capture(
    run_dir: Path, label: str, *, dirty: bool
) -> dict[str, object]:
    state_dir = run_dir / "source_state"
    state_dir.mkdir(exist_ok=True)
    tracked = state_dir / f"{label}.tracked.patch"
    archive = state_dir / f"{label}.untracked.tar"
    tracked_payload = f"{label}-tracked-state".encode() if dirty else b""
    tracked.write_bytes(tracked_payload)
    archive.write_bytes(f"{label}-untracked-state".encode())
    head = "a" * 40
    untracked_files: list[dict[str, object]] = []
    snapshot = {
        "head": head,
        "tracked_diff_sha256": _digest_bytes(tracked_payload),
        "tracked_diff_bytes": len(tracked_payload),
        "untracked": untracked_files,
    }
    return {
        "root": f"/fixture/{label}",
        "head": head,
        "dirty": dirty,
        "status_short": [" M fixture.py"] if dirty else [],
        "tracked_patch": tracked.relative_to(run_dir).as_posix(),
        "tracked_patch_sha256": _digest_bytes(tracked.read_bytes()),
        "untracked_archive": archive.relative_to(run_dir).as_posix(),
        "untracked_archive_sha256": _digest_bytes(archive.read_bytes()),
        "untracked_files": untracked_files,
        "snapshot_sha256": _canonical_digest(snapshot),
    }


def _content_captures() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    model_file = {
        "path": "model.safetensors",
        "size": 13,
        "kind": "weight",
        "sha256": _digest_bytes(b"model-weight"),
        "mtime_ns": 1,
        "inode": 2,
    }
    model_content = [
        {key: model_file[key] for key in ("path", "size", "kind", "sha256")}
    ]
    model = {
        "root": "/fixture/model",
        "full_hashes": True,
        "files": [model_file],
        "manifest_sha256": _canonical_digest(model_content),
    }

    extension = {
        "path": "vllm/_C.abi3.so",
        "size": 17,
        "sha256": _digest_bytes(b"vllm-extension"),
        "mtime_ns": 3,
        "inode": 4,
    }
    python_entry = {
        "path": "/fixture/python",
        "size": 19,
        "sha256": _digest_bytes(b"python-binary"),
        "mtime_ns": 5,
        "inode": 6,
    }
    runtime_content = {
        "vllm_extensions": [
            {key: extension[key] for key in ("path", "size", "sha256")}
        ],
        "python_executable": {
            key: python_entry[key] for key in ("path", "size", "sha256")
        },
    }
    runtime = {
        "root": "/fixture/vllm",
        "full_hashes": True,
        "vllm_extensions": [extension],
        "python_executable": python_entry,
        "manifest_sha256": _canonical_digest(runtime_content),
    }

    packages = [{"name": "fixture", "version": "1.0"}]
    dependencies = {
        "packages": packages,
        "manifest_sha256": _canonical_digest(packages),
    }
    return model, runtime, dependencies


def _make_run(
    campaign: Path,
    index: int,
    *,
    fingerprint_seed: str = "shared",
    tolerance_sha256: str | None = None,
    calibration_sha256: str | None = None,
    protocol_payload: bytes = b"# frozen M2 v3 protocol\n",
    gate_status: str = "M2_ITEM8_FORMAL_HOLDOUT_PASSED",
    formal_run_passed: bool = True,
    item8_accepted: bool = False,
    run_id: str | None = None,
    frozen_at_utc: str = "2026-07-24T00:00:00+00:00",
    nvidia_userspace_bundle_root: str = "/fixture/nvidia-bundle",
) -> Path:
    run_dir = campaign / f"attempt-{index:03d}"
    run_dir.mkdir()
    run_id = run_id or f"m2-formal-{index:03d}"
    tolerance_sha256 = tolerance_sha256 or _digest_bytes(b"frozen-tolerance")
    calibration_sha256 = calibration_sha256 or _digest_bytes(b"calibration-manifest")

    for name, payload in {
        "diagnostic_transfers.jsonl": b'{"event":"terminal"}\n',
        "execution_ids.json": b"{}\n",
        "native_lifecycle.jsonl": b'{"event":"lookup"}\n',
        "protocol.md": protocol_payload,
    }.items():
        (run_dir / name).write_bytes(payload)
    logits_hashes: dict[str, str] = {}
    for phase in ("A1", "G", "B1", "B2", "A2"):
        path = run_dir / f"logits_{phase}.npy"
        path.write_bytes(f"fixture-logits-{phase}-{index}".encode())
        logits_hashes[phase] = _digest_bytes(path.read_bytes())

    dagkv_git = _fixture_git_capture(run_dir, "dagkv", dirty=False)
    vllm_git = _fixture_git_capture(run_dir, "vllm", dirty=True)
    model, runtime, dependencies = _content_captures()
    implementation_files = [
        {
            "path": "tools/run_m2_vllm_abba.py",
            "size": 23,
            "sha256": _digest_bytes(b"implementation-source"),
        }
    ]
    implementation_sha = _canonical_digest(implementation_files)
    implementation = {
        "files": implementation_files,
        "manifest_sha256": implementation_sha,
    }
    nvidia_content_digest = _digest_bytes(b"nvidia-userspace-bundle")
    nvidia_manifest_sha = _digest_bytes(b"nvidia-userspace-manifest")
    nvidia_userspace = {
        "root": nvidia_userspace_bundle_root,
        "content_digest": nvidia_content_digest,
        "manifest_sha256": nvidia_manifest_sha,
        "kernel_module_version": "580.173.02",
    }
    system = {"fixture_environment": fingerprint_seed}
    engine_config = {
        "model": "/fixture/model",
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
    connector_config = {
        "cpu_bytes_to_use": 1 << 30,
        "spec_name": "DAGKVDiagnosticCPUOffloadingSpec",
        "spec_module_path": "dagkv_vllm_m2.spec",
        "dagkv_diagnostic_trace_file": str(run_dir / "diagnostic_transfers.jsonl"),
        "dagkv_diagnostic_run_id": run_id,
        "dagkv_diagnostic_phase": "ABBA",
        "fanout_layerwise_load": False,
        "lifecycle_accounting_enabled": True,
    }
    static_connector = {
        key: value
        for key, value in connector_config.items()
        if key
        not in {
            "dagkv_diagnostic_trace_file",
            "dagkv_diagnostic_run_id",
        }
    }
    components = {
        "implementation_manifest_sha256": implementation_sha,
        "vllm_snapshot_sha256": vllm_git["snapshot_sha256"],
        "model_manifest_sha256": model["manifest_sha256"],
        "runtime_binary_manifest_sha256": runtime["manifest_sha256"],
        "dependency_manifest_sha256": dependencies["manifest_sha256"],
        "nvidia_driver_userspace_content_digest": nvidia_content_digest,
        "system": system,
        "prompt_token_ids": list(range(1000, 1017)),
        "block_size": 16,
        "cpu_bytes": 1 << 30,
        "engine_config": engine_config,
        "connector_config": static_connector,
    }
    fingerprint = _canonical_digest(components)
    tolerance = {"atol": 0.125, "rtol": 0.0}

    measurements = {
        phase: {
            "request_id": f"{run_id}:{phase}:request",
            "trace_id": f"{run_id}:{phase}:trace",
            "token_id": 42,
            "num_cached_tokens": 0 if phase in {"A1", "A2"} else 16,
            "elapsed_ms": 1.0 + index / 1000,
            "top1_margin": 0.5,
            "logits_file": f"logits_{phase}.npy",
            "logits_sha256": logits_hashes[phase],
        }
        for phase in ("A1", "G", "B1", "B2", "A2")
    }
    comparisons = [
        {
            "left": left,
            "right": right,
            "token_equal": True,
            "allclose": True,
            "max_abs_error": 0.1,
            "max_rel_error": 0.01,
        }
        for left, right in TOLERANT_PAIRS
    ]
    comparisons.extend(
        {
            "left": left,
            "right": right,
            "token_equal": True,
            "allclose": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        }
        for left, right in EXACT_PAIRS
    )
    transfer_bytes = {
        "B1_D2H": 4096,
        "B1_H2D": 4096,
        "B2_D2H": 4096,
        "B2_H2D": 4096,
    }
    transfer_digest = _digest_bytes(b"canonical-kv-payload")
    result = {
        "schema_version": PROTOCOL_SCHEMA,
        "run_id": run_id,
        "mode": "formal",
        "gate_status": gate_status,
        "m2_accepted": False,
        "m2_item8_accepted": item8_accepted,
        "formal_run_passed": formal_run_passed,
        "within_requested_tolerance": True,
        "minimum_top1_margin": 0.5,
        "reproducibility_fingerprint": fingerprint,
        "completed_at_utc": "2026-07-25T01:00:00+00:00",
        "tolerance": tolerance,
        "measurements": measurements,
        "comparisons": comparisons,
        "prefetch": {"B1": {"completed": True}, "B2": {"completed": True}},
        "native_bytes": transfer_bytes,
        "diagnostic_bytes": transfer_bytes.copy(),
        "transfer_digests": {"B1": transfer_digest, "B2": transfer_digest},
        "artifacts": {
            "native_trace": "native_lifecycle.jsonl",
            "diagnostic_trace": "diagnostic_transfers.jsonl",
            "protocol": "protocol.md",
            "provenance": "provenance.json",
        },
    }
    provenance = {
        "schema_version": PROTOCOL_SCHEMA,
        "run_id": run_id,
        "mode": "formal",
        "started_at_utc": "2026-07-25T00:00:00+00:00",
        "argv": ["run_m2_vllm_abba.py", "--mode", "formal"],
        "python": "3.12.fixture",
        "executable": "/fixture/python",
        "prompt_token_ids": list(range(1000, 1017)),
        "block_size": 16,
        "cpu_bytes": 1 << 30,
        "tolerance": tolerance,
        "frozen_tolerance": {
            "atol": 0.125,
            "rtol": 0.0,
            "frozen_at_utc": frozen_at_utc,
            "calibration_manifest_sha256": calibration_sha256,
            "reproducibility_fingerprint": fingerprint,
            "calibration_run_count": 59,
            "derivation": TOLERANCE_DERIVATION,
            "file_sha256": tolerance_sha256,
        },
        "calibration_cohort": {
            "path": "/fixture/M2_CALIBRATION_MANIFEST.json",
            "sha256": calibration_sha256,
            "run_count": 59,
        },
        "full_provenance": True,
        "preflight": {},
        "implementation": implementation,
        "dagkv_git": dagkv_git,
        "vllm_git": vllm_git,
        "model": model,
        "runtime_binaries": runtime,
        "dependencies": dependencies,
        "system": system,
        "nvidia_driver_userspace": nvidia_userspace,
        "reproducibility_components": components,
        "reproducibility_fingerprint": fingerprint,
        "engine_config": engine_config,
        "connector_config": connector_config,
        "postflight": {
            "completed_at_utc": "2026-07-25T01:00:00+00:00",
            "dagkv_git_snapshot_sha256": dagkv_git["snapshot_sha256"],
            "vllm_git_snapshot_sha256": vllm_git["snapshot_sha256"],
            "implementation_manifest_sha256": implementation_sha,
            "model_file_stats_unchanged": True,
            "runtime_binary_stats_unchanged": True,
            "nvidia_driver_userspace_content_digest": nvidia_content_digest,
            "nvidia_driver_userspace_manifest_sha256": nvidia_manifest_sha,
            "nvidia_driver_userspace_unchanged": True,
            "libcuda_mapping_unchanged": True,
        },
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "provenance.json", provenance)
    _write_checksums(run_dir)

    formal_manifest = {
        "schema_version": FORMAL_RUN_SCHEMA,
        "run_id": run_id,
        "completed_at_utc": "2026-07-25T01:00:01+00:00",
        "result_sha256": _digest_bytes((run_dir / "result.json").read_bytes()),
        "provenance_sha256": _digest_bytes((run_dir / "provenance.json").read_bytes()),
        "sha256sums_sha256": _digest_bytes((run_dir / "SHA256SUMS").read_bytes()),
        "frozen_tolerance_sha256": tolerance_sha256,
        "calibration_manifest_sha256": calibration_sha256,
        "reproducibility_fingerprint": fingerprint,
        "statement": FORMAL_RUN_STATEMENT,
    }
    _write_json(run_dir / FORMAL_RUN_MANIFEST, formal_manifest)
    return run_dir


def _make_campaign(
    root: Path,
    parent: ParentEvidenceFixture,
    count: int = FORMAL_RUN_COUNT,
    *,
    frozen_at_utc: str = "2026-07-24T00:00:00+00:00",
) -> Path:
    campaign = root / "formal-campaign"
    campaign.mkdir()
    for index in range(count):
        _make_run(
            campaign,
            index,
            tolerance_sha256=parent.tolerance_sha256,
            calibration_sha256=parent.calibration_sha256,
            frozen_at_utc=frozen_at_utc,
        )
    return campaign


@pytest.fixture(scope="module")
def parent_evidence(tmp_path_factory: pytest.TempPathFactory) -> ParentEvidenceFixture:
    root = tmp_path_factory.mktemp("formal-parent-evidence")
    probe = root / "probe"
    probe.mkdir()
    probe_run = _make_run(probe, 0)
    probe_provenance = json.loads(
        (probe_run / "provenance.json").read_text(encoding="utf-8")
    )
    calibration_root = root / "calibration-campaign"
    calibration_root.mkdir()
    calibration_manifest = calibration_root / "M2_CALIBRATION_MANIFEST.json"
    calibration_payload = {
        "schema_version": "dagkv.m2.calibration_cohort.v3",
        "run_count": 59,
        "formal_atol": 0.125,
        "formal_rtol": 0.0,
        "reproducibility_fingerprint": probe_provenance["reproducibility_fingerprint"],
    }
    _write_json(calibration_manifest, calibration_payload)
    calibration_sha = _digest_bytes(calibration_manifest.read_bytes())
    tolerance_path = root / "evidence" / "M2_FROZEN_TOLERANCE.json"
    tolerance_path.parent.mkdir()
    tolerance = {
        "schema_version": TOLERANCE_SCHEMA,
        "frozen": True,
        "frozen_at_utc": "2026-07-24T00:00:00+00:00",
        "atol": 0.125,
        "rtol": 0.0,
        "calibration_manifest_sha256": calibration_sha,
        "reproducibility_fingerprint": probe_provenance["reproducibility_fingerprint"],
        "calibration_run_count": 59,
        "derivation": TOLERANCE_DERIVATION,
    }
    _write_json(tolerance_path, tolerance)
    return ParentEvidenceFixture(
        calibration_manifest=calibration_manifest,
        frozen_tolerance=tolerance_path,
        calibration_sha256=calibration_sha,
        tolerance_sha256=_digest_bytes(tolerance_path.read_bytes()),
        reproducibility_fingerprint=tolerance["reproducibility_fingerprint"],
    )


@pytest.fixture(autouse=True)
def stub_published_calibration_validation(
    monkeypatch: pytest.MonkeyPatch,
    parent_evidence: ParentEvidenceFixture,
) -> None:
    def validate(
        path: Path,
        *,
        run_validator: object,
        expected_manifest_sha256: str | None = None,
        expected_implementation_manifest_sha256: str | None = None,
    ) -> tuple[dict[str, object], str, SimpleNamespace]:
        del expected_implementation_manifest_sha256
        assert run_validator is formal_aggregator._validate_calibration_run
        raw = path.read_bytes()
        observed = _digest_bytes(raw)
        if (
            expected_manifest_sha256 is not None
            and observed != expected_manifest_sha256
        ):
            raise CalibrationEvidenceError(
                "calibration manifest SHA-256 differs from the expected digest"
            )
        payload = json.loads(raw)
        evidence = SimpleNamespace(
            reproducibility_fingerprint=payload["reproducibility_fingerprint"],
            nvidia_userspace_bundle_root="/fixture/nvidia-bundle",
            nvidia_userspace_bundle_manifest_sha256=_digest_bytes(
                b"nvidia-userspace-manifest"
            ),
            nvidia_userspace_bundle_content_digest=_digest_bytes(
                b"nvidia-userspace-bundle"
            ),
            nvidia_driver_version="580.173.02",
            runs=tuple(
                SimpleNamespace(run_id=f"m2-calibration-{index:03d}")
                for index in range(1, 60)
            ),
        )
        return payload, observed, evidence

    monkeypatch.setattr(
        formal_aggregator, "validate_published_calibration_bundle", validate
    )


@pytest.fixture(autouse=True)
def stub_raw_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    calls: list[Path] = []

    def validate(run_dir: Path) -> RawReplayValidation:
        calls.append(run_dir)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        provenance = json.loads(
            (run_dir / "provenance.json").read_text(encoding="utf-8")
        )
        return RawReplayValidation(
            run_id=result["run_id"],
            mode=result["mode"],
            observed_max_abs_error=max(
                row["max_abs_error"] for row in result["comparisons"]
            ),
            minimum_top1_margin=result["minimum_top1_margin"],
            reproducibility_fingerprint=result["reproducibility_fingerprint"],
            implementation_manifest_sha256=provenance["implementation"][
                "manifest_sha256"
            ],
        )

    monkeypatch.setattr(formal_aggregator, "validate_raw_run", validate)
    return calls


def _aggregate(
    campaign: Path, parent: ParentEvidenceFixture, *, output_path: Path
) -> dict[str, object]:
    return aggregate_campaign(
        campaign,
        calibration_manifest=parent.calibration_manifest,
        frozen_tolerance=parent.frozen_tolerance,
        output_path=output_path,
    )


def _tolerance_variant(
    root: Path,
    parent: ParentEvidenceFixture,
    **updates: object,
) -> ParentEvidenceFixture:
    payload = json.loads(parent.frozen_tolerance.read_text(encoding="utf-8"))
    payload.update(updates)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "M2_FROZEN_TOLERANCE.json"
    _write_json(path, payload)
    return replace(
        parent,
        frozen_tolerance=path,
        tolerance_sha256=_digest_bytes(path.read_bytes()),
    )


def test_accepts_exactly_twenty_formal_runs_atomically(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
    stub_raw_replay: list[Path],
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    output = tmp_path / ACCEPTANCE_MANIFEST

    manifest = _aggregate(campaign, parent_evidence, output_path=output)

    assert manifest["schema_version"] == ITEM8_ACCEPTANCE_SCHEMA
    assert manifest["protocol_schema"] == PROTOCOL_SCHEMA
    assert PROTOCOL_SCHEMA == RUNNER_PROTOCOL_SCHEMA
    assert FORMAL_RUN_SCHEMA == RUNNER_FORMAL_RUN_SCHEMA
    assert manifest["gate_status"] == "M2_ITEM8_ACCEPTED"
    assert manifest["run_count"] == 20
    assert manifest["passed_run_count"] == 20
    assert manifest["m2_item8_accepted"] is True
    assert manifest["m2_accepted"] is False
    assert manifest["performance_claims_supported"] is False
    assert manifest["frozen_tolerance_sha256"] == parent_evidence.tolerance_sha256
    assert manifest["calibration_manifest_sha256"] == parent_evidence.calibration_sha256
    assert manifest["nvidia_userspace_bundle_root"] == "/fixture/nvidia-bundle"
    assert manifest["nvidia_userspace_bundle_manifest_sha256"] == _digest_bytes(
        b"nvidia-userspace-manifest"
    )
    assert manifest["nvidia_userspace_bundle_content_digest"] == _digest_bytes(
        b"nvidia-userspace-bundle"
    )
    assert manifest["nvidia_driver_version"] == "580.173.02"
    assert manifest["statement"] == ACCEPTANCE_STATEMENT
    assert len(manifest["runs"]) == 20
    assert manifest["runs"] == sorted(manifest["runs"], key=lambda run: run["run_id"])
    assert all(
        set(run)
        == {
            "run_id",
            "formal_run_manifest_sha256",
            "result_sha256",
            "provenance_sha256",
            "sha256sums_sha256",
        }
        for run in manifest["runs"]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert not list(tmp_path.glob(f".{ACCEPTANCE_MANIFEST}.*.tmp"))
    assert not list(tmp_path.rglob("M2_ACCEPTANCE_MANIFEST.json"))
    assert len(stub_raw_replay) == FORMAL_RUN_COUNT
    assert set(stub_raw_replay) == set(campaign.iterdir())


def test_nvidia_reproducibility_component_must_match_provenance(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    run_dir = campaign / "attempt-000"
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["nvidia_driver_userspace"]["content_digest"] = "0" * 64
    provenance["postflight"]["nvidia_driver_userspace_content_digest"] = "0" * 64
    _write_json(provenance_path, provenance)
    _write_checksums(run_dir)
    formal_manifest_path = run_dir / FORMAL_RUN_MANIFEST
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    formal_manifest["provenance_sha256"] = _digest_bytes(provenance_path.read_bytes())
    formal_manifest["sha256sums_sha256"] = _digest_bytes(
        (run_dir / "SHA256SUMS").read_bytes()
    )
    _write_json(formal_manifest_path, formal_manifest)

    with pytest.raises(
        FormalAggregationError,
        match="NVIDIA reproducibility component differs",
    ):
        _aggregate(
            campaign,
            parent_evidence,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_rejects_formal_bundle_identity_that_differs_from_calibration_parent(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
) -> None:
    campaign = tmp_path / "formal-campaign"
    campaign.mkdir()
    for index in range(FORMAL_RUN_COUNT):
        _make_run(
            campaign,
            index,
            tolerance_sha256=parent_evidence.tolerance_sha256,
            calibration_sha256=parent_evidence.calibration_sha256,
            nvidia_userspace_bundle_root="/fixture/other-nvidia-bundle",
        )
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(
        FormalAggregationError,
        match="do not bind the calibration NVIDIA userspace bundle",
    ):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_cli_requires_both_parent_evidence_inputs() -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--campaign-dir", "/fixture/formal"])

    args = parser.parse_args(
        [
            "--campaign-dir",
            "/fixture/formal",
            "--calibration-manifest",
            "/fixture/M2_CALIBRATION_MANIFEST.json",
            "--frozen-tolerance",
            "/fixture/M2_FROZEN_TOLERANCE.json",
        ]
    )
    assert args.calibration_manifest.name == "M2_CALIBRATION_MANIFEST.json"
    assert args.frozen_tolerance.name == "M2_FROZEN_TOLERANCE.json"


def test_raw_replay_rejection_blocks_formal_publication(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    accepted = formal_aggregator.validate_raw_run

    def reject_one(run_dir: Path) -> RawReplayValidation:
        if run_dir.name == "attempt-007":
            raise M2RawReplayError("fixture raw trace mismatch")
        return accepted(run_dir)

    monkeypatch.setattr(formal_aggregator, "validate_raw_run", reject_one)
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="raw artifact replay failed"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("mode", "calibration", "raw replay mode differs"),
        ("run_id", "different-run", "raw replay run_id differs"),
        (
            "reproducibility_fingerprint",
            "0" * 64,
            "raw replay fingerprint differs",
        ),
        (
            "implementation_manifest_sha256",
            "0" * 64,
            "raw replay implementation differs",
        ),
    ],
)
def test_raw_replay_identity_must_match_stable_json_evidence(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    error: str,
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    accepted = formal_aggregator.validate_raw_run

    def drift_one(run_dir: Path) -> RawReplayValidation:
        raw = accepted(run_dir)
        if run_dir.name == "attempt-000":
            return replace(raw, **{field: value})
        return raw

    monkeypatch.setattr(formal_aggregator, "validate_raw_run", drift_one)

    with pytest.raises(FormalAggregationError, match=error):
        _aggregate(
            campaign,
            parent_evidence,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_partial_attempt_invalidates_exact_twenty_directory_set(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence, count=19)
    (campaign / "attempt-partial").mkdir()
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="missing result.json"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_failed_attempt_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence, count=19)
    _make_run(
        campaign,
        19,
        tolerance_sha256=parent_evidence.tolerance_sha256,
        calibration_sha256=parent_evidence.calibration_sha256,
        gate_status="FAILED",
        formal_run_passed=False,
    )
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="gate did not pass"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_extra_direct_directory_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    (campaign / "undeclared-attempt").mkdir()
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="exactly 20 direct attempt"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_fewer_than_twenty_directories_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence, count=19)
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="exactly 20 direct attempt"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_checksum_tampering_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    result = campaign / "attempt-007" / "result.json"
    result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="checksum mismatch"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_formal_manifest_must_bind_actual_result_hash(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    path = campaign / "attempt-011" / FORMAL_RUN_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["result_sha256"] = "0" * 64
    _write_json(path, manifest)
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="does not bind the actual"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_unchecksummed_internal_artifact_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    (campaign / "attempt-003" / "leftover.bin").write_bytes(b"undeclared")
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="coverage mismatch"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_empty_internal_directory_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    (campaign / "attempt-003" / "undeclared-empty").mkdir()
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="directory set differs"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_hard_linked_run_artifact_invalidates_campaign(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    result = campaign / "attempt-003" / "result.json"
    os.link(result, tmp_path / "external-result-link.json")
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="hard-linked"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("drift_kind", "error"),
    [
        ("fingerprint", "reproducibility fingerprints differ"),
        ("tolerance", "frozen tolerance hashes differ"),
        ("calibration", "calibration manifest hashes differ"),
        ("protocol", "protocol hashes differ"),
    ],
)
def test_cross_run_drift_invalidates_campaign(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
    drift_kind: str,
    error: str,
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence, count=19)
    kwargs: dict[str, object] = {
        "tolerance_sha256": parent_evidence.tolerance_sha256,
        "calibration_sha256": parent_evidence.calibration_sha256,
    }
    if drift_kind == "fingerprint":
        kwargs["fingerprint_seed"] = "drifted"
    elif drift_kind == "tolerance":
        kwargs["tolerance_sha256"] = _digest_bytes(b"different-tolerance")
    elif drift_kind == "calibration":
        kwargs["calibration_sha256"] = _digest_bytes(b"different-calibration")
    else:
        kwargs["protocol_payload"] = b"# drifted protocol\n"
    _make_run(campaign, 19, **kwargs)
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match=error):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_single_run_cannot_preclaim_item8_acceptance(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence, count=19)
    _make_run(
        campaign,
        19,
        tolerance_sha256=parent_evidence.tolerance_sha256,
        calibration_sha256=parent_evidence.calibration_sha256,
        item8_accepted=True,
    )

    with pytest.raises(FormalAggregationError, match="single run claims item 8"):
        _aggregate(
            campaign,
            parent_evidence,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_supplied_parent_artifacts_must_match_every_formal_run(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    other_parent = _tolerance_variant(
        tmp_path / "other-parent",
        parent_evidence,
        frozen_at_utc="2026-07-23T00:00:00+00:00",
    )

    with pytest.raises(FormalAggregationError, match="supplied frozen tolerance"):
        _aggregate(
            campaign,
            other_parent,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_frozen_tolerance_must_predate_all_formal_holdouts(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    future_timestamp = "2026-07-25T00:30:00+00:00"
    future_parent = _tolerance_variant(
        tmp_path,
        parent_evidence,
        frozen_at_utc=future_timestamp,
    )
    campaign_root = tmp_path / "formal"
    campaign_root.mkdir()
    campaign = _make_campaign(
        campaign_root,
        future_parent,
        frozen_at_utc=future_timestamp,
    )

    with pytest.raises(FormalAggregationError, match="predate every formal holdout"):
        _aggregate(
            campaign,
            future_parent,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_calibration_and_formal_run_ids_must_be_disjoint(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence, count=19)
    _make_run(
        campaign,
        19,
        tolerance_sha256=parent_evidence.tolerance_sha256,
        calibration_sha256=parent_evidence.calibration_sha256,
        run_id="m2-calibration-001",
    )

    with pytest.raises(FormalAggregationError, match="run IDs must be disjoint"):
        _aggregate(
            campaign,
            parent_evidence,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_calibration_manifest_content_must_match_frozen_digest(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    calibration_manifest = calibration_root / "M2_CALIBRATION_MANIFEST.json"
    calibration_manifest.write_bytes(parent_evidence.calibration_manifest.read_bytes())
    local_parent = replace(
        parent_evidence,
        calibration_manifest=calibration_manifest,
    )
    campaign_root = tmp_path / "formal"
    campaign_root.mkdir()
    campaign = _make_campaign(campaign_root, local_parent)
    calibration_manifest.write_text(
        calibration_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FormalAggregationError, match="invalid published calibration"):
        _aggregate(
            campaign,
            local_parent,
            output_path=tmp_path / ACCEPTANCE_MANIFEST,
        )


def test_publication_rescan_rejects_late_formal_input(
    tmp_path: Path,
    parent_evidence: ParentEvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    original = formal_aggregator._discover_run_dirs
    calls = 0

    def discover_with_late_input(path: Path) -> list[Path]:
        nonlocal calls
        calls += 1
        discovered = original(path)
        if calls == 2:
            (discovered[0] / "late-input.bin").write_bytes(b"late mutation")
        return discovered

    monkeypatch.setattr(
        formal_aggregator, "_discover_run_dirs", discover_with_late_input
    )
    output = tmp_path / ACCEPTANCE_MANIFEST

    with pytest.raises(FormalAggregationError, match="changed before publication"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert not output.exists()


def test_runner_git_capture_paths_match_required_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "DAGKV Test")
    _git(repo, "config", "user.email", "dagkv-test@example.invalid")
    (repo / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "fixture")
    output = tmp_path / "runner-output"

    capture = _runner_git_capture(repo, output_dir=output, label="dagkv")

    assert capture["tracked_patch"] == "source_state/dagkv.tracked.patch"
    assert capture["untracked_archive"] == "source_state/dagkv.untracked.tar"
    assert capture["tracked_patch"] in REQUIRED_INTERNAL_ARTIFACTS
    assert capture["untracked_archive"] in REQUIRED_INTERNAL_ARTIFACTS


def test_refuses_to_overwrite_existing_acceptance_manifest(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    output = tmp_path / ACCEPTANCE_MANIFEST
    output.write_text("existing evidence\n", encoding="utf-8")

    with pytest.raises(FormalAggregationError, match="already exists"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert output.read_text(encoding="utf-8") == "existing evidence\n"


def test_rejects_acceptance_output_symlink(
    tmp_path: Path, parent_evidence: ParentEvidenceFixture
) -> None:
    campaign = _make_campaign(tmp_path, parent_evidence)
    target = tmp_path / "unrelated.json"
    target.write_text("preserve\n", encoding="utf-8")
    output = tmp_path / ACCEPTANCE_MANIFEST
    output.symlink_to(target)

    with pytest.raises(FormalAggregationError, match="cannot be a symlink"):
        _aggregate(campaign, parent_evidence, output_path=output)

    assert output.is_symlink()
    assert target.read_text(encoding="utf-8") == "preserve\n"
