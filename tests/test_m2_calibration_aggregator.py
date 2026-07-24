from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from tools.aggregate_m2_calibration import (
    CALIBRATION_COHORT_SCHEMA,
    EXACT_PAIRS,
    PROTOCOL_SCHEMA,
    TOLERANT_PAIRS,
    CalibrationAggregationError,
    aggregate_campaign,
)
from tools.run_m2_vllm_abba import (
    CALIBRATION_COHORT_SCHEMA as RUNNER_COHORT_SCHEMA,
)
from tools.run_m2_vllm_abba import MIN_CALIBRATION_RUNS as RUNNER_CALIBRATION_RUN_COUNT
from tools.run_m2_vllm_abba import PROTOCOL_SCHEMA as RUNNER_PROTOCOL_SCHEMA
from tools.run_m2_vllm_abba import (
    TOLERANCE_DERIVATION,
    FrozenTolerance,
    load_calibration_cohort,
)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_checksums(run_dir: Path) -> None:
    paths = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(run_dir).as_posix()}"
        for path in paths
    ]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def _make_run(
    campaign: Path,
    index: int,
    *,
    fingerprint_seed: str = "shared",
    gate_status: str = "CALIBRATED_NOT_ACCEPTED",
    with_checksums: bool = True,
) -> Path:
    run_dir = campaign / f"run-{index:03d}"
    run_dir.mkdir()
    run_id = f"m2-run-{index:03d}"
    for name, payload in {
        "diagnostic_transfers.jsonl": b'{"event":"terminal"}\n',
        "execution_ids.json": b"{}\n",
        "native_lifecycle.jsonl": b'{"event":"lookup"}\n',
        "protocol.md": b"# frozen protocol\n",
    }.items():
        (run_dir / name).write_bytes(payload)
    logits_hashes: dict[str, str] = {}
    for phase in ("A1", "G", "B1", "B2", "A2"):
        path = run_dir / f"logits_{phase}.npy"
        path.write_bytes(f"fixture-logits-{phase}-{index}".encode())
        logits_hashes[phase] = hashlib.sha256(path.read_bytes()).hexdigest()
    measurements = {
        phase: {
            "token_id": 42,
            "top1_margin": 0.5 + index / 100_000,
            "num_cached_tokens": 0 if phase in {"A1", "A2"} else 16,
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
            "allclose": False,
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
    components = {"frozen_environment": fingerprint_seed}
    fingerprint = _digest(components)
    result = {
        "schema_version": PROTOCOL_SCHEMA,
        "run_id": run_id,
        "mode": "calibration",
        "gate_status": gate_status,
        "m2_accepted": False,
        "m2_item8_accepted": False,
        "formal_run_passed": False,
        "within_requested_tolerance": False,
        "tolerance": {"atol": 0.0, "rtol": 0.0},
        "minimum_top1_margin": min(
            measurement["top1_margin"] for measurement in measurements.values()
        ),
        "reproducibility_fingerprint": fingerprint,
        "measurements": measurements,
        "comparisons": comparisons,
    }
    implementation_digest = hashlib.sha256(b"implementation").hexdigest()
    dagkv_digest = hashlib.sha256(b"dagkv").hexdigest()
    vllm_digest = hashlib.sha256(b"vllm").hexdigest()
    provenance = {
        "schema_version": PROTOCOL_SCHEMA,
        "run_id": run_id,
        "mode": "calibration",
        "full_provenance": True,
        "prompt_token_ids": list(range(1000, 1017)),
        "block_size": 16,
        "tolerance": {"atol": 0.0, "rtol": 0.0},
        "reproducibility_components": components,
        "reproducibility_fingerprint": fingerprint,
        "implementation": {"manifest_sha256": implementation_digest},
        "dagkv_git": {"snapshot_sha256": dagkv_digest},
        "vllm_git": {"snapshot_sha256": vllm_digest},
        "postflight": {
            "implementation_manifest_sha256": implementation_digest,
            "dagkv_git_snapshot_sha256": dagkv_digest,
            "vllm_git_snapshot_sha256": vllm_digest,
            "model_file_stats_unchanged": True,
            "runtime_binary_stats_unchanged": True,
        },
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "provenance.json", provenance)
    if with_checksums:
        _write_checksums(run_dir)
    return run_dir


def _make_campaign(root: Path, count: int = 59) -> Path:
    campaign = root / "campaign"
    campaign.mkdir()
    for index in range(count):
        _make_run(campaign, index)
    return campaign


def test_aggregates_59_valid_runs_atomically(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path)
    output = tmp_path / "cohort.json"

    manifest = aggregate_campaign(campaign, output_path=output)

    assert manifest["schema_version"] == CALIBRATION_COHORT_SCHEMA
    assert manifest["protocol_schema"] == PROTOCOL_SCHEMA
    assert CALIBRATION_COHORT_SCHEMA == RUNNER_COHORT_SCHEMA
    assert PROTOCOL_SCHEMA == RUNNER_PROTOCOL_SCHEMA
    assert manifest["run_count"] == 59
    assert RUNNER_CALIBRATION_RUN_COUNT == 59
    assert manifest["observed_max_abs_error"] == 0.1
    assert manifest["formal_atol"] == 0.125
    assert manifest["formal_rtol"] == 0.0
    assert manifest["all_passed"] is True
    assert manifest["failures"] == []
    assert len({run["run_id"] for run in manifest["runs"]}) == 59
    assert all(
        set(run)
        == {
            "run_id",
            "result_sha256",
            "provenance_sha256",
            "sha256sums_sha256",
        }
        for run in manifest["runs"]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    manifest_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    frozen_tolerance = FrozenTolerance(
        atol=0.125,
        rtol=0.0,
        frozen_at_utc="2026-07-25T00:00:00+00:00",
        calibration_manifest_sha256=manifest_sha256,
        reproducibility_fingerprint=manifest["reproducibility_fingerprint"],
        calibration_run_count=59,
        derivation=TOLERANCE_DERIVATION,
        file_sha256="f" * 64,
    )
    assert (
        load_calibration_cohort(
            output,
            frozen_tolerance=frozen_tolerance,
            run_started_ns=time.time_ns() + 1_000_000_000,
        )
        == manifest
    )
    assert not list(tmp_path.glob(".cohort.json.*.tmp"))
    assert not list(tmp_path.rglob("*ACCEPTANCE*"))


def test_rejects_fewer_than_59_runs_without_writing_manifest(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path, count=58)
    output = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="exactly 59"):
        aggregate_campaign(campaign, output_path=output)

    assert not output.exists()


def test_rejects_more_than_59_runs_without_writing_manifest(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path, count=60)
    output = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="exactly 59"):
        aggregate_campaign(campaign, output_path=output)

    assert not output.exists()


def test_partial_attempt_directory_invalidates_campaign(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path)
    (campaign / "run-partial").mkdir()
    output = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="lacks result.json"):
        aggregate_campaign(campaign, output_path=output)

    assert not output.exists()


def test_any_failed_result_invalidates_the_whole_campaign(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path)
    _make_run(
        campaign,
        59,
        gate_status="FAILED",
        with_checksums=False,
    )
    output = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="did not complete"):
        aggregate_campaign(campaign, output_path=output)

    assert not output.exists()


def test_checksum_tampering_invalidates_the_campaign(tmp_path: Path) -> None:
    campaign = _make_campaign(tmp_path)
    provenance = campaign / "run-017" / "provenance.json"
    provenance.write_text(
        provenance.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    output = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="checksum mismatch"):
        aggregate_campaign(campaign, output_path=output)

    assert not output.exists()


def test_reproducibility_fingerprint_drift_invalidates_campaign(
    tmp_path: Path,
) -> None:
    campaign = _make_campaign(tmp_path, count=58)
    _make_run(campaign, 58, fingerprint_seed="drifted")
    output = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="fingerprints differ"):
        aggregate_campaign(campaign, output_path=output)

    assert not output.exists()
