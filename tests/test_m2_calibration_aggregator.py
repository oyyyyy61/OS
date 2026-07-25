from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.m2_calibration_evidence as calibration_evidence_module
from tests.m2_calibration_fixtures import build_calibration_campaign
from tools.aggregate_m2_calibration import (
    CALIBRATION_COHORT_SCHEMA,
    PROTOCOL_SCHEMA,
    CalibrationAggregationError,
    _validate_run,
    aggregate_campaign,
)
from tools.m2_calibration_evidence import (
    MANIFEST_FIELDS,
    MANIFEST_RUN_FIELDS,
    CalibrationEvidenceError,
    _validate_frozen_entry,
    validate_published_calibration_bundle,
    validate_published_calibration_candidate,
)
from tools.run_m2_vllm_abba import (
    CALIBRATION_COHORT_SCHEMA as RUNNER_COHORT_SCHEMA,
)
from tools.run_m2_vllm_abba import MIN_CALIBRATION_RUNS


def _attempt_rows(campaign: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (campaign / "ATTEMPTS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _write_attempt_rows(campaign: Path, rows: list[dict[str, object]]) -> None:
    encoded = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
    )
    (campaign / "ATTEMPTS.jsonl").write_text(encoded, encoding="utf-8")


def _marker_replay_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], str, dict[str, str]]:
    preparation = "b" * 40
    execution = "a" * 40
    preregistration = {
        "campaign_id": "m2-calibration-" + "1" * 32,
        "campaign_root": str(tmp_path / "campaign"),
        "preparation_git_head": preparation,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    marker = {
        "schema_version": "dagkv.m2.calibration_launch_marker.v1",
        "campaign_id": preregistration["campaign_id"],
        "campaign_root": preregistration["campaign_root"],
        "campaign_preregistration_sha256": "d" * 64,
        "preparation_git_head": preparation,
        "created_at_utc": "2026-01-01T00:00:01+00:00",
        "claim_scope": "M2_CALIBRATION_ONLY_NO_PERFORMANCE_CLAIM",
    }
    marker_bytes = (
        json.dumps(marker, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    binding = {
        "preparation_git_head": preparation,
        "execution_git_head": execution,
        "launch_marker_repository_path": (
            "evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json"
        ),
        "launch_marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
    }
    return preregistration, marker_bytes.decode(), binding


def test_marker_replay_uses_historical_execution_object_not_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration, marker_text, binding = _marker_replay_fixture(tmp_path)
    calls: list[tuple[str, ...]] = []

    def git_bytes(*arguments: str) -> bytes:
        calls.append(arguments)
        if arguments[:3] == ("rev-list", "--parents", "-n"):
            return f"{'a' * 40} {'b' * 40}\n".encode()
        if arguments[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            return b"evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json\n"
        if arguments[0] == "cat-file":
            return marker_text.encode()
        if arguments == ("rev-parse", "HEAD"):
            return ("f" * 40 + "\n").encode()
        raise AssertionError(arguments)

    monkeypatch.setattr(calibration_evidence_module, "_git_bytes", git_bytes)
    assert (
        calibration_evidence_module._marker_and_execution_binding(
            preregistration, "d" * 64, binding
        )
        == binding
    )
    assert ("rev-parse", "HEAD") not in calls


@pytest.mark.parametrize("field", ["execution_git_head", "launch_marker_sha256"])
def test_marker_replay_rejects_tampered_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    preregistration, marker_text, binding = _marker_replay_fixture(tmp_path)
    tampered = dict(binding)
    tampered[field] = "f" * (40 if field.endswith("head") else 64)

    def git_bytes(*arguments: str) -> bytes:
        if arguments[:3] == ("rev-list", "--parents", "-n"):
            return f"{'a' * 40} {'b' * 40}\n".encode()
        if arguments[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            return b"evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json\n"
        if arguments[0] == "cat-file":
            return marker_text.encode()
        raise AssertionError(arguments)

    monkeypatch.setattr(calibration_evidence_module, "_git_bytes", git_bytes)
    with pytest.raises(CalibrationEvidenceError):
        calibration_evidence_module._marker_and_execution_binding(
            preregistration, "d" * 64, tampered
        )


@pytest.mark.parametrize("tamper", ["parent", "diff", "blob"])
def test_marker_replay_rejects_tampered_historical_git_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    preregistration, marker_text, binding = _marker_replay_fixture(tmp_path)

    def git_bytes(*arguments: str) -> bytes:
        if arguments[:3] == ("rev-list", "--parents", "-n"):
            parent = "f" * 40 if tamper == "parent" else "b" * 40
            return f"{'a' * 40} {parent}\n".encode()
        if arguments[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            return (
                b"evidence/m2/extra.json\n"
                if tamper == "diff"
                else b"evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json\n"
            )
        if arguments[0] == "cat-file":
            return ("{}\n" if tamper == "blob" else marker_text).encode()
        raise AssertionError(arguments)

    monkeypatch.setattr(calibration_evidence_module, "_git_bytes", git_bytes)
    with pytest.raises(CalibrationEvidenceError):
        calibration_evidence_module._marker_and_execution_binding(
            preregistration, "d" * 64, binding
        )


def test_aggregates_the_preregistered_59_process_closed_set(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    preregistration = json.loads(
        (fixture.campaign_root / "CAMPAIGN_PREREGISTRATION.json").read_text(
            encoding="utf-8"
        )
    )
    command = preregistration["runner_command_template"]
    manifest_option = command.index(
        "--expected-nvidia-userspace-bundle-manifest-sha256"
    )
    assert (
        command[manifest_option + 1]
        == (preregistration["expected_nvidia_userspace_bundle_manifest_sha256"])
    )

    manifest = aggregate_campaign(fixture.campaign_root)

    assert set(manifest) == MANIFEST_FIELDS
    assert manifest["schema_version"] == CALIBRATION_COHORT_SCHEMA
    assert CALIBRATION_COHORT_SCHEMA == RUNNER_COHORT_SCHEMA
    assert manifest["protocol_schema"] == PROTOCOL_SCHEMA
    assert manifest["campaign_preregistration_file"] == (
        "CAMPAIGN_PREREGISTRATION.json"
    )
    assert manifest["campaign_preregistration_sha256"] == (
        fixture.preregistration_sha256
    )
    assert manifest["attempt_file"] == "ATTEMPTS.jsonl"
    assert manifest["attempt_prefix_record_count"] == 118
    assert manifest["attempt_count"] == manifest["run_count"] == 59
    assert MIN_CALIBRATION_RUNS == 59
    assert manifest["implementation_manifest_sha256"] == (
        fixture.implementation_manifest_sha256
    )
    assert manifest["reproducibility_fingerprint"] == (
        fixture.reproducibility_fingerprint
    )
    assert manifest["observed_max_abs_error"] == 0.1
    assert manifest["formal_atol"] == 0.125
    assert manifest["formal_rtol"] == 0.0
    assert manifest["all_passed"] is True
    assert manifest["failures"] == []
    assert [run["sequence"] for run in manifest["runs"]] == list(range(1, 60))
    assert [run["run_name"] for run in manifest["runs"]] == [
        f"run-{index:03d}" for index in range(1, 60)
    ]
    assert all(set(run) == MANIFEST_RUN_FIELDS for run in manifest["runs"])
    assert json.loads(fixture.manifest_path.read_text(encoding="utf-8")) == manifest
    candidate, candidate_sha256, evidence = validate_published_calibration_candidate(
        fixture.manifest_path,
        run_validator=_validate_run,
        expected_implementation_manifest_sha256=(
            fixture.implementation_manifest_sha256
        ),
    )
    assert candidate == manifest
    assert (
        candidate_sha256
        == hashlib.sha256(fixture.manifest_path.read_bytes()).hexdigest()
    )
    assert len(evidence.runs) == 59
    assert not list(fixture.campaign_root.glob(".*.tmp"))


def test_independent_evidence_binds_each_run_to_preregistered_nvidia_bundle(
    tmp_path: Path,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    aggregate_campaign(fixture.campaign_root)
    terminals = {
        row["run_name"]: row["validation"]
        for row in _attempt_rows(fixture.campaign_root)
        if row.get("kind") == "calibration_run" and row.get("event") == "terminal"
    }

    provenance_path = fixture.campaign_root / "run-001" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    original_root = provenance["nvidia_driver_userspace"]["root"]
    alternate_root = f"{original_root[:-1]}x"
    assert len(alternate_root) == len(original_root)
    provenance["nvidia_driver_userspace"]["root"] = alternate_root
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def ledger_validator(run_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(**terminals[run_dir.name])

    with pytest.raises(
        CalibrationEvidenceError,
        match="NVIDIA userspace root differs from preregistration",
    ):
        validate_published_calibration_candidate(
            fixture.manifest_path,
            run_validator=ledger_validator,
            expected_implementation_manifest_sha256=(
                fixture.implementation_manifest_sha256
            ),
        )


def test_rejects_tampered_execution_binding_in_run_journal(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    assert rows[0]["event"] == "submitted"
    rows[0]["execution_binding"]["execution_git_head"] = "f" * 40
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationEvidenceError, match="execution binding"):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_tampered_dagkv_head_or_snapshot_in_completed_run(
    tmp_path: Path,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    terminal = rows[1]
    assert terminal["event"] == "terminal"
    terminal["validation"]["dagkv_git_head"] = "f" * 40
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationEvidenceError, match="validation mapping drifted"):
        aggregate_campaign(fixture.campaign_root)


def test_candidate_requires_exactly_119_attempt_records(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)

    with pytest.raises(CalibrationEvidenceError, match="record count"):
        validate_published_calibration_candidate(
            fixture.manifest_path,
            run_validator=_validate_run,
        )


def test_final_bundle_requires_the_aggregate_terminal(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    aggregate_campaign(fixture.campaign_root)

    with pytest.raises(CalibrationEvidenceError, match="record count"):
        validate_published_calibration_bundle(
            fixture.manifest_path,
            run_validator=_validate_run,
        )


def test_candidate_rejects_manifest_mapping_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    aggregate_campaign(fixture.campaign_root)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["attempt_id"] = "replacement-attempt"
    fixture.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationEvidenceError, match="mapping drifted"):
        validate_published_calibration_candidate(
            fixture.manifest_path,
            run_validator=_validate_run,
        )


def test_candidate_rejects_campaign_mutation_during_replay(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    aggregate_campaign(fixture.campaign_root)

    def mutating_validator(run_dir: Path):
        validated = _validate_run(run_dir)
        if run_dir.name == "run-059":
            earlier = fixture.campaign_root / "run-001" / "result.json"
            earlier.write_bytes(earlier.read_bytes() + b" ")
        return validated

    with pytest.raises(CalibrationEvidenceError, match="changed during validation"):
        validate_published_calibration_candidate(
            fixture.manifest_path,
            run_validator=mutating_validator,
        )


def test_candidate_rejects_external_hardlink_to_evidence(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    aggregate_campaign(fixture.campaign_root)
    external_alias = tmp_path / "external-result-alias.json"
    os.link(fixture.campaign_root / "run-001" / "result.json", external_alias)

    with pytest.raises(CalibrationEvidenceError, match="external hard link"):
        validate_published_calibration_candidate(
            fixture.manifest_path,
            run_validator=_validate_run,
        )

    assert external_alias.is_file()


def test_requires_the_exclusive_manifest_location(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    outside = tmp_path / "cohort.json"

    with pytest.raises(CalibrationAggregationError, match="output must be"):
        aggregate_campaign(fixture.campaign_root, output_path=outside)

    assert not outside.exists()
    assert not fixture.manifest_path.exists()


def test_refuses_to_overwrite_a_published_manifest(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    before = fixture.manifest_path.read_bytes()

    with pytest.raises(CalibrationAggregationError, match="refusing to overwrite"):
        aggregate_campaign(fixture.campaign_root)

    assert fixture.manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pid", 0, "clean process terminal"),
        ("exit_code", 7, "clean process terminal"),
        ("timed_out", True, "clean process terminal"),
        ("started_at_utc", "2027-01-01T00:00:00+00:00", "out of order"),
    ],
)
def test_rejects_invalid_process_activity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    rows[33][field] = value
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationAggregationError, match=message):
        aggregate_campaign(fixture.campaign_root)

    assert not fixture.manifest_path.exists()


def test_rejects_log_hash_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    log = fixture.campaign_root / "run-017.stdout.log"
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(CalibrationAggregationError, match="stdout size drifted"):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_terminal_to_artifact_mapping_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    validation = rows[33]["validation"]
    assert isinstance(validation, dict)
    validation["result_sha256"] = hashlib.sha256(b"replacement").hexdigest()
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationAggregationError, match="validation mapping drifted"):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_aggregate_prefix_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    prefix = rows[-1]["calibration_prefix"]
    assert isinstance(prefix, dict)
    prefix["prefix_bytes"] = int(prefix["prefix_bytes"]) + 1
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationAggregationError, match="calibration_prefix drifted"):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_any_calibration_after_the_sealed_prefix(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    rows.append(dict(rows[0]))
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationAggregationError, match="record count"):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_preregistration_or_frozen_protocol_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    protocol = fixture.campaign_root / "run-017" / "protocol.md"
    protocol.write_text("tampered protocol\n", encoding="utf-8")

    with pytest.raises(CalibrationAggregationError, match="checksum mismatch"):
        aggregate_campaign(fixture.campaign_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("within_requested_tolerance", False, "did not pass"),
        ("gate_status", "FAILED", "did not complete"),
    ],
)
def test_rejects_failed_result_even_when_the_ledger_claims_passed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    result_path = fixture.campaign_root / "run-017" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationAggregationError, match=message):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_undeclared_or_missing_attempt_directories(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    (fixture.campaign_root / "run-060").mkdir()

    with pytest.raises(CalibrationAggregationError, match="undeclared entries"):
        aggregate_campaign(fixture.campaign_root)


def test_rejects_frozen_evidence_helper_drift(tmp_path: Path) -> None:
    helper = tmp_path / "m2_calibration_evidence.py"
    helper.write_text("frozen helper\n", encoding="utf-8")
    entry = {
        "path": str(helper),
        "size": helper.stat().st_size,
        "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
    }
    helper.write_text("drifted helper\n", encoding="utf-8")

    with pytest.raises(CalibrationEvidenceError, match="size drifted"):
        _validate_frozen_entry(entry, label="frozen evidence")


def test_rejects_python_entrypoint_retargeting(tmp_path: Path) -> None:
    first = tmp_path / "python-a"
    second = tmp_path / "python-b"
    first.write_bytes(b"interpreter-a")
    second.write_bytes(b"interpreter-b")
    entrypoint = tmp_path / "venv" / "bin" / "python"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(first)
    entry = {
        "path": str(entrypoint),
        "size": first.stat().st_size,
        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
    }
    _validate_frozen_entry(
        entry,
        label="Python executable",
        allow_executable_symlink=True,
    )
    entrypoint.unlink()
    entrypoint.symlink_to(second)

    with pytest.raises(CalibrationEvidenceError, match="SHA-256 drifted"):
        _validate_frozen_entry(
            entry,
            label="Python executable",
            allow_executable_symlink=True,
        )


@pytest.mark.parametrize(
    ("row_index", "source", "message"),
    [
        (0, "preregistration", "frozen predecessor"),
        (2, "previous_terminal", "frozen predecessor"),
        (118, "last_terminal", "aggregate started before"),
    ],
)
def test_rejects_campaign_chronology_tampering(
    tmp_path: Path,
    row_index: int,
    source: str,
    message: str,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=False)
    rows = _attempt_rows(fixture.campaign_root)
    preregistration = json.loads(
        (fixture.campaign_root / "CAMPAIGN_PREREGISTRATION.json").read_text()
    )
    if source == "preregistration":
        timestamp = datetime.fromisoformat(preregistration["created_at_utc"])
        rows[row_index]["timestamp_utc"] = (
            timestamp - timedelta(seconds=1)
        ).isoformat()
    elif source == "previous_terminal":
        timestamp = datetime.fromisoformat(str(rows[1]["timestamp_utc"]))
        rows[row_index]["timestamp_utc"] = (
            timestamp - timedelta(seconds=1)
        ).isoformat()
    else:
        timestamp = datetime.fromisoformat(str(rows[117]["timestamp_utc"]))
        rows[row_index]["timestamp_utc"] = (
            timestamp - timedelta(seconds=1)
        ).isoformat()
    _write_attempt_rows(fixture.campaign_root, rows)

    with pytest.raises(CalibrationAggregationError, match=message):
        aggregate_campaign(fixture.campaign_root)


@pytest.mark.parametrize(
    "script_name",
    [
        "aggregate_m2_calibration.py",
        "freeze_m2_tolerance.py",
        "run_m2_vllm_abba.py",
    ],
)
def test_tools_import_in_absolute_script_mode(
    tmp_path: Path,
    script_name: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "integrations" / "vllm_m2")

    completed = subprocess.run(
        [sys.executable, str(repo_root / "tools" / script_name), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
