from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.m2_calibration_fixtures import build_calibration_campaign
from tools.freeze_m2_tolerance import (
    TOLERANCE_DERIVATION,
    TOLERANCE_FIELDS,
    TOLERANCE_SCHEMA,
    ToleranceFreezeError,
    _parser,
    freeze_tolerance,
    validate_calibration_cohort,
)
from tools.run_m2_vllm_abba import TOLERANCE_SCHEMA as RUNNER_TOLERANCE_SCHEMA
from tools.run_m2_vllm_abba import (
    M2ValidationError,
    load_calibration_cohort,
    load_frozen_tolerance,
)


def test_freezes_full_bundle_and_runner_revalidates_after_freeze(
    tmp_path: Path,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    manifest_sha256 = hashlib.sha256(fixture.manifest_path.read_bytes()).hexdigest()
    output = tmp_path / "evidence" / "M2_FROZEN_TOLERANCE.json"

    frozen = freeze_tolerance(
        fixture.manifest_path,
        output,
        expected_manifest_sha256=manifest_sha256,
        frozen_at=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert set(frozen) == TOLERANCE_FIELDS
    assert frozen == json.loads(output.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == TOLERANCE_SCHEMA == RUNNER_TOLERANCE_SCHEMA
    assert frozen["frozen"] is True
    assert frozen["atol"] == 0.125
    assert frozen["rtol"] == 0.0
    assert frozen["calibration_manifest_sha256"] == manifest_sha256
    assert frozen["reproducibility_fingerprint"] == (
        fixture.reproducibility_fingerprint
    )
    assert frozen["calibration_run_count"] == 59
    assert frozen["derivation"] == TOLERANCE_DERIVATION

    evidence = validate_calibration_cohort(
        fixture.manifest_path,
        expected_manifest_sha256=manifest_sha256,
    )
    assert evidence.manifest_sha256 == manifest_sha256
    run_started_ns = time.time_ns() + 1_000_000_000
    loaded = load_frozen_tolerance(output, run_started_ns=run_started_ns)
    cohort = load_calibration_cohort(
        fixture.manifest_path,
        frozen_tolerance=loaded,
        run_started_ns=run_started_ns,
        expected_implementation_manifest_sha256=(
            fixture.implementation_manifest_sha256
        ),
    )
    assert cohort["campaign_preregistration_sha256"] == (fixture.preregistration_sha256)
    assert not list(output.parent.glob(".*.tmp"))


def test_rejects_byte_level_manifest_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    expected_sha256 = hashlib.sha256(fixture.manifest_path.read_bytes()).hexdigest()
    fixture.manifest_path.write_text(
        fixture.manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence" / "tolerance.json"

    with pytest.raises(ToleranceFreezeError, match="differs from the expected"):
        freeze_tolerance(
            fixture.manifest_path,
            output,
            expected_manifest_sha256=expected_sha256,
        )

    assert not output.exists()


def test_rejects_upstream_run_log_tampering_after_manifest_publication(
    tmp_path: Path,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    log = fixture.campaign_root / "run-023.stdout.log"
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    output = tmp_path / "evidence" / "tolerance.json"

    with pytest.raises(ToleranceFreezeError, match="stdout size drifted"):
        freeze_tolerance(fixture.manifest_path, output)

    assert not output.exists()


def test_rejects_aggregate_terminal_tampering(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    attempts = fixture.campaign_root / "ATTEMPTS.jsonl"
    rows = [json.loads(line) for line in attempts.read_text().splitlines()]
    rows[-1]["exit_code"] = 9
    attempts.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence" / "tolerance.json"

    with pytest.raises(ToleranceFreezeError, match="clean terminal"):
        freeze_tolerance(fixture.manifest_path, output)


def test_rejects_current_implementation_drift_in_formal_loader(
    tmp_path: Path,
) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    manifest_sha256 = hashlib.sha256(fixture.manifest_path.read_bytes()).hexdigest()
    output = tmp_path / "evidence" / "tolerance.json"
    freeze_tolerance(fixture.manifest_path, output)
    run_started_ns = time.time_ns() + 1_000_000_000
    loaded = load_frozen_tolerance(output, run_started_ns=run_started_ns)

    with pytest.raises(M2ValidationError, match="current implementation differs"):
        load_calibration_cohort(
            fixture.manifest_path,
            frozen_tolerance=loaded,
            run_started_ns=run_started_ns,
            expected_implementation_manifest_sha256=hashlib.sha256(
                b"drifted"
            ).hexdigest(),
        )

    assert loaded.calibration_manifest_sha256 == manifest_sha256


def test_refuses_tolerance_output_inside_campaign(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    output = fixture.campaign_root / "M2_FROZEN_TOLERANCE.json"

    with pytest.raises(ToleranceFreezeError, match="outside the calibration"):
        freeze_tolerance(fixture.manifest_path, output)

    assert not output.exists()
    validate_calibration_cohort(fixture.manifest_path)


def test_refuses_symlinked_output_parent_into_campaign(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    alias = tmp_path / "campaign-alias"
    alias.symlink_to(fixture.campaign_root, target_is_directory=True)
    output = alias / "M2_FROZEN_TOLERANCE.json"

    with pytest.raises(ToleranceFreezeError, match="outside the calibration"):
        freeze_tolerance(fixture.manifest_path, output)

    assert not (fixture.campaign_root / output.name).exists()


def test_refuses_to_overwrite_frozen_tolerance(tmp_path: Path) -> None:
    fixture = build_calibration_campaign(tmp_path, publish=True)
    output = tmp_path / "evidence" / "tolerance.json"
    output.parent.mkdir()
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(ToleranceFreezeError, match="refusing to overwrite"):
        freeze_tolerance(fixture.manifest_path, output)

    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_cli_requires_an_explicit_external_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--calibration-manifest", str(tmp_path / "manifest")])
