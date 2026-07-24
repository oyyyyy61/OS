from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.m2_formal_evidence as formal_evidence
import tools.run_m2_formal_campaign as formal_campaign
from tools.m2_formal_evidence import (
    SEAL_NAME,
    FormalEvidenceError,
    publish_formal_bundle_seal,
    validate_published_formal_bundle,
)
from tools.run_m2_formal_campaign import (
    ACCEPTANCE_NAME,
    ATTEMPTS_NAME,
    PREREGISTRATION_NAME,
    PRODUCTION_RUN_COUNT,
    CampaignConfig,
    FormalCampaignError,
    execute_prepared_campaign,
    prepare_campaign,
    run_campaign,
)

IMPLEMENTATION = hashlib.sha256(b"formal-implementation").hexdigest()
FINGERPRINT = hashlib.sha256(b"formal-fingerprint").hexdigest()
PREPARATION_HEAD = "1" * 40
EXECUTION_HEAD = "2" * 40
MARKER_SHA256 = hashlib.sha256(b"formal-launch-marker").hexdigest()


def _execution_binding() -> dict[str, str]:
    return {
        "preparation_git_head": PREPARATION_HEAD,
        "execution_git_head": EXECUTION_HEAD,
        "launch_marker_repository_path": (
            formal_campaign.LAUNCH_MARKER_REPOSITORY_PATH
        ),
        "launch_marker_sha256": MARKER_SHA256,
    }


FAKE_RUNNER = r"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--tolerance-file", type=Path, required=True)
parser.add_argument("--calibration-manifest", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--vllm-root", required=True)
parser.add_argument("--cpu-bytes", required=True)
parser.add_argument("--timeout-s", required=True)
parser.add_argument("--cuda-device", required=True)
parser.add_argument("--full-provenance", action="store_true")
args = parser.parse_args()
assert args.mode == "formal"
assert args.full_provenance
assert args.tolerance_file.is_file()
assert args.calibration_manifest.is_file()
index = int(args.output_dir.name.removeprefix("run-"))
mode, _, target = os.environ.get("FAKE_FORMAL_MODE", "success").partition(":")
selected = bool(target) and index == int(target)
args.output_dir.mkdir()
if selected and mode == "timeout":
    (args.output_dir / "partial.txt").write_text("timeout\n", encoding="utf-8")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)
if selected and mode == "nonzero":
    (args.output_dir / "failure.txt").write_text("failed\n", encoding="utf-8")
    raise SystemExit(7)
payload = {
    "run_id": f"formal-run-{index:03d}",
    "sequence": index,
}
for name in (
    "result.json",
    "provenance.json",
    "SHA256SUMS",
    "M2_ITEM8_FORMAL_RUN_MANIFEST.json",
):
    (args.output_dir / name).write_text(
        json.dumps({**payload, "artifact": name}) + "\n", encoding="utf-8"
    )
print(f"completed {args.output_dir.name}", flush=True)
"""

FAKE_AGGREGATOR = r"""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--campaign-dir", type=Path, required=True)
parser.add_argument("--calibration-manifest", required=True)
parser.add_argument("--frozen-tolerance", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("FAKE_FORMAL_AGGREGATOR_FAIL") == "1":
    raise SystemExit(9)
prereg = json.loads(
    (args.campaign_dir / "FORMAL_CAMPAIGN_PREREGISTRATION.json").read_text()
)
rows = [
    json.loads(line)
    for line in (args.campaign_dir / "FORMAL_ATTEMPTS.jsonl")
    .read_text()
    .splitlines()
]
validations = [
    rows[index]["validation"]
    for index in range(1, len(prereg["run_names"]) * 2, 2)
]
runs = sorted(
    [
        {
            "run_id": row["run_id"],
            "formal_run_manifest_sha256": row["formal_run_manifest_sha256"],
            "result_sha256": row["result_sha256"],
            "provenance_sha256": row["provenance_sha256"],
            "sha256sums_sha256": row["sha256sums_sha256"],
        }
        for row in validations
    ],
    key=lambda row: row["run_id"],
)
manifest = {
    "schema_version": "dagkv.m2.item8.acceptance.v1",
    "protocol_schema": "dagkv.m2.vllm_abba.v2",
    "gate_status": "M2_ITEM8_ACCEPTED",
    "run_count": len(runs),
    "passed_run_count": len(runs),
    "m2_item8_accepted": True,
    "m2_accepted": False,
    "performance_claims_supported": False,
    "frozen_tolerance_sha256": prereg["parent_binding"]["frozen_tolerance_sha256"],
    "calibration_manifest_sha256": prereg["parent_binding"][
        "calibration_manifest_sha256"
    ],
    "reproducibility_fingerprint": prereg["expected_reproducibility_fingerprint"],
    "protocol_sha256": prereg["data_plane_protocol_sha256"],
    "runs": runs,
    "statement": (
        "Exactly twenty frozen M2 item 8 holdouts passed. This closes item 8 "
        "only; the aggregate M2 gate remains open and this evidence supports "
        "no latency, throughput, hit-rate, scheduling-policy, or "
        "paper-performance claim."
    ),
}
args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: str) -> None:
    path.write_text(payload.lstrip(), encoding="utf-8")


def _fake_validation(
    run_dir: Path,
    *,
    calibration_sha256: str,
    tolerance_sha256: str,
) -> dict[str, object]:
    index = int(run_dir.name.removeprefix("run-"))
    digest = hashlib.sha256(run_dir.name.encode()).hexdigest()
    return {
        "run_id": f"formal-run-{index:03d}",
        "result_sha256": digest,
        "provenance_sha256": hashlib.sha256(f"p{index}".encode()).hexdigest(),
        "sha256sums_sha256": hashlib.sha256(f"s{index}".encode()).hexdigest(),
        "formal_run_manifest_sha256": hashlib.sha256(f"f{index}".encode()).hexdigest(),
        "frozen_tolerance_sha256": tolerance_sha256,
        "calibration_manifest_sha256": calibration_sha256,
        "implementation_manifest_sha256": IMPLEMENTATION,
        "reproducibility_fingerprint": FINGERPRINT,
        "protocol_sha256": _sha(formal_campaign.DATA_PLANE_PROTOCOL_PATH),
        "observed_max_abs_error": 0.1,
        "minimum_top1_margin": 0.5,
        "dagkv_git_head": EXECUTION_HEAD,
        "dagkv_snapshot_sha256": hashlib.sha256(b"dagkv-snapshot").hexdigest(),
    }


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CampaignConfig:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    runner = inputs / "runner.py"
    aggregator = inputs / "aggregator.py"
    calibration = inputs / "M2_CALIBRATION_MANIFEST.json"
    tolerance = inputs / "M2_FROZEN_TOLERANCE.json"
    model = inputs / "model"
    vllm = inputs / "vllm"
    _write(runner, FAKE_RUNNER)
    _write(aggregator, FAKE_AGGREGATOR)
    calibration.write_text("{}\n", encoding="utf-8")
    tolerance.write_text("{}\n", encoding="utf-8")
    model.mkdir()
    vllm.mkdir()
    parent = {
        "calibration_manifest_sha256": _sha(calibration),
        "frozen_tolerance_sha256": _sha(tolerance),
        "reproducibility_fingerprint": FINGERPRINT,
        "frozen_at_utc": "2026-07-24T00:00:00+00:00",
        "calibration_run_count": 59,
    }
    monkeypatch.setattr(
        formal_campaign,
        "_current_implementation_manifest_sha256",
        lambda: IMPLEMENTATION,
    )
    monkeypatch.setattr(
        formal_campaign,
        "_validate_parent_inputs",
        lambda *_args, **_kwargs: dict(parent),
    )
    monkeypatch.setattr(
        formal_campaign,
        "_preparation_git_head",
        lambda: PREPARATION_HEAD,
    )
    monkeypatch.setattr(
        formal_campaign,
        "_establish_execution_binding",
        lambda *_args, **_kwargs: _execution_binding(),
    )
    monkeypatch.setattr(
        formal_campaign,
        "_revalidate_execution_binding",
        lambda *_args, **_kwargs: None,
    )

    def validate_run(run_dir: Path, **_: object) -> dict[str, object]:
        return _fake_validation(
            run_dir,
            calibration_sha256=str(parent["calibration_manifest_sha256"]),
            tolerance_sha256=str(parent["frozen_tolerance_sha256"]),
        )

    monkeypatch.setattr(formal_campaign, "_validate_completed_run", validate_run)
    monkeypatch.setenv("FAKE_FORMAL_MODE", "success")
    monkeypatch.delenv("FAKE_FORMAL_AGGREGATOR_FAIL", raising=False)
    return CampaignConfig(
        campaign_root=tmp_path / "formal-campaign",
        calibration_manifest=calibration,
        frozen_tolerance=tolerance,
        expected_implementation_manifest_sha256=IMPLEMENTATION,
        expected_reproducibility_fingerprint=FINGERPRINT,
        python_executable=Path(sys.executable),
        runner=runner,
        aggregator=aggregator,
        model=model,
        vllm_root=vllm,
        cpu_bytes=1024,
        runner_timeout_s=1,
        process_timeout_s=1,
        aggregation_timeout_s=2,
        terminate_grace_s=0.05,
        kill_wait_s=1,
    )


def _attempts(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (root / ATTEMPTS_NAME).read_text(encoding="utf-8").splitlines()
    ]


def test_prepare_then_execute_three_fake_holdouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    preregistration_sha = prepare_campaign(config, _expected_runs=3)
    assert [path.name for path in config.campaign_root.iterdir()] == [
        PREREGISTRATION_NAME
    ]

    acceptance = execute_prepared_campaign(
        config.campaign_root, preregistration_sha, _expected_runs=3
    )

    assert acceptance["run_count"] == 3
    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 8
    assert [attempts[index]["run_name"] for index in range(0, 6, 2)] == [
        "run-001",
        "run-002",
        "run-003",
    ]
    assert all(attempts[index]["status"] == "passed" for index in (1, 3, 5, 7))
    assert attempts[6]["formal_prefix"]["prefix_record_count"] == 6
    assert (config.campaign_root / ACCEPTANCE_NAME).is_file()


def test_exact_twenty_run_order_and_final_replay_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(
        formal_campaign,
        "_revalidate_production_candidate",
        lambda *_args, **_kwargs: events.append("candidate"),
    )
    monkeypatch.setattr(
        formal_campaign,
        "_revalidate_production_bundle",
        lambda *_args, **_kwargs: events.append("bundle"),
    )
    monkeypatch.setattr(
        formal_campaign,
        "_publish_and_revalidate_production_seal",
        lambda *_args, **_kwargs: events.append("seal") or "a" * 64,
    )

    preregistration_sha = prepare_campaign(config)
    preregistration = json.loads(
        (config.campaign_root / PREREGISTRATION_NAME).read_text(encoding="utf-8")
    )
    assert preregistration["preparation_git_head"] == PREPARATION_HEAD
    assert preregistration["launch_marker_repository_path"] == (
        formal_campaign.LAUNCH_MARKER_REPOSITORY_PATH
    )
    acceptance = execute_prepared_campaign(config.campaign_root, preregistration_sha)

    assert PRODUCTION_RUN_COUNT == acceptance["run_count"] == 20
    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 42
    assert [attempts[index]["run_name"] for index in range(0, 40, 2)] == [
        f"run-{index:03d}" for index in range(1, 21)
    ]
    assert events == ["candidate", "bundle", "seal"]
    submitted = [row for row in attempts if row["event"] == "submitted"]
    assert len(submitted) == 21
    assert all(row["execution_binding"] == _execution_binding() for row in submitted)


def test_production_shortcut_is_rejected_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(FormalCampaignError, match="separate prepare and execute"):
        run_campaign(config)

    assert not config.campaign_root.exists()


@pytest.mark.parametrize(
    ("failure"),
    [
        "DAGKV worktree must be clean during formal execution",
        "formal launch marker differs from the committed Git object",
    ],
)
def test_production_execute_rejects_git_or_marker_drift_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    preregistration_sha = prepare_campaign(config)

    def reject_binding(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise FormalCampaignError(failure)

    monkeypatch.setattr(formal_campaign, "_establish_execution_binding", reject_binding)
    with pytest.raises(FormalCampaignError, match=failure):
        execute_prepared_campaign(config.campaign_root, preregistration_sha)

    assert not (config.campaign_root / ATTEMPTS_NAME).exists()


def test_competing_execute_lock_fails_before_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    preregistration_sha = prepare_campaign(config, _expected_runs=2)
    descriptor = os.open(
        config.campaign_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FormalCampaignError, match="already executing"):
            execute_prepared_campaign(
                config.campaign_root,
                preregistration_sha,
                _expected_runs=2,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert not (config.campaign_root / ATTEMPTS_NAME).exists()


def test_first_nonzero_exit_stops_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FORMAL_MODE", "nonzero:2")

    with pytest.raises(FormalCampaignError, match="stopped without retry"):
        run_campaign(config, _expected_runs=4)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 4
    assert attempts[-1]["run_name"] == "run-002"
    assert attempts[-1]["status"] == "process_failed"
    assert not (config.campaign_root / "run-003").exists()


def test_timeout_kills_process_group_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FORMAL_MODE", "timeout:1")

    with pytest.raises(FormalCampaignError, match="stopped without retry"):
        run_campaign(config, _expected_runs=2)

    terminal = _attempts(config.campaign_root)[-1]
    assert terminal["status"] == "timed_out"
    assert terminal["sigterm_sent"] is True
    assert terminal["sigkill_sent"] is True


def test_execute_rejects_frozen_runner_drift_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    preregistration_sha = prepare_campaign(config, _expected_runs=2)
    config.runner.write_text(
        config.runner.read_text() + "\n# drift\n", encoding="utf-8"
    )

    with pytest.raises(FormalCampaignError, match="frozen-file inventory drifted"):
        execute_prepared_campaign(
            config.campaign_root, preregistration_sha, _expected_runs=2
        )

    assert not (config.campaign_root / ATTEMPTS_NAME).exists()


def test_aggregation_failure_is_terminal_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FORMAL_AGGREGATOR_FAIL", "1")

    with pytest.raises(FormalCampaignError, match="aggregation failed"):
        run_campaign(config, _expected_runs=2)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 6
    assert attempts[-2]["kind"] == "aggregate"
    assert attempts[-1]["status"] == "process_failed"
    assert not (config.campaign_root / ACCEPTANCE_NAME).exists()


def _completed_unsealed_production_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CampaignConfig, str]:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        formal_campaign,
        "_revalidate_production_candidate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        formal_campaign,
        "_revalidate_production_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        formal_campaign,
        "_publish_and_revalidate_production_seal",
        lambda *_args, **_kwargs: "a" * 64,
    )
    preregistration_sha = prepare_campaign(config)
    execute_prepared_campaign(config.campaign_root, preregistration_sha)
    preregistration_sha = _sha(config.campaign_root / PREREGISTRATION_NAME)
    calibration_sha = _sha(config.calibration_manifest)
    tolerance_sha = _sha(config.frozen_tolerance)
    parent = SimpleNamespace(
        calibration_manifest_sha256=calibration_sha,
        frozen_tolerance_sha256=tolerance_sha,
        reproducibility_fingerprint=FINGERPRINT,
        frozen_at_utc=datetime.fromisoformat("2026-07-24T00:00:00+00:00"),
        calibration_run_ids=frozenset(
            f"calibration-run-{index:03d}" for index in range(1, 60)
        ),
    )
    monkeypatch.setattr(
        formal_evidence,
        "_current_implementation_sha256",
        lambda: IMPLEMENTATION,
    )
    monkeypatch.setattr(
        formal_evidence,
        "_validate_parent",
        lambda _preregistration: parent,
    )
    monkeypatch.setattr(
        formal_evidence,
        "_validate_execution_binding",
        lambda binding, **_kwargs: dict(binding),
    )
    monkeypatch.setattr(
        formal_evidence,
        "_validate_formal_run",
        lambda run_dir: _fake_validation(
            run_dir,
            calibration_sha256=calibration_sha,
            tolerance_sha256=tolerance_sha,
        ),
    )
    return config, preregistration_sha


def test_formal_evidence_publishes_and_replays_create_only_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, preregistration_sha = _completed_unsealed_production_campaign(
        tmp_path, monkeypatch
    )

    seal, seal_sha = publish_formal_bundle_seal(
        config.campaign_root,
        expected_preregistration_sha256=preregistration_sha,
    )
    replayed, observed_sha, validation = validate_published_formal_bundle(
        config.campaign_root / SEAL_NAME,
        expected_seal_sha256=seal_sha,
        expected_preregistration_sha256=preregistration_sha,
    )

    assert replayed == seal
    assert observed_sha == seal_sha
    assert validation.formal_prefix_bytes < validation.full_journal_bytes
    assert len(validation.ordered_runs) == 20
    assert seal["formal_prefix_record_count"] == 40
    assert seal["full_journal_record_count"] == 42
    with pytest.raises(FormalEvidenceError, match="already exists"):
        publish_formal_bundle_seal(
            config.campaign_root,
            expected_preregistration_sha256=preregistration_sha,
        )


@pytest.mark.parametrize("target", ["journal", "aggregate_log"])
def test_published_formal_evidence_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    config, preregistration_sha = _completed_unsealed_production_campaign(
        tmp_path, monkeypatch
    )
    _, seal_sha = publish_formal_bundle_seal(
        config.campaign_root,
        expected_preregistration_sha256=preregistration_sha,
    )
    path = (
        config.campaign_root / ATTEMPTS_NAME
        if target == "journal"
        else config.campaign_root / "aggregate.stdout.log"
    )
    path.write_bytes(path.read_bytes() + b"tampered\n")

    with pytest.raises(FormalEvidenceError):
        validate_published_formal_bundle(
            config.campaign_root / SEAL_NAME,
            expected_seal_sha256=seal_sha,
            expected_preregistration_sha256=preregistration_sha,
        )


def test_formal_evidence_rejects_hardlinked_root_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, preregistration_sha = _completed_unsealed_production_campaign(
        tmp_path, monkeypatch
    )
    acceptance = config.campaign_root / ACCEPTANCE_NAME
    alias = tmp_path / "acceptance-hardlink.json"
    os.link(acceptance, alias)

    with pytest.raises(FormalEvidenceError, match="hard link"):
        publish_formal_bundle_seal(
            config.campaign_root,
            expected_preregistration_sha256=preregistration_sha,
        )


def test_formal_evidence_reconstructs_frozen_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _completed_unsealed_production_campaign(tmp_path, monkeypatch)
    preregistration_path = config.campaign_root / PREREGISTRATION_NAME
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    preregistration["runner_command_template"].append("--unregistered-option")
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FormalEvidenceError, match="command template drifted"):
        publish_formal_bundle_seal(
            config.campaign_root,
            expected_preregistration_sha256=_sha(preregistration_path),
        )


def test_formal_evidence_reconstructs_marker_git_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    preregistration_sha = prepare_campaign(config)
    preregistration = json.loads(
        (config.campaign_root / PREREGISTRATION_NAME).read_text(encoding="utf-8")
    )
    marker = {
        "schema_version": formal_campaign.MARKER_SCHEMA,
        "campaign_id": preregistration["campaign_id"],
        "campaign_root": preregistration["campaign_root"],
        "campaign_preregistration_sha256": preregistration_sha,
        "preparation_git_head": PREPARATION_HEAD,
        "created_at_utc": preregistration["created_at_utc"],
        "claim_scope": formal_campaign.MARKER_CLAIM_SCOPE,
    }
    marker_raw = (
        json.dumps(marker, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    binding = {
        **_execution_binding(),
        "launch_marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
    }

    def git_bytes(*arguments: str) -> bytes:
        if arguments[:4] == ("rev-list", "--parents", "-n", "1"):
            return f"{EXECUTION_HEAD} {PREPARATION_HEAD}\n".encode()
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return f"{formal_campaign.LAUNCH_MARKER_REPOSITORY_PATH}\n".encode()
        if arguments[:2] == ("cat-file", "blob"):
            return marker_raw
        raise AssertionError(arguments)

    monkeypatch.setattr(formal_evidence, "_git_bytes", git_bytes)
    assert (
        formal_evidence._validate_execution_binding(
            binding,
            preregistration=preregistration,
            preregistration_sha256=preregistration_sha,
        )
        == binding
    )
