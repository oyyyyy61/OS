from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from dataclasses import replace
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
from tools.nvidia_driver_userspace_bundle import (
    BundleValidation,
    NvidiaUserspaceBundleError,
    RuntimeMapping,
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
NVIDIA_DRIVER_VERSION = "580.173.02"


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
parser.add_argument("--expected-nvidia-driver-version", required=True)
parser.add_argument("--nvidia-userspace-bundle-root", type=Path, required=True)
parser.add_argument("--expected-nvidia-userspace-bundle-manifest-sha256", required=True)
parser.add_argument("--expected-nvidia-userspace-bundle-content-digest", required=True)
parser.add_argument("--cpu-bytes", required=True)
parser.add_argument("--timeout-s", required=True)
parser.add_argument("--cuda-device", required=True)
parser.add_argument("--full-provenance", action="store_true")
args = parser.parse_args()
assert args.mode == "formal"
assert args.full_provenance
assert args.tolerance_file.is_file()
assert args.calibration_manifest.is_file()
assert args.expected_nvidia_driver_version == os.environ["FAKE_NVIDIA_DRIVER_VERSION"]
assert args.nvidia_userspace_bundle_root == Path(os.environ["FAKE_BUNDLE_ROOT"])
assert args.expected_nvidia_userspace_bundle_manifest_sha256 == os.environ[
    "FAKE_BUNDLE_MANIFEST_SHA256"
]
assert args.expected_nvidia_userspace_bundle_content_digest == os.environ[
    "FAKE_BUNDLE_CONTENT_DIGEST"
]
assert os.environ["LD_LIBRARY_PATH"] == (
    f"{os.environ['FAKE_BUNDLE_LIBRARY_PATH']}:/usr/local/cuda/lib64"
)
assert "LD_AUDIT" not in os.environ and "LD_PRELOAD" not in os.environ
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
if os.environ.get("FAKE_FORMAL_BUNDLE_TAMPER_RUN") == str(index):
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered during formal run\n", encoding="utf-8"
    )
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
    "schema_version": "dagkv.m2.item8.acceptance.v2",
    "protocol_schema": "dagkv.m2.vllm_abba.v3",
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
    "nvidia_userspace_bundle_root": prereg["nvidia_userspace_bundle_root"],
    "nvidia_userspace_bundle_manifest_sha256": prereg[
        "expected_nvidia_userspace_bundle_manifest_sha256"
    ],
    "nvidia_userspace_bundle_content_digest": prereg[
        "expected_nvidia_userspace_bundle_content_digest"
    ],
    "nvidia_driver_version": prereg["expected_nvidia_driver_version"],
    "runs": runs,
    "statement": (
        "Exactly twenty frozen M2 item 8 holdouts passed. This closes item 8 "
        "only; the aggregate M2 gate remains open and this evidence supports "
        "no latency, throughput, hit-rate, scheduling-policy, or "
        "paper-performance claim."
    ),
}
args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
if os.environ.get("FAKE_FORMAL_AGGREGATOR_BUNDLE_TAMPER") == "1":
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered during formal aggregate\n", encoding="utf-8"
    )
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: str) -> None:
    path.write_text(payload.lstrip(), encoding="utf-8")


def _fake_bundle(root: Path) -> tuple[Path, str, str, Path, Path]:
    bundle_root = root / "nvidia-bundle"
    library_path = bundle_root / "rootfs/usr/lib/x86_64-linux-gnu"
    binary_path = bundle_root / "rootfs/usr/bin"
    library_path.mkdir(parents=True)
    binary_path.mkdir(parents=True)
    manifest_path = bundle_root / "NVIDIA_USERSPACE_BUNDLE_MANIFEST.json"
    content_path = bundle_root / "fixture-content.bin"
    manifest_path.write_bytes(b'{"fixture":"nvidia-bundle"}\n')
    content_path.write_bytes(b"closed fixture content\n")
    return (
        bundle_root,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        hashlib.sha256(content_path.read_bytes()).hexdigest(),
        library_path,
        content_path,
    )


def _fake_bundle_validator(
    bundle_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    **_: object,
) -> BundleValidation:
    root = bundle_root.absolute()
    manifest_path = root / "NVIDIA_USERSPACE_BUNDLE_MANIFEST.json"
    content_path = root / "fixture-content.bin"
    try:
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        content_digest = hashlib.sha256(content_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise NvidiaUserspaceBundleError("fixture bundle is unavailable") from exc
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise NvidiaUserspaceBundleError("bundle manifest digest differs")
    runtime_root = root / "rootfs"
    library_path = runtime_root / "usr/lib/x86_64-linux-gnu"
    return BundleValidation(
        manifest={"fixture": "nvidia-bundle"},
        manifest_sha256=manifest_sha256,
        content_digest=content_digest,
        kernel_module_version=NVIDIA_DRIVER_VERSION,
        stat_snapshot=(),
        runtime=RuntimeMapping(
            rootfs=runtime_root,
            library_directory=library_path,
            nvidia_smi=runtime_root / "usr/bin/nvidia-smi",
            libcuda=library_path / f"libcuda.so.{NVIDIA_DRIVER_VERSION}",
            libnvidia_ml=library_path / f"libnvidia-ml.so.{NVIDIA_DRIVER_VERSION}",
        ),
    )


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
    (
        bundle_root,
        bundle_manifest_sha256,
        bundle_content_digest,
        bundle_library_path,
        bundle_content_path,
    ) = _fake_bundle(tmp_path)
    parent = {
        "calibration_manifest_sha256": _sha(calibration),
        "frozen_tolerance_sha256": _sha(tolerance),
        "reproducibility_fingerprint": FINGERPRINT,
        "frozen_at_utc": "2026-07-24T00:00:00+00:00",
        "calibration_run_count": 59,
        "nvidia_userspace_bundle_root": str(bundle_root),
        "expected_nvidia_userspace_bundle_manifest_sha256": (bundle_manifest_sha256),
        "expected_nvidia_userspace_bundle_content_digest": bundle_content_digest,
        "expected_nvidia_driver_version": NVIDIA_DRIVER_VERSION,
    }
    monkeypatch.setenv("FAKE_NVIDIA_DRIVER_VERSION", NVIDIA_DRIVER_VERSION)
    monkeypatch.setenv("FAKE_BUNDLE_ROOT", str(bundle_root))
    monkeypatch.setenv("FAKE_BUNDLE_MANIFEST_SHA256", bundle_manifest_sha256)
    monkeypatch.setenv("FAKE_BUNDLE_CONTENT_DIGEST", bundle_content_digest)
    monkeypatch.setenv("FAKE_BUNDLE_LIBRARY_PATH", str(bundle_library_path))
    monkeypatch.setenv("FAKE_BUNDLE_CONTENT_PATH", str(bundle_content_path))
    monkeypatch.setattr(formal_campaign, "validate_bundle", _fake_bundle_validator)
    monkeypatch.setattr(formal_evidence, "validate_bundle", _fake_bundle_validator)
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
    monkeypatch.delenv("FAKE_FORMAL_BUNDLE_TAMPER_RUN", raising=False)
    monkeypatch.delenv("FAKE_FORMAL_AGGREGATOR_BUNDLE_TAMPER", raising=False)
    return CampaignConfig(
        campaign_root=tmp_path / "formal-campaign",
        calibration_manifest=calibration,
        frozen_tolerance=tolerance,
        expected_implementation_manifest_sha256=IMPLEMENTATION,
        expected_reproducibility_fingerprint=FINGERPRINT,
        nvidia_userspace_bundle_root=bundle_root,
        expected_nvidia_userspace_bundle_manifest_sha256=bundle_manifest_sha256,
        expected_nvidia_userspace_bundle_content_digest=bundle_content_digest,
        expected_nvidia_driver_version=NVIDIA_DRIVER_VERSION,
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
    preregistration = json.loads(
        (config.campaign_root / PREREGISTRATION_NAME).read_text(encoding="utf-8")
    )
    assert preregistration["data_plane_protocol_schema"] == ("dagkv.m2.vllm_abba.v3")
    assert preregistration["nvidia_userspace_bundle_root"] == str(
        config.nvidia_userspace_bundle_root
    )
    assert (
        preregistration["expected_nvidia_userspace_bundle_manifest_sha256"]
        == config.expected_nvidia_userspace_bundle_manifest_sha256
    )
    assert (
        preregistration["expected_nvidia_userspace_bundle_content_digest"]
        == config.expected_nvidia_userspace_bundle_content_digest
    )
    assert preregistration["expected_nvidia_driver_version"] == (NVIDIA_DRIVER_VERSION)
    assert preregistration["parent_binding"]["expected_nvidia_driver_version"] == (
        NVIDIA_DRIVER_VERSION
    )
    command = preregistration["runner_command_template"]
    assert command[command.index("--expected-nvidia-driver-version") + 1] == (
        NVIDIA_DRIVER_VERSION
    )
    assert command[command.index("--nvidia-userspace-bundle-root") + 1] == str(
        config.nvidia_userspace_bundle_root
    )
    assert (
        command[command.index("--expected-nvidia-userspace-bundle-manifest-sha256") + 1]
        == config.expected_nvidia_userspace_bundle_manifest_sha256
    )
    assert preregistration["environment_overrides"]["LD_LIBRARY_PATH"] == (
        f"{config.nvidia_userspace_bundle_root}/rootfs/"
        "usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
    )
    assert "LD_PRELOAD" not in preregistration["environment_overrides"]
    assert "LD_AUDIT" not in preregistration["environment_overrides"]

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


def test_prepare_rejects_nvidia_driver_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(tmp_path, monkeypatch),
        expected_nvidia_driver_version="580.999.99",
    )

    with pytest.raises(FormalCampaignError, match="kernel driver version drifted"):
        prepare_campaign(config, _expected_runs=2)

    assert not config.campaign_root.exists()


def test_execute_rejects_bundle_tamper_before_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    preregistration_sha = prepare_campaign(config, _expected_runs=2)
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered before execution\n", encoding="utf-8"
    )

    with pytest.raises(FormalCampaignError, match="content digest drifted"):
        execute_prepared_campaign(
            config.campaign_root,
            preregistration_sha,
            _expected_runs=2,
        )

    assert not (config.campaign_root / ATTEMPTS_NAME).exists()


def test_post_run_bundle_tamper_is_terminal_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FORMAL_BUNDLE_TAMPER_RUN", "1")

    with pytest.raises(FormalCampaignError, match="content digest drifted"):
        run_campaign(config, _expected_runs=2)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 2
    assert attempts[-1]["run_name"] == "run-001"
    assert attempts[-1]["status"] == "validation_failed"
    assert not (config.campaign_root / "run-002").exists()


def test_post_aggregate_bundle_tamper_blocks_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_FORMAL_AGGREGATOR_BUNDLE_TAMPER", "1")

    with pytest.raises(FormalCampaignError, match="content digest drifted"):
        run_campaign(config, _expected_runs=2)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 6
    assert attempts[-1]["kind"] == "aggregate"
    assert attempts[-1]["status"] == "validation_failed"


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
        "_validate_calibration_nvidia_binding",
        lambda preregistration: {
            field: preregistration[field]
            for field in (
                "nvidia_userspace_bundle_root",
                "expected_nvidia_userspace_bundle_manifest_sha256",
                "expected_nvidia_userspace_bundle_content_digest",
                "expected_nvidia_driver_version",
            )
        },
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
    assert seal["nvidia_userspace_bundle_root"] == str(
        config.nvidia_userspace_bundle_root
    )
    assert seal["nvidia_userspace_bundle_manifest_sha256"] == (
        config.expected_nvidia_userspace_bundle_manifest_sha256
    )
    assert seal["nvidia_userspace_bundle_content_digest"] == (
        config.expected_nvidia_userspace_bundle_content_digest
    )
    assert seal["nvidia_driver_version"] == NVIDIA_DRIVER_VERSION
    with pytest.raises(FormalEvidenceError, match="already exists"):
        publish_formal_bundle_seal(
            config.campaign_root,
            expected_preregistration_sha256=preregistration_sha,
        )


def test_formal_evidence_rejects_nvidia_bundle_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, preregistration_sha = _completed_unsealed_production_campaign(
        tmp_path, monkeypatch
    )
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered before independent replay\n", encoding="utf-8"
    )

    with pytest.raises(FormalEvidenceError, match="content digest drifted"):
        publish_formal_bundle_seal(
            config.campaign_root,
            expected_preregistration_sha256=preregistration_sha,
        )

    assert not (config.campaign_root / SEAL_NAME).exists()


def test_formal_evidence_rejects_nvidia_driver_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, preregistration_sha = _completed_unsealed_production_campaign(
        tmp_path, monkeypatch
    )

    def drift_driver(*args: object, **kwargs: object) -> BundleValidation:
        return replace(
            _fake_bundle_validator(*args, **kwargs),
            kernel_module_version="580.999.99",
        )

    monkeypatch.setattr(formal_evidence, "validate_bundle", drift_driver)
    with pytest.raises(FormalEvidenceError, match="kernel driver version drifted"):
        publish_formal_bundle_seal(
            config.campaign_root,
            expected_preregistration_sha256=preregistration_sha,
        )

    assert not (config.campaign_root / SEAL_NAME).exists()


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
