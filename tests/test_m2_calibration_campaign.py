from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import tools.aggregate_m2_calibration as calibration_aggregator
import tools.run_m2_calibration_campaign as campaign_module
from tools.freeze_m2_tolerance import ToleranceFreezeError, freeze_tolerance
from tools.m2_raw_replay import RawReplayValidation
from tools.nvidia_driver_userspace_bundle import (
    BundleValidation,
    NvidiaUserspaceBundleError,
    RuntimeMapping,
    StatEntry,
)
from tools.run_m2_calibration_campaign import (
    ATTEMPTS_NAME,
    CAMPAIGN_SCHEMA,
    COHORT_NAME,
    COHORT_SCHEMA,
    PREREGISTRATION_NAME,
    PRODUCTION_RUN_COUNT,
    CalibrationCampaignError,
    CampaignConfig,
    _parser,
    execute_prepared_campaign,
    prepare_campaign,
    run_campaign,
)

FAKE_DRIVER_VERSION = "999.888.777"


def _mock_production_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests off the repository while production stays Git-bound."""

    binding = {
        "preparation_git_head": "b" * 40,
        "execution_git_head": "a" * 40,
        "launch_marker_repository_path": (
            "evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json"
        ),
        "launch_marker_sha256": "c" * 64,
    }
    monkeypatch.setattr(campaign_module, "_clean_git_head", lambda **_: "b" * 40)
    monkeypatch.setattr(
        campaign_module,
        "_establish_execution_binding",
        lambda *_: dict(binding),
    )
    monkeypatch.setattr(
        campaign_module, "_revalidate_execution_binding", lambda *_, **__: None
    )


FAKE_RUNNER = r"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def digest(payload):
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--atol", type=float, required=True)
parser.add_argument("--rtol", type=float, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--vllm-root", required=True)
parser.add_argument("--expected-nvidia-driver-version", required=True)
parser.add_argument("--nvidia-userspace-bundle-root", required=True)
parser.add_argument("--expected-nvidia-userspace-bundle-manifest-sha256", required=True)
parser.add_argument(
    "--expected-nvidia-userspace-bundle-content-digest", required=True
)
parser.add_argument("--cpu-bytes", required=True)
parser.add_argument("--timeout-s", required=True)
parser.add_argument("--cuda-device", required=True)
parser.add_argument("--full-provenance", action="store_true")
args = parser.parse_args()

root = args.output_dir.parent
run_name = args.output_dir.name
index = int(run_name.removeprefix("run-"))
prereg = root / "CAMPAIGN_PREREGISTRATION.json"
attempts = root / "ATTEMPTS.jsonl"
assert prereg.is_file()
last = json.loads(attempts.read_text(encoding="utf-8").splitlines()[-1])
assert last["event"] == "submitted"
assert last["run_name"] == run_name
assert args.mode == "calibration"
assert args.atol == 0.125 and args.rtol == 0.0
assert args.full_provenance
assert args.expected_nvidia_driver_version == os.environ["FAKE_DRIVER_VERSION"]
assert args.nvidia_userspace_bundle_root == os.environ["FAKE_BUNDLE_ROOT"]
assert args.expected_nvidia_userspace_bundle_manifest_sha256 == os.environ[
    "FAKE_BUNDLE_MANIFEST_SHA256"
]
assert (
    args.expected_nvidia_userspace_bundle_content_digest
    == os.environ["FAKE_BUNDLE_CONTENT_DIGEST"]
)
assert os.environ["LD_LIBRARY_PATH"] == (
    f"{os.environ['FAKE_BUNDLE_LIBRARY_PATH']}:/usr/local/cuda/lib64"
)
assert "LD_AUDIT" not in os.environ and "LD_PRELOAD" not in os.environ

mode, _, target_text = os.environ.get("FAKE_MODE", "success").partition(":")
target = int(target_text) if target_text else -1
selected = index == target
args.output_dir.mkdir()
if selected and mode == "bundle-tamper":
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered during run\n", encoding="utf-8"
    )
if selected and mode == "timeout":
    (args.output_dir / "partial-evidence.txt").write_text("alive\n")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)
if selected and mode == "nonzero":
    (args.output_dir / "failure-evidence.txt").write_text("exit-7\n")
    raise SystemExit(7)
if selected and mode in {"orphan", "transient-child"}:
    delay = 60 if mode == "orphan" else 0.2
    orphan = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({delay})"],
    )
    (args.output_dir / "orphan.pid").write_text(f"{orphan.pid}\n")

run_id = f"m2-run-{index:03d}"
for name, payload in {
    "diagnostic_transfers.jsonl": b'{"event":"terminal"}\n',
    "execution_ids.json": b"{}\n",
    "native_lifecycle.jsonl": b'{"event":"lookup"}\n',
    "protocol.md": b"# frozen protocol\n",
}.items():
    (args.output_dir / name).write_bytes(payload)

logits_hashes = {}
for phase in ("A1", "G", "B1", "B2", "A2"):
    path = args.output_dir / f"logits_{phase}.npy"
    path.write_bytes(f"fixture-{phase}-{index}".encode())
    logits_hashes[phase] = hashlib.sha256(path.read_bytes()).hexdigest()

measurements = {
    phase: {
        "token_id": 42,
        "top1_margin": 0.5 + index / 100000,
        "num_cached_tokens": 0 if phase in {"A1", "A2"} else 16,
        "logits_file": f"logits_{phase}.npy",
        "logits_sha256": logits_hashes[phase],
    }
    for phase in ("A1", "G", "B1", "B2", "A2")
}
tolerant = (("A1", "G"), ("A1", "B1"), ("A1", "B2"))
exact = (("A1", "A2"), ("G", "B1"), ("G", "B2"), ("B1", "B2"))
comparisons = [
    {
        "left": left,
        "right": right,
        "token_equal": True,
        "allclose": True,
        "max_abs_error": 0.1,
        "max_rel_error": 0.01,
    }
    for left, right in tolerant
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
    for left, right in exact
)
components = {
    "frozen_environment": "drift" if selected and mode == "fingerprint" else "shared"
}
fingerprint = digest(components)
gate = "FAILED" if selected and mode == "gate" else "CALIBRATED_NOT_ACCEPTED"
result = {
    "schema_version": "dagkv.m2.vllm_abba.v3",
    "run_id": run_id,
    "mode": "calibration",
    "gate_status": gate,
    "m2_accepted": False,
    "m2_item8_accepted": False,
    "formal_run_passed": False,
    "within_requested_tolerance": True,
    "tolerance": {"atol": 0.125, "rtol": 0.0},
    "minimum_top1_margin": min(row["top1_margin"] for row in measurements.values()),
    "reproducibility_fingerprint": fingerprint,
    "measurements": measurements,
    "comparisons": comparisons,
}
implementation = os.environ["FAKE_IMPLEMENTATION_SHA"]
if selected and mode == "implementation":
    implementation = hashlib.sha256(b"drifted-implementation").hexdigest()
dagkv = hashlib.sha256(b"dagkv").hexdigest()
vllm = hashlib.sha256(b"vllm").hexdigest()
provenance = {
    "schema_version": "dagkv.m2.vllm_abba.v3",
    "run_id": run_id,
    "mode": "calibration",
    "full_provenance": True,
    "prompt_token_ids": list(range(1000, 1017)),
    "block_size": 16,
    "tolerance": {"atol": 0.125, "rtol": 0.0},
    "reproducibility_components": components,
    "reproducibility_fingerprint": fingerprint,
    "implementation": {"manifest_sha256": implementation},
    "dagkv_git": {"head": "a" * 40, "snapshot_sha256": dagkv},
    "vllm_git": {"snapshot_sha256": vllm},
    "nvidia_driver_userspace": {
        "root": os.environ["FAKE_BUNDLE_ROOT"],
        "expected_driver_version": os.environ["FAKE_DRIVER_VERSION"],
        "expected_content_digest": os.environ["FAKE_BUNDLE_CONTENT_DIGEST"],
        "kernel_module_version": os.environ["FAKE_DRIVER_VERSION"],
        "manifest_sha256": os.environ["FAKE_BUNDLE_MANIFEST_SHA256"],
        "content_digest": os.environ["FAKE_BUNDLE_CONTENT_DIGEST"],
        "runtime": {
            "rootfs": f"{os.environ['FAKE_BUNDLE_ROOT']}/rootfs",
            "library_directory": os.environ["FAKE_BUNDLE_LIBRARY_PATH"],
            "nvidia_smi": f"{os.environ['FAKE_BUNDLE_ROOT']}/rootfs/usr/bin/nvidia-smi",
            "libcuda": (
                f"{os.environ['FAKE_BUNDLE_LIBRARY_PATH']}/"
                f"libcuda.so.{os.environ['FAKE_DRIVER_VERSION']}"
            ),
            "libnvidia_ml": (
                f"{os.environ['FAKE_BUNDLE_LIBRARY_PATH']}/"
                f"libnvidia-ml.so.{os.environ['FAKE_DRIVER_VERSION']}"
            ),
        },
    },
    "postflight": {
        "implementation_manifest_sha256": implementation,
        "dagkv_git_snapshot_sha256": dagkv,
        "vllm_git_snapshot_sha256": vllm,
        "model_file_stats_unchanged": True,
        "runtime_binary_stats_unchanged": True,
    },
}
if selected and mode == "bundle-provenance":
    provenance["nvidia_driver_userspace"]["content_digest"] = hashlib.sha256(
        b"alternate NVIDIA userspace"
    ).hexdigest()
write_json(args.output_dir / "result.json", result)
write_json(args.output_dir / "provenance.json", provenance)
if not (selected and mode == "missing"):
    files = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    rows = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in files
    ]
    (args.output_dir / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n",
        encoding="ascii",
    )
print(f"fake runner completed {run_name}", flush=True)
"""


FAKE_AGGREGATOR = r"""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--campaign-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("FAKE_AGGREGATOR_FAIL") == "1":
    raise SystemExit(9)
prereg_path = args.campaign_dir / "CAMPAIGN_PREREGISTRATION.json"
prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
attempt_lines = (args.campaign_dir / "ATTEMPTS.jsonl").read_bytes().splitlines(
    keepends=True
)
prefix = b"".join(attempt_lines[: prereg["expected_runs"] * 2])
manifest = {
    "schema_version": "dagkv.m2.calibration_cohort.v3",
    "protocol_schema": "dagkv.m2.vllm_abba.v3",
    "campaign_id": prereg["campaign_id"],
    "protocol_sha256": prereg["protocol_sha256"],
    "campaign_preregistration_sha256": hashlib.sha256(
        prereg_path.read_bytes()
    ).hexdigest(),
    "attempt_count": prereg["expected_runs"],
    "selection_rule": prereg["selection_rule"],
    "implementation_manifest_sha256": prereg[
        "expected_implementation_manifest_sha256"
    ],
    "prefix_bytes": len(prefix),
    "prefix_record_count": prereg["expected_runs"] * 2,
    "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
    "pilot_excluded": True,
    "run_count": prereg["expected_runs"],
    "all_passed": True,
    "failures": [],
    "observed_max_abs_error": 0.1,
    "formal_atol": 0.125,
    "formal_rtol": 0.0,
    "reproducibility_fingerprint": os.environ["FAKE_FINGERPRINT"],
    "runs": [],
}
args.output.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if os.environ.get("FAKE_AGGREGATOR_BUNDLE_TAMPER") == "1":
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered during aggregate\n", encoding="utf-8"
    )
print("fake aggregation completed", flush=True)
"""


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_script(path: Path, source: str) -> None:
    path.write_text(source.lstrip(), encoding="utf-8")


def _fake_bundle(
    root: Path,
) -> tuple[Path, str, str, Path, Path]:
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
    driver_version = os.environ["FAKE_DRIVER_VERSION"]
    return BundleValidation(
        manifest={"fixture": "nvidia-bundle"},
        manifest_sha256=manifest_sha256,
        content_digest=content_digest,
        kernel_module_version=driver_version,
        stat_snapshot=(),
        runtime=RuntimeMapping(
            rootfs=runtime_root,
            library_directory=library_path,
            nvidia_smi=runtime_root / "usr/bin/nvidia-smi",
            libcuda=library_path / f"libcuda.so.{driver_version}",
            libnvidia_ml=library_path / f"libnvidia-ml.so.{driver_version}",
        ),
    )


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CampaignConfig:
    runner = tmp_path / "fake_runner.py"
    aggregator = tmp_path / "fake_aggregator.py"
    model = tmp_path / "model"
    vllm_root = tmp_path / "vllm"
    model.mkdir()
    vllm_root.mkdir()
    _write_script(runner, FAKE_RUNNER)
    _write_script(aggregator, FAKE_AGGREGATOR)
    (
        bundle_root,
        bundle_manifest_sha256,
        bundle_content_digest,
        bundle_library_path,
        bundle_content_path,
    ) = _fake_bundle(tmp_path)
    implementation = hashlib.sha256(b"implementation").hexdigest()
    fingerprint = _digest({"frozen_environment": "shared"})
    monkeypatch.setenv("FAKE_MODE", "success")
    monkeypatch.setenv("FAKE_IMPLEMENTATION_SHA", implementation)
    monkeypatch.setenv("FAKE_FINGERPRINT", fingerprint)
    monkeypatch.setenv("FAKE_DRIVER_VERSION", FAKE_DRIVER_VERSION)
    monkeypatch.setenv("FAKE_BUNDLE_ROOT", str(bundle_root))
    monkeypatch.setenv("FAKE_BUNDLE_MANIFEST_SHA256", bundle_manifest_sha256)
    monkeypatch.setenv("FAKE_BUNDLE_CONTENT_DIGEST", bundle_content_digest)
    monkeypatch.setenv("FAKE_BUNDLE_LIBRARY_PATH", str(bundle_library_path))
    monkeypatch.setenv("FAKE_BUNDLE_CONTENT_PATH", str(bundle_content_path))
    monkeypatch.setattr(
        campaign_module,
        "validate_bundle",
        _fake_bundle_validator,
    )

    def validate_fake_raw_run(run_dir: Path) -> RawReplayValidation:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        provenance = json.loads(
            (run_dir / "provenance.json").read_text(encoding="utf-8")
        )
        return RawReplayValidation(
            run_id=result["run_id"],
            mode=result["mode"],
            observed_max_abs_error=max(
                float(comparison["max_abs_error"])
                for comparison in result["comparisons"]
            ),
            minimum_top1_margin=float(result["minimum_top1_margin"]),
            reproducibility_fingerprint=provenance["reproducibility_fingerprint"],
            implementation_manifest_sha256=provenance["implementation"][
                "manifest_sha256"
            ],
        )

    monkeypatch.setattr(
        calibration_aggregator,
        "validate_raw_run",
        validate_fake_raw_run,
    )
    return CampaignConfig(
        campaign_root=tmp_path / "campaign",
        expected_implementation_manifest_sha256=implementation,
        expected_reproducibility_fingerprint=fingerprint,
        nvidia_userspace_bundle_root=bundle_root,
        expected_nvidia_driver_version=FAKE_DRIVER_VERSION,
        expected_nvidia_userspace_bundle_manifest_sha256=(bundle_manifest_sha256),
        expected_nvidia_userspace_bundle_content_digest=bundle_content_digest,
        python_executable=Path(sys.executable),
        runner=runner,
        aggregator=aggregator,
        model=model,
        vllm_root=vllm_root,
        cpu_bytes=1024,
        runner_timeout_s=1.0,
        process_timeout_s=2.0,
        aggregation_timeout_s=2.0,
        terminate_grace_s=0.1,
        kill_wait_s=1.0,
    )


def _attempts(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (root / ATTEMPTS_NAME).read_text(encoding="utf-8").splitlines()
    ]


def test_prepare_then_execute_successfully_binds_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    preregistration_sha256 = prepare_campaign(config, _expected_runs=3)

    assert sorted(path.name for path in config.campaign_root.iterdir()) == [
        PREREGISTRATION_NAME
    ]
    preregistration_path = config.campaign_root / PREREGISTRATION_NAME
    assert hashlib.sha256(preregistration_path.read_bytes()).hexdigest() == (
        preregistration_sha256
    )
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    assert preregistration["schema_version"] == CAMPAIGN_SCHEMA
    assert preregistration["expected_runs"] == 3
    assert preregistration["calibration_attempt_prefix_record_count"] == 6
    assert preregistration["run_names"] == ["run-001", "run-002", "run-003"]
    assert preregistration["runner_command_template"][3] == "<RUN_DIR>"
    assert preregistration["nvidia_userspace_bundle_root"] == str(
        config.nvidia_userspace_bundle_root
    )
    assert preregistration["expected_nvidia_driver_version"] == (
        config.expected_nvidia_driver_version
    )
    assert (
        preregistration["expected_nvidia_userspace_bundle_manifest_sha256"]
        == config.expected_nvidia_userspace_bundle_manifest_sha256
    )
    assert (
        preregistration["expected_nvidia_userspace_bundle_content_digest"]
        == config.expected_nvidia_userspace_bundle_content_digest
    )
    command = preregistration["runner_command_template"]
    driver_option = command.index("--expected-nvidia-driver-version")
    assert command[driver_option + 1] == config.expected_nvidia_driver_version
    bundle_option = command.index("--nvidia-userspace-bundle-root")
    assert command[bundle_option + 1] == str(config.nvidia_userspace_bundle_root)
    manifest_option = command.index(
        "--expected-nvidia-userspace-bundle-manifest-sha256"
    )
    assert command[manifest_option + 1] == (
        config.expected_nvidia_userspace_bundle_manifest_sha256
    )
    content_option = command.index("--expected-nvidia-userspace-bundle-content-digest")
    assert command[content_option + 1] == (
        config.expected_nvidia_userspace_bundle_content_digest
    )
    assert (
        preregistration["environment_overrides"]["LD_LIBRARY_PATH"]
        == f"{config.nvidia_userspace_bundle_root}/rootfs/"
        "usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
    )
    assert not preregistration["environment_overrides"]["LD_LIBRARY_PATH"].endswith(":")
    assert "LD_PRELOAD" not in preregistration["environment_overrides"]
    assert "LD_AUDIT" not in preregistration["environment_overrides"]

    cohort = execute_prepared_campaign(
        config.campaign_root,
        preregistration_sha256,
        _expected_runs=3,
    )

    assert cohort["schema_version"] == COHORT_SCHEMA
    assert cohort["run_count"] == 3
    assert sorted(path.name for path in config.campaign_root.glob("run-*")) == [
        "run-001",
        "run-001.stderr.log",
        "run-001.stdout.log",
        "run-002",
        "run-002.stderr.log",
        "run-002.stdout.log",
        "run-003",
        "run-003.stderr.log",
        "run-003.stdout.log",
    ]
    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 8
    for index in range(3):
        submitted, terminal = attempts[index * 2 : index * 2 + 2]
        assert submitted["event"] == "submitted"
        assert terminal["event"] == "terminal"
        assert terminal["status"] == "passed"
        assert terminal["pid"] > 0
        assert terminal["started_at_utc"]
        assert terminal["ended_at_utc"]
        assert terminal["sequence"] == index + 1
        assert terminal["run_name"] == f"run-{index + 1:03d}"
    aggregate_submitted = attempts[6]
    assert aggregate_submitted["kind"] == "aggregate"
    assert aggregate_submitted["calibration_prefix"]["prefix_record_count"] == 6
    raw = (config.campaign_root / ATTEMPTS_NAME).read_bytes()
    prefix = b"".join(raw.splitlines(keepends=True)[:6])
    assert aggregate_submitted["calibration_prefix"]["prefix_sha256"] == (
        hashlib.sha256(prefix).hexdigest()
    )
    assert (config.campaign_root / COHORT_NAME).is_file()


def test_execute_rejects_a_contaminated_prepared_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    digest = prepare_campaign(config, _expected_runs=2)
    (config.campaign_root / "unexpected.log").write_text("contamination\n")

    with pytest.raises(CalibrationCampaignError, match="only its preregistration"):
        execute_prepared_campaign(
            config.campaign_root,
            digest,
            _expected_runs=2,
        )

    assert not (config.campaign_root / ATTEMPTS_NAME).exists()


def test_execute_rejects_bundle_drift_after_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    digest = prepare_campaign(config, _expected_runs=1)
    Path(os.environ["FAKE_BUNDLE_CONTENT_PATH"]).write_text(
        "tampered after preparation\n",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationCampaignError, match="content digest drifted"):
        execute_prepared_campaign(
            config.campaign_root,
            digest,
            _expected_runs=1,
        )

    assert not (config.campaign_root / ATTEMPTS_NAME).exists()


@pytest.mark.parametrize("variable", ["LD_AUDIT", "LD_PRELOAD"])
def test_preparation_rejects_loader_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv(variable, "/tmp/fixture-loader.so")

    with pytest.raises(CalibrationCampaignError, match="loader injection"):
        prepare_campaign(config, _expected_runs=1)

    assert not config.campaign_root.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "expected_nvidia_driver_version",
            "580.173.03",
            "driver version differs",
        ),
        (
            "expected_nvidia_userspace_bundle_manifest_sha256",
            "0" * 64,
            "bundle validation failed",
        ),
        (
            "expected_nvidia_userspace_bundle_content_digest",
            "0" * 64,
            "content digest drifted",
        ),
    ],
)
def test_preparation_rejects_nvidia_bundle_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    config = replace(_config(tmp_path, monkeypatch), **{field: value})

    with pytest.raises(CalibrationCampaignError, match=message):
        prepare_campaign(config, _expected_runs=1)

    assert not config.campaign_root.exists()


def test_preparation_preserves_virtual_environment_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    command_link = tmp_path / "venv-python"
    command_link.symlink_to(config.python_executable)
    config = CampaignConfig(
        **{
            name: getattr(config, name)
            for name in CampaignConfig.__dataclass_fields__
            if name != "python_executable"
        },
        python_executable=command_link,
    )

    preregistration_sha256 = prepare_campaign(config, _expected_runs=1)
    preregistration = json.loads(
        (config.campaign_root / PREREGISTRATION_NAME).read_text(encoding="utf-8")
    )

    assert preregistration["python_executable"]["path"] == str(command_link)
    execute_prepared_campaign(
        config.campaign_root,
        preregistration_sha256,
        _expected_runs=1,
    )


def test_nonzero_run_stops_without_retry_and_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "nonzero:2")

    with pytest.raises(CalibrationCampaignError, match="exited 7"):
        run_campaign(config, _expected_runs=3)

    attempts = _attempts(config.campaign_root)
    assert [row["status"] for row in attempts if row["event"] == "terminal"] == [
        "passed",
        "process_failed",
    ]
    assert not (config.campaign_root / "run-003").exists()
    assert (config.campaign_root / "run-002" / "failure-evidence.txt").is_file()
    assert not (config.campaign_root / COHORT_NAME).exists()


@pytest.mark.parametrize(
    "mode",
    [
        "missing:2",
        "gate:2",
        "fingerprint:2",
        "implementation:2",
        "bundle-provenance:2",
    ],
)
def test_artifact_or_fingerprint_drift_stops_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MODE", mode)

    with pytest.raises(CalibrationCampaignError):
        run_campaign(config, _expected_runs=3)

    attempts = _attempts(config.campaign_root)
    assert attempts[-1]["status"] == "validation_failed"
    assert attempts[-1]["run_name"] == "run-002"
    assert not (config.campaign_root / "run-003").exists()
    assert not (config.campaign_root / COHORT_NAME).exists()


def test_run_time_bundle_tampering_stops_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "bundle-tamper:1")

    with pytest.raises(CalibrationCampaignError, match="content digest drifted"):
        run_campaign(config, _expected_runs=2)

    attempts = _attempts(config.campaign_root)
    assert [row["event"] for row in attempts] == ["submitted", "terminal"]
    assert attempts[-1]["status"] == "validation_failed"
    assert "post-run bundle validation failed" in attempts[-1]["error"]
    assert not (config.campaign_root / "run-002").exists()


def test_run_time_bundle_identity_drift_is_detected_after_byte_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    original_validator = campaign_module.validate_bundle

    def identity_drifting_validator(
        *args: object, **kwargs: object
    ) -> BundleValidation:
        validation = original_validator(*args, **kwargs)
        if (config.campaign_root / "run-001" / "result.json").is_file():
            return replace(
                validation,
                stat_snapshot=(
                    StatEntry(
                        path=".",
                        entry_type="directory",
                        device=1,
                        inode=2,
                        mode=0o40555,
                        link_count=1,
                        size=0,
                        mtime_ns=3,
                        ctime_ns=4,
                    ),
                ),
            )
        return validation

    monkeypatch.setattr(
        campaign_module,
        "validate_bundle",
        identity_drifting_validator,
    )

    with pytest.raises(CalibrationCampaignError, match="filesystem identity changed"):
        run_campaign(config, _expected_runs=2)

    attempts = _attempts(config.campaign_root)
    assert attempts[-1]["status"] == "validation_failed"
    assert not (config.campaign_root / "run-002").exists()


def test_timeout_escalates_to_sigkill_and_preserves_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config = CampaignConfig(
        **{
            name: getattr(config, name)
            for name in CampaignConfig.__dataclass_fields__
            if name != "process_timeout_s"
        },
        process_timeout_s=0.2,
    )
    monkeypatch.setenv("FAKE_MODE", "timeout:1")

    with pytest.raises(CalibrationCampaignError, match="timed out"):
        run_campaign(config, _expected_runs=2)

    terminal = _attempts(config.campaign_root)[-1]
    assert terminal["status"] == "timed_out"
    assert terminal["sigterm_sent"] is True
    assert terminal["sigkill_sent"] is True
    assert terminal["pid"] > 0
    assert (config.campaign_root / "run-001" / "partial-evidence.txt").is_file()


def test_external_sigterm_terminalizes_attempt_and_cleans_child_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config = CampaignConfig(
        **{
            name: getattr(config, name)
            for name in CampaignConfig.__dataclass_fields__
            if name != "process_timeout_s"
        },
        process_timeout_s=10.0,
    )
    monkeypatch.setenv("FAKE_MODE", "timeout:1")
    timer = threading.Timer(0.2, os.kill, args=(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(CalibrationCampaignError, match="interrupted"):
            run_campaign(config, _expected_runs=1)
    finally:
        timer.cancel()

    terminal = _attempts(config.campaign_root)[-1]
    assert terminal["status"] == "orchestrator_interrupted"
    assert terminal["pid"] > 0
    assert terminal["sigterm_sent"] is True
    assert terminal["sigkill_sent"] is True
    with pytest.raises(ProcessLookupError):
        os.killpg(terminal["pid"], 0)


def test_successful_parent_with_orphaned_child_invalidates_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "orphan:1")

    with pytest.raises(CalibrationCampaignError, match="live process group"):
        run_campaign(config, _expected_runs=1)

    terminal = _attempts(config.campaign_root)[-1]
    assert terminal["status"] == "lingering_process_group"
    assert terminal["sigterm_sent"] is True
    orphan_pid = int(
        (config.campaign_root / "run-001" / "orphan.pid").read_text().strip()
    )
    with pytest.raises(ProcessLookupError):
        os.kill(orphan_pid, 0)


def test_short_lived_descendant_gets_a_natural_shutdown_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "transient-child:1")

    cohort = run_campaign(config, _expected_runs=1)

    assert cohort["run_count"] == 1
    run_terminal = _attempts(config.campaign_root)[1]
    assert run_terminal["status"] == "passed"
    assert run_terminal["sigterm_sent"] is False
    assert run_terminal["sigkill_sent"] is False


def test_aggregator_failure_is_recorded_after_the_sealed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_AGGREGATOR_FAIL", "1")

    with pytest.raises(CalibrationCampaignError, match="aggregation failed"):
        run_campaign(config, _expected_runs=2)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 6
    assert all(row["status"] == "passed" for row in (attempts[1], attempts[3]))
    assert attempts[4]["kind"] == "aggregate"
    assert attempts[5]["status"] == "process_failed"
    assert attempts[5]["exit_code"] == 9
    assert not (config.campaign_root / COHORT_NAME).exists()


def test_aggregate_time_bundle_tampering_invalidates_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_AGGREGATOR_BUNDLE_TAMPER", "1")

    with pytest.raises(CalibrationCampaignError, match="content digest drifted"):
        run_campaign(config, _expected_runs=1)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 4
    assert attempts[-1]["kind"] == "aggregate"
    assert attempts[-1]["status"] == "validation_failed"
    assert "post-aggregate bundle validation failed" in attempts[-1]["error"]
    assert (config.campaign_root / COHORT_NAME).is_file()


def test_production_replays_candidate_before_passed_terminal_and_self_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    _mock_production_execution(monkeypatch)
    observed: list[str] = []

    def validate_candidate(path: Path, **_: object) -> None:
        attempts = _attempts(path.parent)
        assert len(attempts) == 119
        assert attempts[-1]["kind"] == "aggregate"
        assert attempts[-1]["event"] == "submitted"
        observed.append("candidate")

    def validate_bundle(path: Path, **_: object) -> None:
        attempts = _attempts(path.parent)
        assert len(attempts) == 120
        assert attempts[-1]["kind"] == "aggregate"
        assert attempts[-1]["event"] == "terminal"
        assert attempts[-1]["status"] == "passed"
        observed.append("bundle")

    monkeypatch.setattr(
        campaign_module,
        "_revalidate_production_candidate",
        validate_candidate,
    )
    monkeypatch.setattr(
        campaign_module,
        "_revalidate_production_bundle",
        validate_bundle,
    )

    cohort = run_campaign(config)

    assert cohort["run_count"] == 59
    assert observed == ["candidate", "bundle"]


def test_production_candidate_replay_failure_records_validation_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    _mock_production_execution(monkeypatch)
    final_called = False

    def reject_candidate(path: Path, **_: object) -> None:
        attempts = _attempts(path.parent)
        assert len(attempts) == 119
        raise CalibrationCampaignError("candidate replay rejected")

    def unexpected_final(path: Path, **_: object) -> None:
        nonlocal final_called
        del path
        final_called = True

    monkeypatch.setattr(
        campaign_module,
        "_revalidate_production_candidate",
        reject_candidate,
    )
    monkeypatch.setattr(
        campaign_module,
        "_revalidate_production_bundle",
        unexpected_final,
    )

    with pytest.raises(CalibrationCampaignError, match="candidate replay rejected"):
        run_campaign(config)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 120
    assert attempts[-1]["kind"] == "aggregate"
    assert attempts[-1]["status"] == "validation_failed"
    assert final_called is False


def test_production_postcommit_self_check_failure_cannot_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    _mock_production_execution(monkeypatch)

    def validate_candidate(path: Path, **_: object) -> None:
        assert len(_attempts(path.parent)) == 119

    def reject_final(path: Path, **_: object) -> None:
        attempts = _attempts(path.parent)
        assert len(attempts) == 120
        assert attempts[-1]["status"] == "passed"
        aggregate_stdout = path.parent / "aggregate.stdout.log"
        aggregate_stdout.write_bytes(aggregate_stdout.read_bytes() + b"tampered\n")
        raise CalibrationCampaignError("postcommit bundle replay rejected")

    monkeypatch.setattr(
        campaign_module,
        "_revalidate_production_candidate",
        validate_candidate,
    )
    monkeypatch.setattr(
        campaign_module,
        "_revalidate_production_bundle",
        reject_final,
    )

    with pytest.raises(CalibrationCampaignError, match="postcommit bundle"):
        run_campaign(config)

    attempts = _attempts(config.campaign_root)
    assert len(attempts) == 120
    assert attempts[-1]["kind"] == "aggregate"
    assert attempts[-1]["status"] == "passed"
    frozen_output = tmp_path / "M2_FROZEN_TOLERANCE.json"
    with pytest.raises(ToleranceFreezeError):
        freeze_tolerance(config.campaign_root / COHORT_NAME, frozen_output)
    assert not frozen_output.exists()


def test_production_cli_has_no_run_count_override() -> None:
    assert PRODUCTION_RUN_COUNT == 59
    destinations = {action.dest for action in _parser()._actions}
    assert "expected_runs" not in destinations
    assert {"prepare_only", "execute_prepared"}.issubset(destinations)
    assert {
        "nvidia_userspace_bundle_root",
        "expected_nvidia_driver_version",
        "expected_nvidia_userspace_bundle_manifest_sha256",
        "expected_nvidia_userspace_bundle_content_digest",
    }.issubset(destinations)
    with pytest.raises(SystemExit):
        _parser().parse_args(["--campaign-root", "/tmp/campaign"])


def test_production_preparation_rejects_dirty_git_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git_bytes(*arguments: str) -> bytes:
        if arguments == ("rev-parse", "--show-toplevel"):
            return f"{campaign_module.REPO_ROOT}\n".encode()
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b" M tools/run_m2_calibration_campaign.py\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(campaign_module, "_git_bytes", git_bytes)
    with pytest.raises(CalibrationCampaignError, match="worktree must be clean"):
        campaign_module._clean_git_head(phase="calibration preparation")


def test_execution_binding_rejects_non_direct_marker_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = "b" * 40
    execution = "a" * 40
    preregistration = {
        "campaign_id": "m2-calibration-" + "1" * 32,
        "campaign_root": str(tmp_path / "campaign"),
        "preparation_git_head": preparation,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(campaign_module, "_clean_git_head", lambda **_: execution)
    monkeypatch.setattr(
        campaign_module,
        "_git_bytes",
        lambda *arguments: (
            f"{execution} {'c' * 40}\n".encode()
            if arguments[:3] == ("rev-list", "--parents", "-n")
            else b""
        ),
    )
    with pytest.raises(CalibrationCampaignError, match="direct single-parent child"):
        campaign_module._establish_execution_binding(preregistration, "d" * 64)


def test_campaign_execution_lock_rejects_concurrent_launcher(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    with (
        campaign_module._campaign_execution_lock(root),
        pytest.raises(CalibrationCampaignError, match="already executing"),
        campaign_module._campaign_execution_lock(root),
    ):
        pass


def test_execute_prepared_campaign_rejects_symlink_root_before_resolution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "campaign-target"
    target.mkdir()
    alias = tmp_path / "campaign-alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(CalibrationCampaignError, match="cannot be a symlink"):
        execute_prepared_campaign(alias, "a" * 64, _expected_runs=1)
