from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.m2_raw_replay_fixtures import (
    NVIDIA_KERNEL_VERSION,
    RUN_ID,
    _canonical,
    _reseal,
    _sha,
    _write_json,
    _write_jsonl,
    build_raw_run,
)
from tools import m2_raw_replay
from tools.aggregate_m2_calibration import _validate_run
from tools.m2_raw_replay import M2RawReplayError, validate_raw_run


def test_replays_complete_raw_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)

    validated = validate_raw_run(run_dir)

    assert validated.run_id == RUN_ID
    assert validated.observed_max_abs_error == pytest.approx(0.1)
    assert validated.minimum_top1_margin == 1.0


def test_replays_audited_cv2_and_setuptools_import_mutations(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    provenance_path = run_dir / "provenance.json"
    result_path = run_dir / "result.json"
    provenance = json.loads(provenance_path.read_text())
    result = json.loads(result_path.read_text())
    system = provenance["system"]
    boundary = system["runtime_import_boundary"]
    frozen = system["environment"]["LD_LIBRARY_PATH"]
    cv2_prefix = "/fixture/venv/lib/python3.12/site-packages/cv2/../../lib64"
    vendor = "/fixture/venv/lib/python3.12/site-packages/setuptools/_vendor"
    added_entry = {"name": "vendored", "version": "2"}
    effective_packages = sorted(
        [*provenance["dependencies"]["packages"], added_entry],
        key=lambda item: (item["name"], item["version"]),
    )
    effective_sha = _canonical(effective_packages)

    provenance["dependencies"] = {
        "packages": effective_packages,
        "manifest_sha256": effective_sha,
    }
    boundary["loader_environment"].update(
        {
            "observed_after_imports": f"{cv2_prefix}:{frozen}",
            "expected_cv2_prefix": cv2_prefix,
            "prepended_paths": [cv2_prefix],
        }
    )
    boundary["sys_path"].update(
        {
            "after_imports": [*boundary["sys_path"]["before_imports"], vendor],
            "added_paths": [vendor],
            "expected_setuptools_vendor_path": vendor,
        }
    )
    boundary["dependencies"].update(
        {
            "effective_manifest_sha256": effective_sha,
            "effective_count": len(effective_packages),
            "added": [{**added_entry, "origin": vendor}],
            "added_manifest_sha256": _canonical([added_entry]),
        }
    )
    components = provenance["reproducibility_components"]
    components["dependency_manifest_sha256"] = effective_sha
    components["system"] = system
    fingerprint = _canonical(components)
    provenance["reproducibility_fingerprint"] = fingerprint
    result["reproducibility_fingerprint"] = fingerprint
    _write_json(provenance_path, provenance)
    _write_json(result_path, result)
    _reseal(run_dir)

    validate_raw_run(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mapping", "mapped libcuda path differs"),
        ("argv", "argv NVIDIA bundle root differs"),
        ("manifest_capture", "NVIDIA manifest SHA differs"),
        ("manifest_argv", "argv NVIDIA manifest SHA differs"),
        ("digest", "NVIDIA capture content digests differ"),
        ("driver_capture", "NVIDIA expected driver version differs"),
        ("driver_argv", "argv NVIDIA driver version differs"),
        ("derivation", "NVIDIA runtime derivation differs"),
        ("symlink_escape", "symlink target escapes rootfs"),
        ("smi_driver", "system nvidia-smi driver version differs"),
        ("postflight", "postflight NVIDIA content digest differs"),
        ("manifest_postflight", "postflight NVIDIA manifest digest differs"),
    ],
)
def test_rejects_resealed_nvidia_bundle_provenance_tampering(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    if mutation == "mapping":
        provenance["nvidia_driver_userspace"]["libcuda_mapping"]["resolved_path"] = (
            f"/host/libcuda.so.{NVIDIA_KERNEL_VERSION}"
        )
    elif mutation == "argv":
        index = provenance["argv"].index("--nvidia-userspace-bundle-root")
        provenance["argv"][index + 1] = "/fixture/other-bundle"
    elif mutation == "manifest_capture":
        provenance["nvidia_driver_userspace"]["expected_manifest_sha256"] = "f" * 64
    elif mutation == "manifest_argv":
        index = provenance["argv"].index(
            "--expected-nvidia-userspace-bundle-manifest-sha256"
        )
        provenance["argv"][index + 1] = "f" * 64
    elif mutation == "digest":
        provenance["nvidia_driver_userspace"]["content_digest"] = "f" * 64
    elif mutation == "driver_capture":
        provenance["nvidia_driver_userspace"]["expected_driver_version"] = "111.222.333"
    elif mutation == "driver_argv":
        index = provenance["argv"].index("--expected-nvidia-driver-version")
        provenance["argv"][index + 1] = "111.222.333"
    elif mutation == "derivation":
        provenance["nvidia_driver_userspace"]["manifest"]["runtime_derivation"] = (
            "untrusted_derivation"
        )
    elif mutation == "symlink_escape":
        runtime = provenance["nvidia_driver_userspace"]["manifest"]["runtime_tree"]
        symlink = next(
            item
            for item in runtime
            if item["path"] == "usr/lib/x86_64-linux-gnu/libcuda.so.1"
        )
        symlink["target"] = "../../../../escape"
    elif mutation == "smi_driver":
        provenance["system"]["gpu"]["nvidia_smi"] = (
            "GPU-fixture, 111.222.333, 00000000:01:00.0, Fixture GPU, 24564"
        )
    elif mutation == "manifest_postflight":
        provenance["postflight"]["nvidia_driver_userspace_manifest_sha256"] = "f" * 64
    else:
        provenance["postflight"]["nvidia_driver_userspace_content_digest"] = "f" * 64
    _write_json(path, provenance)
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match=message):
        validate_raw_run(run_dir)


def test_accepts_reused_prefetch_slot_with_fresh_generation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    result = json.loads((run_dir / "result.json").read_text())

    assert result["prefetch"]["B1"]["allocated_block_ids"] == [[3]]
    assert result["prefetch"]["B2"]["allocated_block_ids"] == [[3]]
    validate_raw_run(run_dir)


def test_accepts_symlinked_venv_entry_for_captured_runtime_python(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"fixture python\n")
    entry = tmp_path / "venv" / "bin" / "python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(target)
    run_dir = tmp_path / "run"
    build_raw_run(run_dir, python_executable=entry)

    validate_raw_run(run_dir)


def test_calibration_aggregator_invokes_raw_replay(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)

    validated = _validate_run(run_dir)

    assert validated.run_id == RUN_ID
    assert validated.observed_max_abs_error == pytest.approx(0.1)


def test_rejects_semantic_logits_tampering_after_resealing(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "logits_G.npy"
    logits = np.load(path, allow_pickle=False)
    logits[20] = 0.11
    np.save(path, logits, allow_pickle=False)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["measurements"]["G"]["logits_sha256"] = _sha(path)
    _write_json(result_path, result)
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match="max_abs.*raw replay"):
        validate_raw_run(run_dir)


def test_rejects_truncated_vocabulary_logits_after_resealing(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "logits_A1.npy"
    logits = np.load(path, allow_pickle=False)[:1100].copy()
    np.save(path, logits, allow_pickle=False)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["measurements"]["A1"]["logits_sha256"] = _sha(path)
    _write_json(result_path, result)
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match="151936 entries"):
        validate_raw_run(run_dir)


@pytest.mark.parametrize(
    ("capture_name", "replacement"),
    [
        ("implementation", None),
        ("implementation", "A" * 64),
        ("runtime_binaries", None),
        ("runtime_binaries", "A" * 64),
    ],
)
def test_rejects_incomplete_full_hash_capture(
    tmp_path: Path,
    capture_name: str,
    replacement: str | None,
) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    capture = provenance[capture_name]
    list_key = "files" if capture_name == "implementation" else "vllm_extensions"
    content_keys = (
        ("path", "size", "sha256")
        if capture_name == "implementation"
        else ("path", "size", "sha256")
    )
    capture[list_key][0]["sha256"] = replacement
    content = [{key: item[key] for key in content_keys} for item in capture[list_key]]
    if capture_name == "runtime_binaries":
        content = {
            "vllm_extensions": content,
            "python_executable": {
                key: capture["python_executable"][key]
                for key in ("path", "size", "sha256")
            },
        }
    capture["manifest_sha256"] = _canonical(content)
    _write_json(path, provenance)
    _reseal(run_dir)

    with pytest.raises(
        M2RawReplayError, match=f"{capture_name.split('_')[0]} .*sha256"
    ):
        validate_raw_run(run_dir)


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ("executable", "does not resolve to the captured runtime Python"),
        ("runtime_root", "runtime and vLLM Git roots differ"),
        ("vllm_module", "outside the captured vLLM root"),
    ],
)
def test_rejects_runtime_source_binding_mismatch(
    tmp_path: Path,
    binding: str,
    message: str,
) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    if binding == "executable":
        provenance["executable"] = "/fixture/other-python"
    elif binding == "runtime_root":
        provenance["runtime_binaries"]["root"] = "/fixture/other-vllm"
    else:
        provenance["preflight"]["vllm_module"] = "/fixture/other/vllm/__init__.py"
    _write_json(path, provenance)
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match=message):
        validate_raw_run(run_dir)


def test_accepts_identity_keyed_reordered_traces(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    diagnostic_path = run_dir / "diagnostic_transfers.jsonl"
    diagnostic = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    diagnostic[1]["terminal_ns"] = 10_000
    _write_jsonl(diagnostic_path, list(reversed(diagnostic)))
    native_path = run_dir / "native_lifecycle.jsonl"
    native = [json.loads(line) for line in native_path.read_text().splitlines()]
    _write_jsonl(native_path, list(reversed(native)))
    _reseal(run_dir)

    validate_raw_run(run_dir)


def test_rejects_terminal_that_predates_its_own_submission(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "diagnostic_transfers.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    submitted = next(row for row in rows if row["event"] == "submitted")
    terminal = next(
        row
        for row in rows
        if row["event"] == "terminal"
        and row["request_id"] == submitted["request_id"]
        and row["direction"] == submitted["direction"]
    )
    terminal["terminal_ns"] = submitted["submit_ns"] - 1
    _write_jsonl(path, rows)
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match="terminal predates submission"):
        validate_raw_run(run_dir)


def test_rejects_extra_empty_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    (run_dir / "unexpected").mkdir()

    with pytest.raises(M2RawReplayError, match="unexpected directory"):
        validate_raw_run(run_dir)


def test_rejects_special_filesystem_node(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    os.mkfifo(run_dir / "unexpected.fifo")

    with pytest.raises(M2RawReplayError, match="regular file"):
        validate_raw_run(run_dir)


def test_rejects_multiply_linked_evidence_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    os.link(run_dir / "result.json", tmp_path / "external-result-link.json")

    with pytest.raises(M2RawReplayError, match="exactly one hard link"):
        validate_raw_run(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("loader", "observed cv2 loader mutation differs"),
        ("restored", "was not restored before spawn"),
        ("sys_path", "sys.path append relation differs"),
        ("dependency_origin", "added dependency origin differs"),
        ("dependency_removed", "removed a package"),
        ("dependency_count", "differs from provenance"),
        ("python_prefix", "differs from the captured executable entry"),
    ],
)
def test_rejects_resealed_runtime_import_boundary_tampering(
    tmp_path: Path, mutation: str, message: str
) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    boundary = provenance["system"]["runtime_import_boundary"]
    if mutation == "loader":
        boundary["loader_environment"]["expected_cv2_prefix"] = (
            "/fixture/venv/lib/python3.12/site-packages/cv2/../../lib64"
        )
        boundary["loader_environment"]["observed_after_imports"] = (
            "/unknown:/fixture/bundle"
        )
    elif mutation == "restored":
        boundary["loader_environment"]["restored_before_spawn"] = "/unknown"
    elif mutation == "sys_path":
        boundary["sys_path"]["after_imports"].append("/unknown")
    elif mutation == "dependency_origin":
        boundary["sys_path"]["expected_setuptools_vendor_path"] = (
            "/fixture/venv/lib/python3.12/site-packages/setuptools/_vendor"
        )
        boundary["dependencies"]["added"] = [
            {"name": "extra", "version": "1", "origin": "/unknown"}
        ]
    elif mutation == "dependency_removed":
        boundary["dependencies"]["removed"] = [{"name": "fixture", "version": "1"}]
    elif mutation == "python_prefix":
        boundary["python_prefix"] = "/fixture/other-venv"
    else:
        boundary["dependencies"]["effective_count"] += 1
    _write_json(path, provenance)
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match=message):
        validate_raw_run(run_dir)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("execution_ids.json", "prefetch request IDs differ"),
        ("diagnostic_transfers.jsonl", "CPU allocation was not replayed"),
        ("provenance.json", "dependency manifest differs"),
        ("source_state/vllm.untracked.tar", "archive SHA differs"),
    ],
)
def test_rejects_resealed_identity_trace_and_provenance_tampering(
    tmp_path: Path, artifact: str, message: str
) -> None:
    run_dir = tmp_path / "run"
    build_raw_run(run_dir)
    path = run_dir / artifact
    if artifact == "execution_ids.json":
        payload = json.loads(path.read_text())
        payload["transfers"]["B1_H2D"]["engine_request_id"] = "different"
        _write_json(path, payload)
    elif artifact == "diagnostic_transfers.jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[2]["source"]["allocation_generation"] = 9
        rows[3]["source"]["allocation_generation"] = 9
        _write_jsonl(path, rows)
    elif artifact == "provenance.json":
        payload = json.loads(path.read_text())
        payload["dependencies"]["packages"][0]["version"] = "tampered"
        _write_json(path, payload)
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    _reseal(run_dir)

    with pytest.raises(M2RawReplayError, match=message):
        validate_raw_run(run_dir)


def test_missing_numpy_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def fail_numpy(name: str, package: str | None = None) -> Any:
        if name == "numpy":
            raise ImportError("fixture missing NumPy")
        return original(name, package)

    monkeypatch.setattr("tools.m2_raw_replay.importlib.import_module", fail_numpy)
    with pytest.raises(M2RawReplayError, match="NumPy is required"):
        m2_raw_replay._numpy()
