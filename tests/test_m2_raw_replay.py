from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.m2_raw_replay_fixtures import (
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
