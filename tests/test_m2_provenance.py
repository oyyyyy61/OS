"""CPU-only regression tests for fail-closed M2 provenance capture."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.run_m2_vllm_abba import (
    M2ValidationError,
    _git_capture,
    _model_capture,
    _verify_git_capture,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def test_git_capture_is_reconstructable_and_detects_drift(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "DAGKV Test")
    _git(repo, "config", "user.email", "dagkv-test@example.invalid")
    (repo / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "fixture")
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("captured\n", encoding="utf-8")

    evidence = tmp_path / "evidence"
    capture = _git_capture(repo, output_dir=evidence, label="fixture")

    assert capture["dirty"] is True
    assert (evidence / capture["tracked_patch"]).is_file()
    assert (evidence / capture["untracked_archive"]).is_file()
    assert _verify_git_capture(capture, label="fixture") == capture["snapshot_sha256"]

    (repo / "untracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(M2ValidationError, match="changed during run"):
        _verify_git_capture(capture, label="fixture")


def _model_fixture(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    index = {
        "metadata": {},
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    git_metadata = root / ".git"
    git_metadata.mkdir()
    (git_metadata / "ignored").write_bytes(b"not-model-content")


def test_model_capture_hashes_index_closed_set_and_excludes_git(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _model_fixture(model)

    capture = _model_capture(model, full_hashes=True)

    paths = {entry["path"] for entry in capture["files"]}
    assert paths == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
    assert all(entry["sha256"] is not None for entry in capture["files"])


def test_model_capture_rejects_unindexed_weight_shard(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _model_fixture(model)
    (model / "unexpected.safetensors").write_bytes(b"extra")

    with pytest.raises(M2ValidationError, match="closed set"):
        _model_capture(model, full_hashes=True)
