"""Fail-closed tests for sealed M2 CPU component evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_m2_component_evidence import (
    BASE_ENVIRONMENT,
    CHECKSUM_NAME,
    CLAIM_SCOPE,
    DISTRIBUTION_PROBE,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SUITE_SPECS,
    VLLM_RUNTIME_PROBE,
    ComponentEvidenceError,
    _canonical_json,
    _clean_environment,
    _file_entry,
    _git_capture,
    _parse_junit,
    _python_entry,
    _seal_permissions,
    _write_new,
    _write_sha256sums,
    create_component_evidence,
    sha256_file,
    validate_component_evidence,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _make_repo(root: Path, *, dirty: bool) -> Path:
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "DAGKV Test")
    _git(root, "config", "user.email", "dagkv-test@example.invalid")
    (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    module = root / "vllm" / "__init__.py"
    module.parent.mkdir()
    module.write_text('__version__ = "test"\n', encoding="utf-8")
    _git(root, "add", "tracked.py", "vllm/__init__.py")
    _git(root, "commit", "-m", "fixture")
    if dirty:
        (root / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
        (root / "untracked.txt").write_text("captured\n", encoding="utf-8")
    return root


def _junit_bytes(suite_id: str, count: int, *, skipped: bool = False) -> bytes:
    cases = []
    for index in range(count):
        child = "<skipped />" if skipped and index == 0 else ""
        identity = f"test_{index:04d}"
        cases.append(
            f'<testcase classname="{suite_id}" name="{identity}">{child}</testcase>'
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuites><testsuite name="{suite_id}">{"".join(cases)}</testsuite>'
        "</testsuites>\n"
    ).encode()


def _distribution_capture(
    output: Path,
    *,
    label: str,
    python: Path,
) -> dict[str, object]:
    inventory = output / "environment" / f"{label}-distributions.json"
    stderr = output / "environment" / f"{label}-distributions.stderr.txt"
    names = (
        ("dagkv", "numpy", "pytest", "ruff")
        if label == "dagkv"
        else ("numpy", "pytest", "torch", "vllm")
    )
    rows = [
        {
            "name": name,
            "version": "1.0",
            "direct_url_sha256": None,
            "editable": False,
            "source_scheme": None,
        }
        for name in names
    ]
    _write_new(
        inventory,
        _canonical_json(
            {
                "schema_version": "dagkv.python_distribution_inventory.v1",
                "distributions": rows,
            }
        ),
    )
    _write_new(stderr, b"")
    return {
        "argv": [str(python), "-c", DISTRIBUTION_PROBE],
        "environment": _clean_environment(),
        "inventory": _file_entry(inventory, root=output),
        "stderr": _file_entry(stderr, root=output),
    }


def _build_evidence(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    dagkv = _make_repo(tmp_path / "dagkv-source", dirty=False)
    vllm = _make_repo(tmp_path / "vllm-source", dirty=True)
    output = tmp_path / "evidence"
    output.mkdir()
    python = Path(sys.executable)
    dagkv_git = _git_capture(dagkv, output_dir=output, label="dagkv")
    vllm_git = _git_capture(vllm, output_dir=output, label="vllm")
    dagkv_distributions = _distribution_capture(output, label="dagkv", python=python)
    vllm_distributions = _distribution_capture(output, label="vllm", python=python)

    module = (vllm / "vllm" / "__init__.py").resolve()
    runtime_payload = {
        "executable": str(python),
        "pytest": "9.1.1",
        "python": sys.version,
        "torch": "2.11.0",
        "torch_cuda": "13.0",
        "vllm": "test",
        "vllm_file": str(module),
        "vllm_file_sha256": sha256_file(module),
    }
    runtime_path = output / "environment" / "vllm-runtime.json"
    runtime_stderr = output / "environment" / "vllm-runtime.stderr.txt"
    _write_new(runtime_path, _canonical_json(runtime_payload))
    _write_new(runtime_stderr, b"")
    runtime = {
        "argv": [str(python), "-c", VLLM_RUNTIME_PROBE],
        "environment": _clean_environment(),
        "runtime": _file_entry(runtime_path, root=output),
        "stderr": _file_entry(runtime_stderr, root=output),
    }

    suites: list[dict[str, object]] = []
    for spec in SUITE_SPECS:
        stdout = output / "logs" / f"{spec.suite_id}.stdout.txt"
        stderr = output / "logs" / f"{spec.suite_id}.stderr.txt"
        junit = output / "logs" / f"{spec.suite_id}.junit.xml"
        _write_new(stdout, f"{spec.minimum_tests} passed\n".encode())
        _write_new(stderr, b"")
        _write_new(junit, _junit_bytes(spec.suite_id, spec.minimum_tests))
        cwd = dagkv if spec.root_kind == "dagkv" else vllm
        environment = _clean_environment(
            integration_path=spec.needs_integration_path,
            dagkv_root=dagkv,
        )
        suites.append(
            {
                "suite_id": spec.suite_id,
                "cwd": str(cwd),
                "python": str(python),
                "argv": [
                    str(python),
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    f"--junitxml={junit}",
                    *spec.test_paths,
                ],
                "environment": environment,
                "minimum_tests": spec.minimum_tests,
                "started_at_utc": "2026-07-25T00:00:00+00:00",
                "completed_at_utc": "2026-07-25T00:00:01+00:00",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "tests": spec.minimum_tests,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "passed": True,
                "stdout": _file_entry(stdout, root=output),
                "stderr": _file_entry(stderr, root=output),
                "junit": _file_entry(junit, root=output),
            }
        )

    tool = Path(__import__("tools.run_m2_component_evidence", fromlist=["x"]).__file__)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": "2026-07-25T00:00:02+00:00",
        "claim_scope": CLAIM_SCOPE,
        "verification_status": "VERIFIED",
        "determinism": "deterministic",
        "eligible_gate_scope": list(range(1, 8)),
        "evidence_root": str(output),
        "gpu_used": False,
        "item8_accepted": False,
        "m2_accepted": False,
        "dagkv_git": dagkv_git,
        "vllm_git": vllm_git,
        "environment": {
            "tool": {
                "argv": [str(tool), "run"],
                "path": str(tool),
                "sha256": sha256_file(tool),
            },
            "dagkv_python": _python_entry(python, sys.version),
            "vllm_python": _python_entry(python, sys.version),
            "dagkv_distributions": dagkv_distributions,
            "vllm_distributions": vllm_distributions,
            "vllm_runtime": runtime,
            "system": {
                "platform": "test-platform",
                "uname": ["system", "node", "release", "version", "machine", "cpu"],
            },
            "policy": {
                "cuda_visible_devices": "",
                "no_bytecode": True,
                "no_cacheprovider": True,
                "no_gpu": True,
                "no_retry": True,
                "offline": True,
                "timeout_seconds": 900,
            },
        },
        "suites": suites,
        "total_tests": sum(spec.minimum_tests for spec in SUITE_SPECS),
        "all_passed": True,
        "postflight": {
            "dagkv_snapshot_sha256": dagkv_git["snapshot_sha256"],
            "vllm_snapshot_sha256": vllm_git["snapshot_sha256"],
            "source_state_unchanged": True,
        },
    }
    _write_new(output / MANIFEST_NAME, _canonical_json(manifest))
    _write_sha256sums(output)
    validate_component_evidence(
        output,
        require_read_only=False,
        verify_external=True,
    )
    _seal_permissions(output)
    return output, manifest


def _unseal(output: Path) -> None:
    output.chmod(0o755)
    for path in output.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def _rewrite_manifest(output: Path, manifest: dict[str, object]) -> None:
    _unseal(output)
    (output / MANIFEST_NAME).write_bytes(_canonical_json(manifest))
    (output / "SHA256SUMS").unlink()
    _write_sha256sums(output)
    _seal_permissions(output)


def test_suite_contract_is_explicit_and_gpu_free() -> None:
    assert [spec.suite_id for spec in SUITE_SPECS] == [
        "dagkv_core",
        "dagkv_vllm_adapter",
        "vllm_lifecycle_cpu",
    ]
    assert [spec.minimum_tests for spec in SUITE_SPECS] == [269, 13, 345]
    assert BASE_ENVIRONMENT["CUDA_VISIBLE_DEVICES"] == ""
    assert "LD_PRELOAD" not in BASE_ENVIRONMENT
    assert "LD_AUDIT" not in BASE_ENVIRONMENT


def test_distribution_probe_redacts_source_credentials(tmp_path: Path) -> None:
    dist_info = tmp_path / "dagkv_probe_secret-1.2.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: DAGKV-Probe-Secret\nVersion: 1.2\n",
        encoding="utf-8",
    )
    direct_url = {
        "dir_info": {"editable": True},
        "url": "https://user:super-secret-token@example.invalid/repository.git",
    }
    (dist_info / "direct_url.json").write_text(
        json.dumps(direct_url),
        encoding="utf-8",
    )
    environment = dict(BASE_ENVIRONMENT)
    environment["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", DISTRIBUTION_PROBE],
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "super-secret-token" not in result.stdout
    rows = json.loads(result.stdout)
    row = next(item for item in rows if item["name"] == "dagkv-probe-secret")
    encoded = json.dumps(direct_url, sort_keys=True, separators=(",", ":")).encode()
    assert row == {
        "direct_url_sha256": hashlib.sha256(encoded).hexdigest(),
        "editable": True,
        "name": "dagkv-probe-secret",
        "source_scheme": "https",
        "version": "1.2",
    }


def test_failed_creation_has_no_success_seal(tmp_path: Path) -> None:
    output = tmp_path / "failed-evidence"
    missing = tmp_path / "missing-python"
    with pytest.raises(ComponentEvidenceError, match="executable"):
        create_component_evidence(
            output_dir=output,
            dagkv_python=missing,
            vllm_python=missing,
            vllm_root=tmp_path,
            timeout_s=1,
        )
    assert (output / "FAILURE.json").is_file()
    assert not (output / MANIFEST_NAME).exists()
    assert not (output / CHECKSUM_NAME).exists()


def test_junit_parser_rejects_failed_skipped_and_duplicate_cases(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.xml"
    valid.write_bytes(_junit_bytes("suite", 2))
    assert _parse_junit(valid, minimum_tests=2)["tests"] == 2

    skipped = tmp_path / "skipped.xml"
    skipped.write_bytes(_junit_bytes("suite", 2, skipped=True))
    with pytest.raises(ComponentEvidenceError, match="skipped"):
        _parse_junit(skipped, minimum_tests=2)

    failed = tmp_path / "failed.xml"
    failed.write_text(
        '<testsuite><testcase classname="suite" name="test"><failure /></testcase>'
        "</testsuite>",
        encoding="utf-8",
    )
    with pytest.raises(ComponentEvidenceError, match="failed"):
        _parse_junit(failed, minimum_tests=1)

    duplicate = tmp_path / "duplicate.xml"
    duplicate.write_text(
        '<testsuite><testcase classname="suite" name="test" />'
        '<testcase classname="suite" name="test" /></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(ComponentEvidenceError, match="duplicate"):
        _parse_junit(duplicate, minimum_tests=2)


def test_component_evidence_replays_closed_bundle(tmp_path: Path) -> None:
    output, _ = _build_evidence(tmp_path)
    validated = validate_component_evidence(output, verify_external=True)
    assert validated["total_tests"] == 627
    assert validated["m2_accepted"] is False
    python = validated["environment"]["dagkv_python"]
    assert Path(python["path"]) == Path(sys.executable)
    assert Path(python["resolved_path"]) == Path(sys.executable).resolve()


def test_component_evidence_rejects_symlink_root(tmp_path: Path) -> None:
    output, _ = _build_evidence(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)
    with pytest.raises(ComponentEvidenceError, match="symlink"):
        validate_component_evidence(alias)


def test_component_evidence_raw_only_replay_survives_missing_sources(
    tmp_path: Path,
) -> None:
    output, manifest = _build_evidence(tmp_path)
    copied = tmp_path / "relocated-evidence"
    shutil.copytree(output, copied)
    shutil.rmtree(manifest["dagkv_git"]["root"])
    shutil.rmtree(manifest["vllm_git"]["root"])
    validated = validate_component_evidence(copied, verify_external=False)
    assert validated["total_tests"] == 627


def test_component_evidence_rejects_writable_artifact(tmp_path: Path) -> None:
    output, _ = _build_evidence(tmp_path)
    artifact = output / "logs" / "dagkv_core.stdout.txt"
    artifact.chmod(0o644)
    with pytest.raises(ComponentEvidenceError, match="mode"):
        validate_component_evidence(output)


def test_component_evidence_rejects_checksum_tamper(tmp_path: Path) -> None:
    output, _ = _build_evidence(tmp_path)
    _unseal(output)
    artifact = output / "logs" / "dagkv_core.stdout.txt"
    artifact.write_text("tampered\n", encoding="utf-8")
    _seal_permissions(output)
    with pytest.raises(ComponentEvidenceError, match="checksum mismatch"):
        validate_component_evidence(output)


def test_component_evidence_rejects_semantic_junit_tamper(tmp_path: Path) -> None:
    output, manifest = _build_evidence(tmp_path)
    _unseal(output)
    junit = output / "logs" / "dagkv_core.junit.xml"
    junit.write_bytes(_junit_bytes("dagkv_core", 269, skipped=True))
    changed = copy.deepcopy(manifest)
    suite = changed["suites"][0]
    suite["junit"] = _file_entry(junit, root=output)
    suite["skipped"] = 1
    _rewrite_manifest(output, changed)
    with pytest.raises(ComponentEvidenceError, match="skipped"):
        validate_component_evidence(output)


def test_component_evidence_rejects_command_drift(tmp_path: Path) -> None:
    output, manifest = _build_evidence(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["suites"][0]["argv"].append("--maxfail=1")
    _rewrite_manifest(output, changed)
    with pytest.raises(ComponentEvidenceError, match="argv differs"):
        validate_component_evidence(output)


def test_component_evidence_rejects_extra_file(tmp_path: Path) -> None:
    output, _ = _build_evidence(tmp_path)
    _unseal(output)
    (output / "undeclared.txt").write_text("extra\n", encoding="utf-8")
    _seal_permissions(output)
    with pytest.raises(ComponentEvidenceError, match="files differ"):
        validate_component_evidence(output)


def test_manifest_json_is_strict(tmp_path: Path) -> None:
    output, _ = _build_evidence(tmp_path)
    _unseal(output)
    manifest = output / MANIFEST_NAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["m2_accepted"] = False
    raw = json.dumps(payload)
    raw = raw[:-1] + ',"m2_accepted":false}'
    manifest.write_text(raw, encoding="utf-8")
    (output / "SHA256SUMS").unlink()
    _write_sha256sums(output)
    _seal_permissions(output)
    with pytest.raises(ComponentEvidenceError, match="duplicate JSON key"):
        validate_component_evidence(output)
