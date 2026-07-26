#!/usr/bin/env python3
"""Create and replay a fail-closed M3/C1 component evidence bundle."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "dagkv.m3.c1_component_evidence.v1"
MANIFEST_NAME = "M3_C1_COMPONENT_EVIDENCE.json"
CHECKSUM_NAME = "SHA256SUMS"
PUBLICATION_LOCK_SUFFIX = ".m3-c1-component-publication.lock"
LOCK_PREPARING = b"PREPARING\n"
LOCK_PUBLISHED = b"PUBLISHED\n"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_TEST_PATH = "tests/test_c1_shared_leases.py"
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TEXT_BYTES = 64 * 1024 * 1024
MAX_JUNIT_BYTES = 32 * 1024 * 1024

CLAIM_SCOPE = (
    "M3/C1 mathematical and runtime component correctness only. No trace "
    "calibration, policy benefit, latency, throughput, GPU, or novelty claim."
)

SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/dagkv/__init__.py",
    "src/dagkv/c1_leases.py",
    "src/dagkv/domain.py",
    "src/dagkv/ledger.py",
    "src/dagkv/orchestrator.py",
    PROBABILITY_TEST_PATH,
    "tests/test_m3_c1_component_evidence.py",
    "tools/run_m3_c1_component_evidence.py",
    "research/ARCHITECTURE.md",
    "research/REFERENCES.md",
    "research/STAGE_GATES.md",
    "research/imported/RELATED_WORK_MATRIX.md",
    "research/protocols/M3_C1_SHARED_LEASE_PROTOCOL.md",
)

EXPECTED_C1_TESTS = frozenset(
    {
        "test_mutually_exclusive_branches_preserve_joint_union",
        "test_correlated_fanout_counts_one_physical_epoch",
        "test_repeated_reuses_are_separate_from_first_readmission",
        "test_independent_workflows_use_product_only_between_groups",
        "test_total_variation_radius_produces_exact_sound_bounds",
        "test_total_variation_bounds_match_exhaustive_mass_reallocation",
        "test_independent_group_robust_bounds_cover_exhaustive_products",
        "test_priority_modes_are_independently_selectable",
        "test_oracle_forecast_is_excluded_from_online_scoring",
        "test_forecast_rejects_stale_or_cross_owner_scope",
        "test_forecast_rejects_ineligible_node_and_predated_owner",
        "test_probability_and_dependence_identities_fail_closed",
        "test_claim_or_epoch_cannot_cross_independent_groups",
        "test_empty_outcomes_form_a_zero_profile",
        "test_orchestrator_snapshot_is_detached_scoped_and_state_bound",
        "test_released_retention_owner_disappears_from_policy_snapshot",
    }
)

BASE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


class C1EvidenceError(RuntimeError):
    """Raised when a C1 component bundle cannot be created or replayed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C1EvidenceError(message)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            path.unlink()
        raise


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise C1EvidenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise C1EvidenceError(f"non-finite JSON value: {value}")


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    raw = path.read_bytes()
    require(len(raw) <= MAX_TEXT_BYTES, f"{label} is too large")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C1EvidenceError(f"invalid {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, raw


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise C1EvidenceError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result


def _git_text(*args: str) -> str:
    return _git(*args).stdout.decode().strip()


def _file_entry(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_command(
    command_id: str,
    argv_template: Sequence[str],
    *,
    output_root: Path,
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    argv = [item.replace("{output_root}", str(output_root)) for item in argv_template]
    started_at = _utc_now()
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=dict(BASE_ENVIRONMENT),
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            argv,
            124,
            stdout=exc.stdout or b"",
            stderr=exc.stderr or b"",
        )
        timed_out = True
    duration = time.monotonic() - started
    require(len(result.stdout) <= MAX_TEXT_BYTES, f"{command_id} stdout is too large")
    require(len(result.stderr) <= MAX_TEXT_BYTES, f"{command_id} stderr is too large")
    stdout_path = output_root / "logs" / f"{command_id}.stdout.txt"
    stderr_path = output_root / "logs" / f"{command_id}.stderr.txt"
    _write_new(stdout_path, result.stdout)
    _write_new(stderr_path, result.stderr)
    return {
        "command_id": command_id,
        "argv_template": list(argv_template),
        "cwd": str(cwd),
        "environment": dict(BASE_ENVIRONMENT),
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "duration_seconds": duration,
        "exit_code": result.returncode,
        "timed_out": timed_out,
        "stdout": _file_entry(stdout_path, root=output_root),
        "stderr": _file_entry(stderr_path, root=output_root),
    }


def _parse_junit(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing JUnit: {path}")
    raw = path.read_bytes()
    require(len(raw) <= MAX_JUNIT_BYTES, "JUnit is too large")
    require(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "unsafe JUnit XML")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise C1EvidenceError(f"invalid JUnit XML: {exc}") from exc
    cases = root.findall(".//testcase")
    identities: list[str] = []
    for case in cases:
        classname = case.attrib.get("classname")
        name = case.attrib.get("name")
        require(bool(classname and name), "JUnit testcase identity is incomplete")
        identities.append(f"{classname}::{name}")
    require(len(identities) == len(set(identities)), "duplicate JUnit testcase")
    failures = len(root.findall(".//failure"))
    errors = len(root.findall(".//error"))
    skipped = len(root.findall(".//skipped"))
    return {
        "tests": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "identities": sorted(identities),
    }


def _source_capture(output_root: Path, head: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing source: {relative}")
        blob = _git("show", f"{head}:{relative}").stdout
        require(
            blob == path.read_bytes(), f"working source differs from HEAD: {relative}"
        )
        entries.append(
            {"path": relative, "size": len(blob), "sha256": _sha256_bytes(blob)}
        )
    archive = _git("archive", "--format=tar", head, "--", *SOURCE_PATHS).stdout
    archive_path = output_root / "source" / "c1-source.tar"
    _write_new(archive_path, archive)
    return {
        "paths": list(SOURCE_PATHS),
        "entries": entries,
        "archive": _file_entry(archive_path, root=output_root),
    }


def _historical_m2_replay(
    output_root: Path,
    *,
    python: Path,
    current_branch: str,
    current_head: str,
    acceptance: Path,
    expected_acceptance_sha256: str,
    accepted_head: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    require(current_branch == "main", "historical replay requires branch main")
    require(not _git_text("status", "--porcelain=v1"), "repository must be clean")
    switch_to = _git("switch", "--detach", accepted_head, check=False)
    require(switch_to.returncode == 0, "cannot enter accepted M2 HEAD")
    replay: dict[str, Any] | None = None
    restore: subprocess.CompletedProcess[bytes] | None = None
    try:
        replay = _run_command(
            "m2_historical_replay",
            (
                str(python),
                "tools/m2_aggregate_acceptance.py",
                "validate",
                str(acceptance),
                "--expected-acceptance-sha256",
                expected_acceptance_sha256,
            ),
            output_root=output_root,
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
        )
    finally:
        restore = _git("switch", current_branch, check=False)
    require(restore.returncode == 0, "cannot restore branch after M2 replay")
    require(
        _git_text("rev-parse", "HEAD") == current_head, "HEAD changed during replay"
    )
    require(not _git_text("status", "--porcelain=v1"), "replay dirtied repository")
    assert replay is not None
    require(replay["exit_code"] == 0 and not replay["timed_out"], "M2 replay failed")
    replay_stdout = (output_root / replay["stdout"]["path"]).read_text()
    require(
        f"M2 aggregate replay passed: M2_ACCEPTED_CORRECTNESS_ONLY "
        f"sha256={expected_acceptance_sha256}" in replay_stdout,
        "M2 replay success terminal is missing",
    )
    switch_path = output_root / "logs" / "m2_switch.stdout.txt"
    switch_error_path = output_root / "logs" / "m2_switch.stderr.txt"
    restore_path = output_root / "logs" / "m2_restore.stdout.txt"
    restore_error_path = output_root / "logs" / "m2_restore.stderr.txt"
    _write_new(switch_path, switch_to.stdout)
    _write_new(switch_error_path, switch_to.stderr)
    _write_new(restore_path, restore.stdout)
    _write_new(restore_error_path, restore.stderr)
    return {
        "accepted_repository_head": accepted_head,
        "acceptance_path": str(acceptance),
        "acceptance_sha256": expected_acceptance_sha256,
        "replay": replay,
        "switch_stdout": _file_entry(switch_path, root=output_root),
        "switch_stderr": _file_entry(switch_error_path, root=output_root),
        "restore_stdout": _file_entry(restore_path, root=output_root),
        "restore_stderr": _file_entry(restore_error_path, root=output_root),
    }


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_NAME
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths
    ]
    _write_new(root / CHECKSUM_NAME, ("\n".join(lines) + "\n").encode())


def _validate_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / CHECKSUM_NAME
    require(
        checksum_path.is_file() and not checksum_path.is_symlink(), "missing checksums"
    )
    entries: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        parts = line.split("  ", maxsplit=1)
        require(len(parts) == 2, "malformed checksum row")
        digest, relative = parts
        require(relative not in entries, "duplicate checksum path")
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "invalid checksum digest",
        )
        path = root / relative
        require(
            path.resolve().is_relative_to(root.resolve()), "checksum path escapes root"
        )
        require(
            path.is_file() and not path.is_symlink(), f"missing sealed file: {relative}"
        )
        require(_sha256_file(path) == digest, f"checksum mismatch: {relative}")
        entries[relative] = digest
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_NAME
    }
    require(set(entries) == observed, "checksum inventory differs from evidence files")
    return entries


def _validate_file_entry(root: Path, value: object, *, label: str) -> Path:
    require(isinstance(value, dict), f"{label} entry must be an object")
    require(set(value) == {"path", "size", "sha256"}, f"{label} fields differ")
    relative = value["path"]
    require(isinstance(relative, str) and relative, f"{label} path is invalid")
    require(
        type(value["size"]) is int and value["size"] >= 0,
        f"{label} size is invalid",
    )
    digest = value["sha256"]
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} SHA-256 is invalid",
    )
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), f"{label} escapes root")
    require(path.is_file() and not path.is_symlink(), f"{label} file is missing")
    require(path.stat().st_size == value["size"], f"{label} size differs")
    require(_sha256_file(path) == value["sha256"], f"{label} SHA-256 differs")
    return path


def _validate_command(
    root: Path,
    command: object,
    *,
    expected_id: str,
    expected_template: Sequence[str],
) -> None:
    require(isinstance(command, dict), "command record must be an object")
    require(
        set(command)
        == {
            "command_id",
            "argv_template",
            "cwd",
            "environment",
            "started_at_utc",
            "ended_at_utc",
            "duration_seconds",
            "exit_code",
            "timed_out",
            "stdout",
            "stderr",
        },
        f"{expected_id} command fields differ",
    )
    require(command["command_id"] == expected_id, "command identity differs")
    require(
        command["argv_template"] == list(expected_template),
        f"{expected_id} argv differs",
    )
    require(command["cwd"] == str(REPO_ROOT), f"{expected_id} cwd differs")
    require(
        command["environment"] == BASE_ENVIRONMENT,
        f"{expected_id} environment differs",
    )
    for field in ("started_at_utc", "ended_at_utc"):
        try:
            parsed = datetime.fromisoformat(command[field])
        except (TypeError, ValueError) as exc:
            raise C1EvidenceError(f"{expected_id} timestamp is invalid") from exc
        require(parsed.tzinfo is not None, f"{expected_id} timestamp lacks timezone")
    duration = command["duration_seconds"]
    require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration >= 0,
        f"{expected_id} duration is invalid",
    )
    require(
        command["exit_code"] == 0 and command["timed_out"] is False,
        f"{expected_id} command failed",
    )
    _validate_file_entry(root, command["stdout"], label=f"{expected_id} stdout")
    _validate_file_entry(root, command["stderr"], label=f"{expected_id} stderr")


def validate_bundle(
    root: Path,
    *,
    expected_manifest_sha256: str | None,
    expected_checksums_sha256: str | None,
    require_sealed: bool = True,
) -> tuple[dict[str, Any], str, str]:
    root = root.expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), "evidence root is invalid")
    manifest, raw = _read_json(root / MANIFEST_NAME, label="C1 manifest")
    manifest_sha = _sha256_bytes(raw)
    checksums_sha = _sha256_file(root / CHECKSUM_NAME)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "manifest SHA-256 differs")
    if expected_checksums_sha256 is not None:
        require(checksums_sha == expected_checksums_sha256, "checksums SHA-256 differs")
    _validate_checksums(root)
    require(
        set(manifest)
        == {
            "schema_version",
            "created_at_utc",
            "status",
            "claim_scope",
            "repository",
            "python",
            "source",
            "commands",
            "junit",
            "m2_historical_replay",
            "gpu_used",
            "performance_claims_supported",
            "all_passed",
        },
        "manifest fields differ",
    )
    require(manifest["schema_version"] == SCHEMA_VERSION, "schema version differs")
    require(manifest["status"] == "C1_A_COMPONENT_VERIFIED", "status differs")
    require(manifest["claim_scope"] == CLAIM_SCOPE, "claim scope differs")
    require(manifest["gpu_used"] is False, "component evidence claims GPU use")
    require(
        manifest["performance_claims_supported"] is False,
        "component evidence claims performance support",
    )
    require(manifest["all_passed"] is True, "component evidence did not pass")
    try:
        created = datetime.fromisoformat(manifest["created_at_utc"])
    except (TypeError, ValueError) as exc:
        raise C1EvidenceError("manifest timestamp is invalid") from exc
    require(created.tzinfo is not None, "manifest timestamp lacks timezone")

    repository = manifest["repository"]
    require(isinstance(repository, dict), "repository binding is invalid")
    require(
        set(repository) == {"root", "head", "branch", "tree", "clean"},
        "repository fields differ",
    )
    require(Path(repository["root"]).resolve() == REPO_ROOT, "repository root differs")
    head = repository["head"]
    require(
        repository["branch"] == "main" and repository["clean"] is True,
        "repository state differs",
    )
    require(
        _git("cat-file", "-e", f"{head}^{{commit}}", check=False).returncode == 0,
        "repository commit is absent",
    )
    require(
        _git_text("show", "-s", "--format=%T", head) == repository["tree"],
        "repository tree differs",
    )

    python_binding = manifest["python"]
    require(
        isinstance(python_binding, dict)
        and set(python_binding) == {"path", "sha256", "version"},
        "Python binding fields differ",
    )
    python_path = Path(python_binding["path"])
    require(
        python_path.is_file() and not python_path.is_symlink(),
        "bound Python executable is missing",
    )
    require(
        _sha256_file(python_path) == python_binding["sha256"],
        "bound Python executable differs",
    )
    require(
        isinstance(python_binding["version"], str)
        and python_binding["version"].startswith("Python 3.12"),
        "bound Python version differs",
    )

    source = manifest["source"]
    require(isinstance(source, dict), "source capture is invalid")
    require(set(source) == {"paths", "entries", "archive"}, "source fields differ")
    require(source["paths"] == list(SOURCE_PATHS), "source paths differ")
    entries = source["entries"]
    require(
        isinstance(entries, list) and len(entries) == len(SOURCE_PATHS),
        "source entries differ",
    )
    for relative, entry in zip(SOURCE_PATHS, entries, strict=True):
        require(
            isinstance(entry, dict)
            and set(entry) == {"path", "size", "sha256"}
            and entry["path"] == relative,
            "source entry fields differ",
        )
        blob = _git("show", f"{head}:{relative}").stdout
        require(
            len(blob) == entry["size"] and _sha256_bytes(blob) == entry["sha256"],
            f"source Git blob differs: {relative}",
        )
    archive_path = _validate_file_entry(root, source["archive"], label="source archive")
    expected_archive = _git("archive", "--format=tar", head, "--", *SOURCE_PATHS).stdout
    require(archive_path.read_bytes() == expected_archive, "source archive differs")

    commands = manifest["commands"]
    require(isinstance(commands, list) and len(commands) == 4, "command set differs")
    command_specs = (
        (
            "c1_focused",
            (
                str(python_path),
                "-m",
                "pytest",
                "-q",
                "--junitxml={output_root}/logs/c1-focused.junit.xml",
                PROBABILITY_TEST_PATH,
            ),
        ),
        (
            "repository_full",
            (
                str(python_path),
                "-m",
                "pytest",
                "-q",
                "--junitxml={output_root}/logs/repository-full.junit.xml",
            ),
        ),
        ("ruff_check", (str(python_path), "-m", "ruff", "check", ".")),
        (
            "ruff_format_check",
            (str(python_path), "-m", "ruff", "format", "--check", "."),
        ),
    )
    for command, (expected_id, expected_template) in zip(
        commands,
        command_specs,
        strict=True,
    ):
        _validate_command(
            root,
            command,
            expected_id=expected_id,
            expected_template=expected_template,
        )

    junit = manifest["junit"]
    require(
        isinstance(junit, dict) and set(junit) == {"focused", "full"},
        "JUnit fields differ",
    )
    for label in ("focused", "full"):
        require(
            isinstance(junit[label], dict) and set(junit[label]) == {"file", "summary"},
            f"{label} JUnit record fields differ",
        )
    focused_path = _validate_file_entry(
        root, junit["focused"]["file"], label="focused JUnit"
    )
    full_path = _validate_file_entry(root, junit["full"]["file"], label="full JUnit")
    focused = _parse_junit(focused_path)
    full = _parse_junit(full_path)
    require(focused == junit["focused"]["summary"], "focused JUnit summary differs")
    require(full == junit["full"]["summary"], "full JUnit summary differs")
    require(focused["tests"] == len(EXPECTED_C1_TESTS), "focused test count differs")
    focused_names = {
        identity.rsplit("::", maxsplit=1)[-1] for identity in focused["identities"]
    }
    require(focused_names == EXPECTED_C1_TESTS, "focused testcase identities differ")
    require(full["tests"] >= 331, "full regression test count is below M3 baseline")
    require(
        set(focused["identities"]).issubset(set(full["identities"])),
        "full regression omits a focused C1 testcase",
    )
    require(
        all(
            summary[key] == 0
            for summary in (focused, full)
            for key in ("failures", "errors", "skipped")
        ),
        "JUnit contains a non-pass terminal",
    )

    m2 = manifest["m2_historical_replay"]
    require(
        isinstance(m2, dict)
        and set(m2)
        == {
            "accepted_repository_head",
            "acceptance_path",
            "acceptance_sha256",
            "acceptance_copy",
            "replay",
            "switch_stdout",
            "switch_stderr",
            "restore_stdout",
            "restore_stderr",
        },
        "M2 replay binding fields differ",
    )
    acceptance_copy = _validate_file_entry(
        root, m2["acceptance_copy"], label="M2 acceptance copy"
    )
    require(
        _sha256_file(acceptance_copy) == m2["acceptance_sha256"],
        "M2 acceptance copy differs",
    )
    acceptance_value, _ = _read_json(acceptance_copy, label="M2 acceptance copy")
    require(
        acceptance_value.get("gate_status") == "M2_ACCEPTED_CORRECTNESS_ONLY",
        "M2 gate status differs",
    )
    require(
        acceptance_value.get("repository", {}).get("head")
        == m2["accepted_repository_head"],
        "M2 accepted HEAD differs",
    )
    replay = m2["replay"]
    _validate_command(
        root,
        replay,
        expected_id="m2_historical_replay",
        expected_template=(
            str(python_path),
            "tools/m2_aggregate_acceptance.py",
            "validate",
            m2["acceptance_path"],
            "--expected-acceptance-sha256",
            m2["acceptance_sha256"],
        ),
    )
    replay_stdout_path = _validate_file_entry(
        root, replay.get("stdout"), label="M2 replay stdout"
    )
    _validate_file_entry(root, replay.get("stderr"), label="M2 replay stderr")
    require(
        f"sha256={m2['acceptance_sha256']}" in replay_stdout_path.read_text(),
        "M2 replay terminal differs",
    )
    for field in ("switch_stdout", "switch_stderr", "restore_stdout", "restore_stderr"):
        _validate_file_entry(root, m2[field], label=field)

    if require_sealed:
        require(
            stat.S_IMODE(root.stat().st_mode) == 0o555, "evidence root is not read-only"
        )
        for path in root.rglob("*"):
            expected_mode = 0o555 if path.is_dir() else 0o444
            require(
                stat.S_IMODE(path.stat().st_mode) == expected_mode,
                f"sealed mode differs: {path}",
            )
        lock_path = root.parent / f".{root.name}{PUBLICATION_LOCK_SUFFIX}"
        require(
            lock_path.is_file() and not lock_path.is_symlink(),
            "publication sidecar is missing",
        )
        require(
            lock_path.read_bytes() == LOCK_PUBLISHED,
            "publication sidecar is incomplete",
        )
        require(
            stat.S_IMODE(lock_path.stat().st_mode) == 0o444,
            "publication sidecar mode differs",
        )
    return manifest, manifest_sha, checksums_sha


def _seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is required for create-only publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise C1EvidenceError("evidence output already exists")
        raise C1EvidenceError(f"create-only publication failed: {os.strerror(error)}")


def run_bundle(
    output_root: Path,
    *,
    python: Path,
    acceptance: Path,
    expected_acceptance_sha256: str,
    accepted_head: str,
    timeout_seconds: int,
) -> tuple[str, str, int]:
    output_root = output_root.expanduser().resolve()
    python = python.expanduser().resolve(strict=True)
    acceptance = acceptance.expanduser().resolve(strict=True)
    require(output_root.is_absolute(), "output root must be absolute")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}{PUBLICATION_LOCK_SUFFIX}"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    published = False
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(lock_descriptor, LOCK_PREPARING)
        os.fsync(lock_descriptor)
        require(not output_root.exists(), "evidence output already exists")
        require(
            _sha256_file(acceptance) == expected_acceptance_sha256,
            "M2 acceptance SHA-256 differs",
        )
        branch = _git_text("branch", "--show-current")
        head = _git_text("rev-parse", "HEAD")
        require(branch == "main", "component evidence requires branch main")
        require(not _git_text("status", "--porcelain=v1"), "repository must be clean")
        require(
            _git_text("cat-file", "-t", accepted_head) == "commit",
            "accepted M2 HEAD is absent",
        )

        focused_junit = staging / "logs" / "c1-focused.junit.xml"
        full_junit = staging / "logs" / "repository-full.junit.xml"
        commands = [
            _run_command(
                "c1_focused",
                (
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "--junitxml={output_root}/logs/c1-focused.junit.xml",
                    PROBABILITY_TEST_PATH,
                ),
                output_root=staging,
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
            ),
            _run_command(
                "repository_full",
                (
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "--junitxml={output_root}/logs/repository-full.junit.xml",
                ),
                output_root=staging,
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
            ),
            _run_command(
                "ruff_check",
                (str(python), "-m", "ruff", "check", "."),
                output_root=staging,
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
            ),
            _run_command(
                "ruff_format_check",
                (str(python), "-m", "ruff", "format", "--check", "."),
                output_root=staging,
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
            ),
        ]
        require(
            all(
                command["exit_code"] == 0 and not command["timed_out"]
                for command in commands
            ),
            "component command failed",
        )
        focused_summary = _parse_junit(focused_junit)
        full_summary = _parse_junit(full_junit)
        require(
            focused_summary["tests"] == len(EXPECTED_C1_TESTS),
            "focused test count differs",
        )
        require(
            {
                identity.rsplit("::", maxsplit=1)[-1]
                for identity in focused_summary["identities"]
            }
            == EXPECTED_C1_TESTS,
            "focused testcase identities differ",
        )
        require(full_summary["tests"] >= 331, "full test count is below M3 baseline")
        source = _source_capture(staging, head)
        acceptance_copy = staging / "inputs" / "M2_AGGREGATE_ACCEPTANCE.json"
        _write_new(acceptance_copy, acceptance.read_bytes())
        m2 = _historical_m2_replay(
            staging,
            python=python,
            current_branch=branch,
            current_head=head,
            acceptance=acceptance,
            expected_acceptance_sha256=expected_acceptance_sha256,
            accepted_head=accepted_head,
            timeout_seconds=timeout_seconds,
        )
        m2["acceptance_copy"] = _file_entry(acceptance_copy, root=staging)
        python_version = (
            subprocess.run(
                [str(python), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "status": "C1_A_COMPONENT_VERIFIED",
            "claim_scope": CLAIM_SCOPE,
            "repository": {
                "root": str(REPO_ROOT),
                "head": head,
                "branch": branch,
                "tree": _git_text("show", "-s", "--format=%T", head),
                "clean": True,
            },
            "python": {
                "path": str(python),
                "sha256": _sha256_file(python),
                "version": python_version,
            },
            "source": source,
            "commands": commands,
            "junit": {
                "focused": {
                    "file": _file_entry(focused_junit, root=staging),
                    "summary": focused_summary,
                },
                "full": {
                    "file": _file_entry(full_junit, root=staging),
                    "summary": full_summary,
                },
            },
            "m2_historical_replay": m2,
            "gpu_used": False,
            "performance_claims_supported": False,
            "all_passed": True,
        }
        _write_new(staging / MANIFEST_NAME, _canonical_json(manifest))
        _write_checksums(staging)
        validate_bundle(
            staging,
            expected_manifest_sha256=None,
            expected_checksums_sha256=None,
            require_sealed=False,
        )
        _seal(staging)
        _rename_noreplace(staging, output_root)
        published = True
        _fsync_directory(output_root.parent)
        manifest_sha = _sha256_file(output_root / MANIFEST_NAME)
        checksums_sha = _sha256_file(output_root / CHECKSUM_NAME)
        os.lseek(lock_descriptor, 0, os.SEEK_SET)
        os.ftruncate(lock_descriptor, 0)
        os.write(lock_descriptor, LOCK_PUBLISHED)
        os.fsync(lock_descriptor)
        os.fchmod(lock_descriptor, 0o444)
        _fsync_directory(output_root.parent)
        validate_bundle(
            output_root,
            expected_manifest_sha256=manifest_sha,
            expected_checksums_sha256=checksums_sha,
            require_sealed=True,
        )
        return manifest_sha, checksums_sha, full_summary["tests"]
    except BaseException:
        if published:
            if output_root.exists():
                for path in output_root.rglob("*"):
                    with suppress(OSError):
                        os.chmod(path, 0o755 if path.is_dir() else 0o644)
                with suppress(OSError):
                    os.chmod(output_root, 0o755)
                shutil.rmtree(output_root, ignore_errors=True)
            published = False
            with suppress(OSError):
                os.fchmod(lock_descriptor, 0o600)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        os.close(lock_descriptor)
        if not published:
            with suppress(FileNotFoundError):
                lock_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run, seal, and replay C1-A evidence")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--python", required=True, type=Path)
    run.add_argument("--m2-acceptance", required=True, type=Path)
    run.add_argument("--expected-m2-sha256", required=True)
    run.add_argument("--accepted-m2-head", required=True)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    validate = subparsers.add_parser(
        "validate", help="independently replay a sealed bundle"
    )
    validate.add_argument("evidence", type=Path)
    validate.add_argument("--expected-manifest-sha256", required=True)
    validate.add_argument("--expected-checksums-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            require(arguments.timeout_seconds > 0, "timeout must be positive")
            manifest_sha, checksums_sha, tests = run_bundle(
                arguments.output_dir,
                python=arguments.python,
                acceptance=arguments.m2_acceptance,
                expected_acceptance_sha256=arguments.expected_m2_sha256,
                accepted_head=arguments.accepted_m2_head,
                timeout_seconds=arguments.timeout_seconds,
            )
            print(
                f"M3 C1-A component evidence passed: {tests} tests "
                f"manifest_sha256={manifest_sha} checksums_sha256={checksums_sha}"
            )
        else:
            manifest, manifest_sha, checksums_sha = validate_bundle(
                arguments.evidence,
                expected_manifest_sha256=arguments.expected_manifest_sha256,
                expected_checksums_sha256=arguments.expected_checksums_sha256,
            )
            print(
                f"M3 C1-A component replay passed: {manifest['status']} "
                f"manifest_sha256={manifest_sha} checksums_sha256={checksums_sha}"
            )
        return 0
    except (C1EvidenceError, OSError, subprocess.SubprocessError) as exc:
        print(f"M3 C1-A component evidence failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
