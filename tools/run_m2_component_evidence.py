#!/usr/bin/env python3
"""Run and independently replay the deterministic M2 CPU component contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.m2_raw_replay import (
        M2RawReplayError,
    )
    from tools.m2_raw_replay import (
        _validate_git_capture as _validate_raw_git_capture,
    )
    from tools.run_m2_vllm_abba import (
        M2ValidationError,
        _git_capture,
        _verify_git_capture,
        sha256_file,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from m2_raw_replay import (  # type: ignore[no-redef]
        M2RawReplayError,
    )
    from m2_raw_replay import (
        _validate_git_capture as _validate_raw_git_capture,
    )
    from run_m2_vllm_abba import (  # type: ignore[no-redef]
        M2ValidationError,
        _git_capture,
        _verify_git_capture,
        sha256_file,
    )

SCHEMA_VERSION = "dagkv.m2.component_evidence.v1"
CLAIM_SCOPE = (
    "M2 items 1-7 deterministic CPU component contract only; no item-8, "
    "performance, scheduling-policy, C1, C2, C3, or aggregate-M2 claim"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "M2_COMPONENT_EVIDENCE.json"
CHECKSUM_NAME = "SHA256SUMS"
INVALID_MANIFEST_NAME = "INVALID_M2_COMPONENT_EVIDENCE.json"
INVALID_CHECKSUM_NAME = "INVALID_SHA256SUMS"
MAX_JUNIT_BYTES = 32 * 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 900

BASE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
VLLM_RUNTIME_PROBE = (
    "import json,pytest,sys,torch,vllm;"
    "print(json.dumps({"
    "'executable':sys.executable,'python':sys.version,"
    "'pytest':pytest.__version__,'torch':torch.__version__,"
    "'torch_cuda':torch.version.cuda,'vllm':vllm.__version__,"
    "'vllm_file':vllm.__file__},sort_keys=True))"
)
DISTRIBUTION_PROBE = (
    "import hashlib\n"
    "import importlib.metadata as metadata\n"
    "import json\n"
    "import urllib.parse\n"
    "rows=[]\n"
    "for distribution in metadata.distributions():\n"
    " name=(distribution.metadata.get('Name') or '').strip().lower()"
    ".replace('_','-')\n"
    " if not name:\n"
    "  continue\n"
    " raw=distribution.read_text('direct_url.json')\n"
    " direct_url=json.loads(raw) if raw else None\n"
    " encoded=(json.dumps(direct_url,sort_keys=True,separators=(',',':'))"
    ".encode() if direct_url is not None else None)\n"
    " url=(direct_url.get('url') if isinstance(direct_url,dict) else None)\n"
    " scheme=(urllib.parse.urlsplit(url).scheme.lower() "
    "if isinstance(url,str) else None)\n"
    " dir_info=(direct_url.get('dir_info') "
    "if isinstance(direct_url,dict) else None)\n"
    " rows.append({'name':name,'version':distribution.version,"
    "'direct_url_sha256':(hashlib.sha256(encoded).hexdigest() "
    "if encoded is not None else None),'editable':(isinstance(dir_info,dict) "
    "and dir_info.get('editable') is True),'source_scheme':scheme or None})\n"
    "rows.sort(key=lambda item:(item['name'],item['version'],"
    "item['direct_url_sha256'] or '',item['editable'],"
    "item['source_scheme'] or ''))\n"
    "print(json.dumps(rows,separators=(',',':'),sort_keys=True))\n"
)
REQUIRED_DISTRIBUTIONS = {
    "dagkv": frozenset({"dagkv", "numpy", "pytest", "ruff"}),
    "vllm": frozenset({"numpy", "pytest", "torch", "vllm"}),
}


class ComponentEvidenceError(RuntimeError):
    """Raised when CPU component evidence cannot be created or replayed."""


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """One frozen deterministic test suite in the component contract."""

    suite_id: str
    root_kind: str
    python_kind: str
    test_paths: tuple[str, ...]
    minimum_tests: int
    needs_integration_path: bool = False


SUITE_SPECS = (
    SuiteSpec(
        suite_id="dagkv_core",
        root_kind="dagkv",
        python_kind="dagkv",
        test_paths=("tests",),
        minimum_tests=269,
    ),
    SuiteSpec(
        suite_id="dagkv_vllm_adapter",
        root_kind="dagkv",
        python_kind="vllm",
        test_paths=("integrations/vllm_m2/tests/test_contract.py",),
        minimum_tests=13,
        needs_integration_path=True,
    ),
    SuiteSpec(
        suite_id="vllm_lifecycle_cpu",
        root_kind="vllm",
        python_kind="vllm",
        test_paths=(
            "tests/v1/kv_connector/unit/offloading_connector/test_events.py",
            "tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py",
            "tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py",
            "tests/v1/kv_offload/cpu/test_manager.py",
            "tests/v1/kv_offload/cpu/test_shared_offload_region.py",
            "tests/v1/kv_offload/test_factory.py",
            "tests/v1/kv_offload/test_fanout_planner.py",
            "tests/v1/kv_offload/test_lifecycle.py",
        ),
        minimum_tests=345,
    ),
)


def require(condition: bool, message: str) -> None:
    """Raise a stable fail-closed validation error."""

    if not condition:
        raise ComponentEvidenceError(message)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _lower_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} must be non-empty")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ComponentEvidenceError(f"invalid {label}: {value}") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ComponentEvidenceError(f"non-finite JSON constant: {value}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComponentEvidenceError(f"invalid {label}: {exc}") from exc
    require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ComponentEvidenceError(f"refusing to overwrite evidence: {path}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_output_root(path: Path) -> Path:
    require(path.is_absolute(), "component evidence output must be absolute")
    require(path.name not in {"", ".", ".."}, "component evidence output is invalid")
    parent = path.parent.resolve(strict=True)
    require(parent.is_dir() and not parent.is_symlink(), "output parent is invalid")
    output = parent / path.name
    require(not output.exists() and not output.is_symlink(), "output already exists")
    try:
        output.mkdir(mode=0o755)
    except OSError as exc:
        raise ComponentEvidenceError(f"cannot create output directory: {exc}") from exc
    _fsync_directory(parent)
    return output


def _file_entry(
    path: Path,
    *,
    root: Path,
    maximum_size: int = MAX_TEXT_BYTES,
) -> dict[str, Any]:
    observed = path.lstat()
    require(stat.S_ISREG(observed.st_mode), f"evidence file is not regular: {path}")
    require(observed.st_nlink == 1, f"evidence file has a hard link: {path}")
    require(observed.st_size <= maximum_size, f"evidence file is too large: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size": observed.st_size,
        "sha256": sha256_file(path),
    }


def _absolute_executable(path: Path, *, label: str) -> Path:
    require(path.is_absolute(), f"{label} executable path must be absolute")
    launcher = Path(os.path.abspath(path))
    try:
        observed = launcher.lstat()
        resolved = launcher.resolve(strict=True)
    except OSError as exc:
        raise ComponentEvidenceError(
            f"cannot inspect {label} executable: {exc}"
        ) from exc
    require(
        stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode),
        f"{label} executable launcher is invalid",
    )
    require(resolved.is_file(), f"{label} executable target is invalid")
    require(os.access(launcher, os.X_OK), f"{label} executable is not executable")
    return launcher


def _python_entry(path: Path, version: str) -> dict[str, Any]:
    launcher = _absolute_executable(path, label="Python")
    resolved = path.resolve(strict=True)
    observed = resolved.stat()
    require(stat.S_ISREG(observed.st_mode), f"Python executable is invalid: {resolved}")
    return {
        "path": str(launcher),
        "resolved_path": str(resolved),
        "size": observed.st_size,
        "sha256": sha256_file(resolved),
        "version": version,
    }


def _clean_environment(
    *,
    integration_path: bool = False,
    dagkv_root: Path = REPO_ROOT,
) -> dict[str, str]:
    environment = dict(BASE_ENVIRONMENT)
    if integration_path:
        environment["PYTHONPATH"] = str(dagkv_root / "integrations" / "vllm_m2")
    return environment


def _run_capture(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_s: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComponentEvidenceError(f"metadata command failed: {argv}: {exc}") from exc


def _parse_distribution_rows(
    payload: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    require(isinstance(payload, list) and payload, f"{label} inventory is empty")
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    for index, row in enumerate(payload):
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "direct_url_sha256",
                "editable",
                "name",
                "source_scheme",
                "version",
            },
            f"{label} distribution row {index} fields differ",
        )
        name = row["name"]
        version = row["version"]
        direct_url_sha256 = row["direct_url_sha256"]
        editable = row["editable"]
        source_scheme = row["source_scheme"]
        require(
            isinstance(name, str) and name and name == name.lower().replace("_", "-"),
            f"{label} distribution name is invalid",
        )
        require(
            isinstance(version, str) and version,
            f"{label} distribution version is invalid",
        )
        require(
            direct_url_sha256 is None
            or (
                isinstance(direct_url_sha256, str)
                and len(direct_url_sha256) == 64
                and all(
                    character in "0123456789abcdef" for character in direct_url_sha256
                )
            ),
            f"{label} direct_url_sha256 is invalid",
        )
        require(
            isinstance(editable, bool),
            f"{label} editable flag is invalid",
        )
        require(
            source_scheme is None
            or (
                isinstance(source_scheme, str)
                and source_scheme
                and source_scheme == source_scheme.lower()
                and all(
                    character in "abcdefghijklmnopqrstuvwxyz0123456789+.-"
                    for character in source_scheme
                )
            ),
            f"{label} source scheme is invalid",
        )
        rows.append(row)
        names.append(name)
    require(len(names) == len(set(names)), f"{label} distribution names are ambiguous")
    expected = sorted(
        rows,
        key=lambda item: (
            item["name"],
            item["version"],
            item["direct_url_sha256"] or "",
            item["editable"],
            item["source_scheme"] or "",
        ),
    )
    require(rows == expected, f"{label} distribution inventory is not sorted")
    require(
        REQUIRED_DISTRIBUTIONS[label].issubset(names),
        f"{label} required distributions are missing",
    )
    return rows


def _capture_distributions(
    python: Path,
    *,
    label: str,
    output_root: Path,
    cwd: Path,
    timeout_s: int,
) -> dict[str, Any]:
    argv = [str(python), "-c", DISTRIBUTION_PROBE]
    environment = _clean_environment()
    result = _run_capture(
        argv,
        cwd=cwd,
        environment=environment,
        timeout_s=timeout_s,
    )
    require(result.returncode == 0, f"{label} distribution probe failed")
    require(len(result.stdout) <= MAX_TEXT_BYTES, f"{label} inventory is too large")
    require(len(result.stderr) <= MAX_TEXT_BYTES, f"{label} probe stderr is too large")
    try:
        rows = json.loads(
            result.stdout.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComponentEvidenceError(f"invalid {label} inventory: {exc}") from exc
    rows = _parse_distribution_rows(rows, label=label)
    inventory_path = output_root / "environment" / f"{label}-distributions.json"
    stderr_path = output_root / "environment" / f"{label}-distributions.stderr.txt"
    inventory = {
        "schema_version": "dagkv.python_distribution_inventory.v1",
        "distributions": rows,
    }
    _write_new(inventory_path, _canonical_json(inventory))
    _write_new(stderr_path, result.stderr)
    return {
        "argv": argv,
        "environment": environment,
        "inventory": _file_entry(inventory_path, root=output_root),
        "stderr": _file_entry(stderr_path, root=output_root),
    }


def _capture_vllm_runtime(
    python: Path,
    *,
    vllm_root: Path,
    output_root: Path,
    timeout_s: int,
) -> tuple[dict[str, Any], str]:
    argv = [str(python), "-c", VLLM_RUNTIME_PROBE]
    environment = _clean_environment()
    result = _run_capture(
        argv,
        cwd=vllm_root,
        environment=environment,
        timeout_s=timeout_s,
    )
    require(result.returncode == 0, "vLLM runtime probe failed")
    try:
        payload = json.loads(
            result.stdout.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComponentEvidenceError(f"invalid vLLM runtime probe: {exc}") from exc
    require(isinstance(payload, dict), "vLLM runtime probe must be an object")
    require(
        set(payload)
        == {
            "executable",
            "pytest",
            "python",
            "torch",
            "torch_cuda",
            "vllm",
            "vllm_file",
        },
        "vLLM runtime probe fields differ",
    )
    require(
        isinstance(payload["executable"], str) and payload["executable"],
        "vLLM Python executable is invalid",
    )
    executable = Path(os.path.abspath(payload["executable"]))
    require(executable == python, "vLLM Python executable differs")
    module = Path(payload["vllm_file"]).resolve(strict=True)
    require(module.is_relative_to(vllm_root), "imported vLLM is outside vLLM root")
    payload["vllm_file"] = str(module)
    payload["vllm_file_sha256"] = sha256_file(module)
    runtime_path = output_root / "environment" / "vllm-runtime.json"
    stderr_path = output_root / "environment" / "vllm-runtime.stderr.txt"
    encoded_runtime = _canonical_json(payload)
    require(len(encoded_runtime) <= MAX_TEXT_BYTES, "vLLM runtime output is too large")
    require(len(result.stderr) <= MAX_TEXT_BYTES, "vLLM runtime stderr is too large")
    _write_new(runtime_path, encoded_runtime)
    _write_new(stderr_path, result.stderr)
    return (
        {
            "argv": argv,
            "environment": environment,
            "runtime": _file_entry(runtime_path, root=output_root),
            "stderr": _file_entry(stderr_path, root=output_root),
        },
        payload["python"],
    )


def _suite_argv(
    spec: SuiteSpec,
    *,
    python: Path,
    output_root: Path,
) -> list[str]:
    junit = output_root / "logs" / f"{spec.suite_id}.junit.xml"
    return [
        str(python),
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        f"--junitxml={junit}",
        *spec.test_paths,
    ]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _parse_junit(path: Path, *, minimum_tests: int) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing JUnit file: {path}")
    raw = path.read_bytes()
    require(len(raw) <= MAX_JUNIT_BYTES, f"JUnit file is too large: {path}")
    require(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "unsafe JUnit XML")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ComponentEvidenceError(f"invalid JUnit XML at {path}: {exc}") from exc
    cases = list(root.iter("testcase"))
    identities: list[tuple[str, str]] = []
    failures = errors = skipped = 0
    for case in cases:
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        require(classname and name, "JUnit testcase identity is incomplete")
        identities.append((classname, name))
        failures += sum(child.tag == "failure" for child in case)
        errors += sum(child.tag == "error" for child in case)
        skipped += sum(child.tag == "skipped" for child in case)
    require(
        len(identities) == len(set(identities)),
        "JUnit contains duplicate testcase identities",
    )
    tests = len(cases)
    require(tests >= minimum_tests, f"JUnit test count is below {minimum_tests}")
    require(failures == 0, "JUnit contains failed tests")
    require(errors == 0, "JUnit contains errored tests")
    require(skipped == 0, "JUnit contains skipped tests")
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _run_suite(
    spec: SuiteSpec,
    *,
    dagkv_python: Path,
    vllm_python: Path,
    vllm_root: Path,
    output_root: Path,
    timeout_s: int,
) -> dict[str, Any]:
    python = dagkv_python if spec.python_kind == "dagkv" else vllm_python
    cwd = REPO_ROOT if spec.root_kind == "dagkv" else vllm_root
    environment = _clean_environment(integration_path=spec.needs_integration_path)
    argv = _suite_argv(spec, python=python, output_root=output_root)
    stdout_path = output_root / "logs" / f"{spec.suite_id}.stdout.txt"
    stderr_path = output_root / "logs" / f"{spec.suite_id}.stderr.txt"
    junit_path = output_root / "logs" / f"{spec.suite_id}.junit.xml"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except OSError as exc:
            raise ComponentEvidenceError(
                f"cannot start suite {spec.suite_id}: {exc}"
            ) from exc
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise ComponentEvidenceError(
                f"suite {spec.suite_id} exceeded {timeout_s} seconds"
            ) from exc
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    completed_ns = time.monotonic_ns()
    completed_at = _utc_now()
    require(returncode == 0, f"suite {spec.suite_id} exited with {returncode}")
    counts = _parse_junit(junit_path, minimum_tests=spec.minimum_tests)
    return {
        "suite_id": spec.suite_id,
        "cwd": str(cwd),
        "python": str(python),
        "argv": argv,
        "environment": environment,
        "minimum_tests": spec.minimum_tests,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "duration_seconds": (completed_ns - started_ns) / 1_000_000_000,
        "exit_code": returncode,
        **counts,
        "passed": True,
        "stdout": _file_entry(stdout_path, root=output_root),
        "stderr": _file_entry(stderr_path, root=output_root),
        "junit": _file_entry(
            junit_path,
            root=output_root,
            maximum_size=MAX_JUNIT_BYTES,
        ),
    }


def _expected_files() -> set[str]:
    files = {
        MANIFEST_NAME,
        CHECKSUM_NAME,
        "environment/dagkv-distributions.json",
        "environment/dagkv-distributions.stderr.txt",
        "environment/vllm-distributions.json",
        "environment/vllm-distributions.stderr.txt",
        "environment/vllm-runtime.json",
        "environment/vllm-runtime.stderr.txt",
        "source_state/dagkv.tracked.patch",
        "source_state/dagkv.untracked.tar",
        "source_state/vllm.tracked.patch",
        "source_state/vllm.untracked.tar",
    }
    for spec in SUITE_SPECS:
        files.update(
            {
                f"logs/{spec.suite_id}.stdout.txt",
                f"logs/{spec.suite_id}.stderr.txt",
                f"logs/{spec.suite_id}.junit.xml",
            }
        )
    return files


def _write_sha256sums(output_root: Path) -> None:
    paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_NAME
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}"
        for path in paths
    ]
    _write_new(output_root / CHECKSUM_NAME, ("\n".join(lines) + "\n").encode())


def _validate_tree(output_root: Path, *, require_read_only: bool) -> None:
    require(
        output_root.is_dir() and not output_root.is_symlink(), "evidence root missing"
    )
    expected_files = _expected_files()
    expected_dirs = {"environment", "logs", "source_state"}
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in output_root.rglob("*"):
        relative = path.relative_to(output_root).as_posix()
        observed = path.lstat()
        require(not stat.S_ISLNK(observed.st_mode), f"evidence symlink: {relative}")
        if stat.S_ISDIR(observed.st_mode):
            observed_dirs.add(relative)
            if require_read_only:
                require(
                    stat.S_IMODE(observed.st_mode) == 0o555,
                    f"bad directory mode: {relative}",
                )
        elif stat.S_ISREG(observed.st_mode):
            require(observed.st_nlink == 1, f"hard-linked evidence file: {relative}")
            observed_files.add(relative)
            if require_read_only:
                require(
                    stat.S_IMODE(observed.st_mode) == 0o444,
                    f"bad file mode: {relative}",
                )
        else:
            raise ComponentEvidenceError(f"special evidence node: {relative}")
    require(observed_dirs == expected_dirs, "component evidence directories differ")
    require(observed_files == expected_files, "component evidence files differ")
    if require_read_only:
        require(stat.S_IMODE(output_root.stat().st_mode) == 0o555, "bad root mode")


def _validate_sha256sums(output_root: Path) -> dict[str, str]:
    checksum = output_root / CHECKSUM_NAME
    require(checksum.is_file() and not checksum.is_symlink(), "SHA256SUMS missing")
    raw = checksum.read_bytes()
    require(raw.endswith(b"\n") and b"\r" not in raw, "invalid SHA256SUMS lines")
    entries: dict[str, str] = {}
    ordered: list[str] = []
    for line_number, raw_line in enumerate(raw[:-1].split(b"\n"), start=1):
        try:
            line = raw_line.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ComponentEvidenceError("SHA256SUMS must be ASCII") from exc
        require(
            len(line) >= 67 and line[64:66] == "  ", f"bad checksum row {line_number}"
        )
        digest = _lower_sha256(line[:64], label=f"checksum row {line_number}")
        name = line[66:]
        relative = Path(name)
        require(
            name
            and not relative.is_absolute()
            and ".." not in relative.parts
            and name != CHECKSUM_NAME,
            f"unsafe checksum path: {name}",
        )
        require(name not in entries, f"duplicate checksum path: {name}")
        entries[name] = digest
        ordered.append(name)
    require(ordered == sorted(ordered), "SHA256SUMS paths are not sorted")
    require(
        set(entries) == _expected_files() - {CHECKSUM_NAME},
        "checksum closed set differs",
    )
    for name, expected in entries.items():
        require(
            sha256_file(output_root / name) == expected, f"checksum mismatch: {name}"
        )
    return entries


def _validate_file_entry(
    value: Any,
    *,
    output_root: Path,
    entries: Mapping[str, str],
    expected_path: str,
) -> None:
    require(
        isinstance(value, dict) and set(value) == {"path", "sha256", "size"},
        f"file entry fields differ: {expected_path}",
    )
    require(value["path"] == expected_path, f"file entry path differs: {expected_path}")
    digest = _lower_sha256(value["sha256"], label=f"{expected_path} SHA")
    require(
        entries[expected_path] == digest,
        f"file entry checksum differs: {expected_path}",
    )
    size = value["size"]
    require(type(size) is int and size >= 0, f"file size invalid: {expected_path}")
    require(
        (output_root / expected_path).stat().st_size == size,
        f"file size differs: {expected_path}",
    )


def _validate_python_entry(value: Any, *, label: str, verify_external: bool) -> Path:
    require(
        isinstance(value, dict)
        and set(value) == {"path", "resolved_path", "sha256", "size", "version"},
        f"{label} Python fields differ",
    )
    path = Path(value["path"])
    resolved = Path(value["resolved_path"])
    require(path.is_absolute(), f"{label} Python path must be absolute")
    require(resolved.is_absolute(), f"{label} resolved Python path must be absolute")
    _lower_sha256(value["sha256"], label=f"{label} Python SHA")
    require(type(value["size"]) is int and value["size"] > 0, f"{label} size invalid")
    require(
        isinstance(value["version"], str) and value["version"],
        f"{label} version invalid",
    )
    if verify_external:
        require(path.is_file(), f"{label} Python launcher is missing")
        require(path.resolve(strict=True) == resolved, f"{label} Python target differs")
        require(
            resolved.stat().st_size == value["size"], f"{label} Python size differs"
        )
        require(sha256_file(resolved) == value["sha256"], f"{label} Python SHA differs")
    return path


def _validate_distribution_capture(
    value: Any,
    *,
    output_root: Path,
    entries: Mapping[str, str],
    label: str,
    python: Path,
) -> None:
    require(
        isinstance(value, dict)
        and set(value) == {"argv", "environment", "inventory", "stderr"},
        f"{label} distribution capture fields differ",
    )
    require(
        value["argv"] == [str(python), "-c", DISTRIBUTION_PROBE],
        f"{label} distribution argv differs",
    )
    require(
        value["environment"] == _clean_environment(),
        f"{label} distribution environment differs",
    )
    _validate_file_entry(
        value["inventory"],
        output_root=output_root,
        entries=entries,
        expected_path=f"environment/{label}-distributions.json",
    )
    _validate_file_entry(
        value["stderr"],
        output_root=output_root,
        entries=entries,
        expected_path=f"environment/{label}-distributions.stderr.txt",
    )
    inventory = _read_json(
        output_root / "environment" / f"{label}-distributions.json",
        label=f"{label} distribution inventory",
    )
    require(
        set(inventory) == {"distributions", "schema_version"},
        f"{label} distribution inventory fields differ",
    )
    require(
        inventory["schema_version"] == "dagkv.python_distribution_inventory.v1",
        f"{label} distribution inventory schema differs",
    )
    _parse_distribution_rows(inventory["distributions"], label=label)


def _seal_permissions(output_root: Path) -> None:
    for path in output_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (item for item in output_root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    output_root.chmod(0o555)
    _fsync_directory(output_root.parent)


def validate_component_evidence(
    output_root: Path,
    *,
    require_read_only: bool = True,
    verify_external: bool = True,
) -> dict[str, Any]:
    """Independently replay one sealed M2 CPU component evidence directory."""

    require(not output_root.is_symlink(), "component evidence root cannot be a symlink")
    output_root = output_root.resolve(strict=True)
    _validate_tree(output_root, require_read_only=require_read_only)
    entries = _validate_sha256sums(output_root)
    manifest = _read_json(output_root / MANIFEST_NAME, label=MANIFEST_NAME)
    require(
        set(manifest)
        == {
            "all_passed",
            "claim_scope",
            "created_at_utc",
            "dagkv_git",
            "determinism",
            "eligible_gate_scope",
            "environment",
            "evidence_root",
            "gpu_used",
            "item8_accepted",
            "m2_accepted",
            "postflight",
            "schema_version",
            "suites",
            "total_tests",
            "verification_status",
            "vllm_git",
        },
        "component manifest fields differ",
    )
    require(manifest["schema_version"] == SCHEMA_VERSION, "wrong component schema")
    require(manifest["claim_scope"] == CLAIM_SCOPE, "component claim scope differs")
    require(manifest["determinism"] == "deterministic", "wrong determinism class")
    require(manifest["verification_status"] == "VERIFIED", "evidence is not verified")
    require(manifest["eligible_gate_scope"] == list(range(1, 8)), "gate scope differs")
    require(manifest["gpu_used"] is False, "CPU component evidence claims GPU use")
    require(manifest["item8_accepted"] is False, "component evidence accepts item 8")
    require(manifest["m2_accepted"] is False, "component evidence accepts M2")
    require(manifest["all_passed"] is True, "component evidence did not pass")
    _timestamp(manifest["created_at_utc"], label="created_at_utc")
    recorded_output_root = Path(manifest["evidence_root"])
    require(
        recorded_output_root.is_absolute(), "recorded evidence root is not absolute"
    )
    try:
        dagkv_snapshot = _validate_raw_git_capture(
            manifest["dagkv_git"],
            label="dagkv",
            run_dir=output_root,
            entries=entries,
        )
        vllm_snapshot = _validate_raw_git_capture(
            manifest["vllm_git"],
            label="vllm",
            run_dir=output_root,
            entries=entries,
        )
    except M2RawReplayError as exc:
        raise ComponentEvidenceError(str(exc)) from exc
    require(manifest["dagkv_git"]["dirty"] is False, "DAGKV source was dirty")

    environment = manifest["environment"]
    require(
        isinstance(environment, dict)
        and set(environment)
        == {
            "dagkv_distributions",
            "dagkv_python",
            "policy",
            "system",
            "tool",
            "vllm_distributions",
            "vllm_python",
            "vllm_runtime",
        },
        "component environment fields differ",
    )
    dagkv_python = _validate_python_entry(
        environment["dagkv_python"], label="dagkv", verify_external=verify_external
    )
    vllm_python = _validate_python_entry(
        environment["vllm_python"], label="vllm", verify_external=verify_external
    )
    _validate_distribution_capture(
        environment["dagkv_distributions"],
        output_root=output_root,
        entries=entries,
        label="dagkv",
        python=dagkv_python,
    )
    _validate_distribution_capture(
        environment["vllm_distributions"],
        output_root=output_root,
        entries=entries,
        label="vllm",
        python=vllm_python,
    )
    policy = environment["policy"]
    require(
        isinstance(policy, dict)
        and set(policy)
        == {
            "cuda_visible_devices",
            "no_bytecode",
            "no_cacheprovider",
            "no_gpu",
            "no_retry",
            "offline",
            "timeout_seconds",
        },
        "component test policy fields differ",
    )
    require(
        type(policy["timeout_seconds"]) is int and policy["timeout_seconds"] > 0,
        "component timeout is invalid",
    )
    require(
        {key: policy[key] for key in policy if key != "timeout_seconds"}
        == {
            "cuda_visible_devices": "",
            "no_bytecode": True,
            "no_cacheprovider": True,
            "no_gpu": True,
            "no_retry": True,
            "offline": True,
        },
        "component test policy differs",
    )
    system = environment["system"]
    require(
        isinstance(system, dict) and set(system) == {"platform", "uname"},
        "system fields differ",
    )
    require(
        isinstance(system["platform"], str) and system["platform"], "platform missing"
    )
    require(
        isinstance(system["uname"], list)
        and len(system["uname"]) == 6
        and all(isinstance(item, str) for item in system["uname"]),
        "uname differs",
    )
    tool = environment["tool"]
    require(
        isinstance(tool, dict) and set(tool) == {"argv", "path", "sha256"},
        "tool fields differ",
    )
    require(
        isinstance(tool["argv"], list)
        and tool["argv"]
        and all(isinstance(item, str) for item in tool["argv"]),
        "tool argv missing",
    )
    tool_path = Path(tool["path"])
    require(tool_path.is_absolute(), "tool path must be absolute")
    _lower_sha256(tool["sha256"], label="tool SHA")
    if verify_external:
        require(tool_path.is_file(), "component tool is missing")
        require(sha256_file(tool_path) == tool["sha256"], "component tool SHA differs")

    runtime = environment["vllm_runtime"]
    require(
        isinstance(runtime, dict)
        and set(runtime) == {"argv", "environment", "runtime", "stderr"},
        "vLLM runtime capture fields differ",
    )
    require(
        runtime["argv"] == [str(vllm_python), "-c", VLLM_RUNTIME_PROBE],
        "runtime argv differs",
    )
    require(
        runtime["environment"] == _clean_environment(), "runtime environment differs"
    )
    _validate_file_entry(
        runtime["runtime"],
        output_root=output_root,
        entries=entries,
        expected_path="environment/vllm-runtime.json",
    )
    _validate_file_entry(
        runtime["stderr"],
        output_root=output_root,
        entries=entries,
        expected_path="environment/vllm-runtime.stderr.txt",
    )
    runtime_payload = _read_json(
        output_root / "environment" / "vllm-runtime.json",
        label="vLLM runtime",
    )
    require(
        set(runtime_payload)
        == {
            "executable",
            "pytest",
            "python",
            "torch",
            "torch_cuda",
            "vllm",
            "vllm_file",
            "vllm_file_sha256",
        },
        "vLLM runtime fields differ",
    )
    for key in ("executable", "pytest", "python", "torch", "vllm", "vllm_file"):
        require(
            isinstance(runtime_payload[key], str) and runtime_payload[key],
            f"vLLM runtime {key} is invalid",
        )
    require(
        runtime_payload["torch_cuda"] is None
        or (
            isinstance(runtime_payload["torch_cuda"], str)
            and runtime_payload["torch_cuda"]
        ),
        "vLLM runtime torch_cuda is invalid",
    )
    require(
        Path(os.path.abspath(runtime_payload["executable"])) == vllm_python,
        "runtime Python differs",
    )
    vllm_root = Path(manifest["vllm_git"]["root"])
    module = Path(runtime_payload["vllm_file"])
    require(
        module.is_absolute() and module.is_relative_to(vllm_root),
        "vLLM module path differs",
    )
    _lower_sha256(runtime_payload["vllm_file_sha256"], label="vLLM module SHA")
    if verify_external:
        require(module.is_file(), "vLLM module is missing")
        require(
            sha256_file(module) == runtime_payload["vllm_file_sha256"],
            "vLLM module SHA differs",
        )

    dagkv_root = Path(manifest["dagkv_git"]["root"])
    suites = manifest["suites"]
    require(
        isinstance(suites, list) and len(suites) == len(SUITE_SPECS),
        "suite set differs",
    )
    total_tests = 0
    for spec, suite in zip(SUITE_SPECS, suites, strict=True):
        require(
            isinstance(suite, dict)
            and set(suite)
            == {
                "argv",
                "completed_at_utc",
                "cwd",
                "duration_seconds",
                "environment",
                "errors",
                "exit_code",
                "failures",
                "junit",
                "minimum_tests",
                "passed",
                "python",
                "skipped",
                "started_at_utc",
                "stderr",
                "stdout",
                "suite_id",
                "tests",
            },
            f"suite fields differ: {spec.suite_id}",
        )
        require(suite["suite_id"] == spec.suite_id, "suite order differs")
        expected_python = dagkv_python if spec.python_kind == "dagkv" else vllm_python
        expected_cwd = dagkv_root if spec.root_kind == "dagkv" else vllm_root
        require(
            suite["python"] == str(expected_python), f"{spec.suite_id} Python differs"
        )
        require(suite["cwd"] == str(expected_cwd), f"{spec.suite_id} cwd differs")
        require(
            suite["argv"]
            == _suite_argv(
                spec,
                python=expected_python,
                output_root=recorded_output_root,
            ),
            f"{spec.suite_id} argv differs",
        )
        require(
            suite["environment"]
            == _clean_environment(
                integration_path=spec.needs_integration_path,
                dagkv_root=dagkv_root,
            ),
            f"{spec.suite_id} environment differs",
        )
        require(suite["minimum_tests"] == spec.minimum_tests, "minimum tests differ")
        started = _timestamp(suite["started_at_utc"], label=f"{spec.suite_id} start")
        completed = _timestamp(suite["completed_at_utc"], label=f"{spec.suite_id} end")
        require(completed >= started, f"{spec.suite_id} timestamps differ")
        duration = suite["duration_seconds"]
        require(
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            and duration >= 0,
            f"{spec.suite_id} duration differs",
        )
        require(
            suite["exit_code"] == 0 and suite["passed"] is True, "suite did not pass"
        )
        stdout_name = f"logs/{spec.suite_id}.stdout.txt"
        stderr_name = f"logs/{spec.suite_id}.stderr.txt"
        junit_name = f"logs/{spec.suite_id}.junit.xml"
        _validate_file_entry(
            suite["stdout"],
            output_root=output_root,
            entries=entries,
            expected_path=stdout_name,
        )
        _validate_file_entry(
            suite["stderr"],
            output_root=output_root,
            entries=entries,
            expected_path=stderr_name,
        )
        _validate_file_entry(
            suite["junit"],
            output_root=output_root,
            entries=entries,
            expected_path=junit_name,
        )
        counts = _parse_junit(
            output_root / junit_name, minimum_tests=spec.minimum_tests
        )
        for key in ("tests", "failures", "errors", "skipped"):
            require(suite[key] == counts[key], f"{spec.suite_id} {key} differs")
        total_tests += counts["tests"]
    require(manifest["total_tests"] == total_tests, "total test count differs")

    postflight = manifest["postflight"]
    require(
        postflight
        == {
            "dagkv_snapshot_sha256": dagkv_snapshot,
            "source_state_unchanged": True,
            "vllm_snapshot_sha256": vllm_snapshot,
        },
        "component postflight differs",
    )
    return manifest


def create_component_evidence(
    *,
    output_dir: Path,
    dagkv_python: Path,
    vllm_python: Path,
    vllm_root: Path,
    timeout_s: int,
) -> dict[str, Any]:
    """Create, validate, and seal one deterministic component evidence bundle."""

    require(timeout_s > 0, "timeout must be positive")
    output_root = _create_output_root(output_dir)
    started_at = _utc_now()
    try:
        dagkv_python = _absolute_executable(dagkv_python, label="DAGKV Python")
        vllm_python = _absolute_executable(vllm_python, label="vLLM Python")
        require(vllm_root.is_absolute(), "vLLM root must be absolute")
        vllm_root = vllm_root.resolve(strict=True)
        require(
            Path(os.path.abspath(sys.executable)) == dagkv_python,
            "component runner must execute with --dagkv-python",
        )
        require(vllm_root.is_dir(), "vLLM root is missing")
        dagkv_git = _git_capture(REPO_ROOT, output_dir=output_root, label="dagkv")
        require(not dagkv_git["dirty"], "DAGKV worktree must be clean")
        vllm_git = _git_capture(vllm_root, output_dir=output_root, label="vllm")

        dagkv_version = platform.python_version()
        vllm_runtime, vllm_version = _capture_vllm_runtime(
            vllm_python,
            vllm_root=vllm_root,
            output_root=output_root,
            timeout_s=timeout_s,
        )
        dagkv_distributions = _capture_distributions(
            dagkv_python,
            label="dagkv",
            output_root=output_root,
            cwd=REPO_ROOT,
            timeout_s=timeout_s,
        )
        vllm_distributions = _capture_distributions(
            vllm_python,
            label="vllm",
            output_root=output_root,
            cwd=vllm_root,
            timeout_s=timeout_s,
        )
        suites = [
            _run_suite(
                spec,
                dagkv_python=dagkv_python,
                vllm_python=vllm_python,
                vllm_root=vllm_root,
                output_root=output_root,
                timeout_s=timeout_s,
            )
            for spec in SUITE_SPECS
        ]
        dagkv_snapshot = _verify_git_capture(dagkv_git, label="dagkv")
        vllm_snapshot = _verify_git_capture(vllm_git, label="vllm")
        tool_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "claim_scope": CLAIM_SCOPE,
            "verification_status": "VERIFIED",
            "determinism": "deterministic",
            "eligible_gate_scope": list(range(1, 8)),
            "evidence_root": str(output_root),
            "gpu_used": False,
            "item8_accepted": False,
            "m2_accepted": False,
            "dagkv_git": dagkv_git,
            "vllm_git": vllm_git,
            "environment": {
                "tool": {
                    "argv": list(sys.argv),
                    "path": str(tool_path),
                    "sha256": sha256_file(tool_path),
                },
                "dagkv_python": _python_entry(dagkv_python, dagkv_version),
                "vllm_python": _python_entry(vllm_python, vllm_version),
                "dagkv_distributions": dagkv_distributions,
                "vllm_distributions": vllm_distributions,
                "vllm_runtime": vllm_runtime,
                "system": {
                    "platform": platform.platform(),
                    "uname": list(platform.uname()),
                },
                "policy": {
                    "cuda_visible_devices": "",
                    "no_bytecode": True,
                    "no_cacheprovider": True,
                    "no_gpu": True,
                    "no_retry": True,
                    "offline": True,
                    "timeout_seconds": timeout_s,
                },
            },
            "suites": suites,
            "total_tests": sum(suite["tests"] for suite in suites),
            "all_passed": True,
            "postflight": {
                "dagkv_snapshot_sha256": dagkv_snapshot,
                "vllm_snapshot_sha256": vllm_snapshot,
                "source_state_unchanged": True,
            },
        }
        _write_new(output_root / MANIFEST_NAME, _canonical_json(manifest))
        _write_sha256sums(output_root)
        validate_component_evidence(
            output_root,
            require_read_only=False,
            verify_external=True,
        )
        _seal_permissions(output_root)
        validated = validate_component_evidence(output_root, verify_external=True)
        return validated
    except (ComponentEvidenceError, M2ValidationError, OSError) as exc:
        failure_path = output_root / "FAILURE.json"
        with suppress(OSError):
            output_root.chmod(0o755)
        candidate_manifest = output_root / MANIFEST_NAME
        candidate_checksum = output_root / CHECKSUM_NAME
        if candidate_manifest.exists():
            with suppress(OSError):
                os.replace(candidate_manifest, output_root / INVALID_MANIFEST_NAME)
        if candidate_checksum.exists():
            with suppress(OSError):
                os.replace(candidate_checksum, output_root / INVALID_CHECKSUM_NAME)
        if not failure_path.exists():
            failure = {
                "schema_version": "dagkv.m2.component_evidence_failure.v1",
                "started_at_utc": started_at,
                "failed_at_utc": _utc_now(),
                "failure": str(exc),
                "acceptance_claimed": False,
            }
            with suppress(ComponentEvidenceError, OSError):
                _write_new(failure_path, _canonical_json(failure))
        if isinstance(exc, ComponentEvidenceError):
            raise
        raise ComponentEvidenceError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run and seal the component contract")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--dagkv-python", required=True, type=Path)
    run.add_argument("--vllm-python", required=True, type=Path)
    run.add_argument("--vllm-root", required=True, type=Path)
    run.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    validate = subparsers.add_parser("validate", help="independently replay evidence")
    validate.add_argument("evidence_dir", type=Path)
    validate.add_argument(
        "--no-external",
        action="store_true",
        help="validate sealed raw files without re-hashing external executables",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            manifest = create_component_evidence(
                output_dir=args.output_dir,
                dagkv_python=args.dagkv_python,
                vllm_python=args.vllm_python,
                vllm_root=args.vllm_root,
                timeout_s=args.timeout_s,
            )
            print(
                f"M2 component evidence passed: {manifest['total_tests']} tests "
                f"at {args.output_dir}"
            )
        else:
            manifest = validate_component_evidence(
                args.evidence_dir,
                verify_external=not args.no_external,
            )
            print(
                f"M2 component evidence replay passed: {manifest['total_tests']} tests"
            )
    except (ComponentEvidenceError, OSError) as exc:
        print(f"M2 component evidence failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
