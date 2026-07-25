#!/usr/bin/env python3
"""Create and replay a closed NVIDIA Debian userspace bundle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "NVIDIA_USERSPACE_BUNDLE_MANIFEST.json"
MANIFEST_SCHEMA = "dagkv.nvidia_driver_userspace_bundle.v2"
CONTENT_DIGEST_DOMAIN = "dagkv.nvidia_driver_userspace_bundle.content.v2"
RUNTIME_DERIVATION = "dpkg_deb_fsys_tarfile_strip_write_bits_v1"
DEFAULT_KERNEL_VERSION_PATH = Path("/proc/driver/nvidia/version")
MAX_ARCHIVE_ENTRY_BYTES = 1 << 30
MAX_ARCHIVE_TOTAL_BYTES = 4 << 30
MAX_ARCHIVE_ENTRIES = 100_000
ARCHIVE_SCAN_TIMEOUT_SECONDS = 60.0

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "bundle_type",
        "kernel_module_version",
        "package_count",
        "packages",
        "runtime_derivation",
        "runtime_entry_count",
        "runtime_tree",
        "content_digest_algorithm",
        "content_digest",
    }
)
PACKAGE_FIELDS = frozenset(
    {"path", "package", "version", "architecture", "mode", "size", "sha256"}
)
RUNTIME_DIRECTORY_FIELDS = frozenset({"path", "type", "mode"})
RUNTIME_FILE_FIELDS = frozenset({"path", "type", "mode", "size", "sha256"})
RUNTIME_SYMLINK_FIELDS = frozenset({"path", "type", "target"})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PACKAGE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.-]*")
_KERNEL_VERSION_RE = re.compile(rb"\bKernel\s+Module\s+([0-9]+(?:\.[0-9]+){2,})\b")


class NvidiaUserspaceBundleError(RuntimeError):
    """Raised when an NVIDIA userspace bundle fails closed validation."""


@dataclass(frozen=True, slots=True)
class StatEntry:
    """Mutation-sensitive identity for one bundle filesystem entry."""

    path: str
    entry_type: str
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class RuntimeMapping:
    """Absolute paths proven by the validated runtime closed set."""

    rootfs: Path
    library_directory: Path
    nvidia_smi: Path
    libcuda: Path
    libnvidia_ml: Path


@dataclass(frozen=True, slots=True)
class BundleValidation:
    """Validated bundle identity and its final filesystem snapshot."""

    manifest: Mapping[str, Any]
    manifest_sha256: str
    content_digest: str
    kernel_module_version: str
    stat_snapshot: tuple[StatEntry, ...]
    runtime: RuntimeMapping

    @property
    def nvidia_smi_path(self) -> Path:
        """Return the closed-set absolute nvidia-smi path."""

        return self.runtime.nvidia_smi

    @property
    def library_path(self) -> Path:
        """Return the closed-set NVIDIA library directory."""

        return self.runtime.library_directory

    @property
    def libcuda_path(self) -> Path:
        """Return the resolved closed-set libcuda implementation."""

        return self.runtime.libcuda


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NvidiaUserspaceBundleError(message)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _digest_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(before.st_mode), f"expected regular file: {path}")
    _require(before.st_nlink == 1, f"regular file has a hard link: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    _require(
        _stat_identity(before) == _stat_identity(after),
        f"file changed while hashing: {path}",
    )
    return digest.hexdigest()


def _stable_file_bytes(
    path: Path,
    *,
    label: str,
    maximum_size: int,
    require_size_match: bool = True,
) -> bytes:
    before = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
    _require(before.st_size <= maximum_size, f"{label} is unexpectedly large")
    with path.open("rb") as stream:
        raw = stream.read(maximum_size + 1)
    after = path.stat(follow_symlinks=False)
    _require(
        _stat_identity(before) == _stat_identity(after),
        f"{label} changed while being read",
    )
    _require(len(raw) <= maximum_size, f"{label} is unexpectedly large")
    if require_size_match:
        _require(
            len(raw) == before.st_size,
            f"{label} size changed while being read",
        )
    return raw


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "special"


def _snapshot_entry(path: Path, relative: str) -> StatEntry:
    observed = path.stat(follow_symlinks=False)
    kind = _entry_type(observed.st_mode)
    _require(kind != "special", f"bundle contains a special node: {relative}")
    if kind in {"file", "symlink"}:
        _require(
            observed.st_nlink == 1,
            f"bundle entry has a hard link: {relative}",
        )
    return StatEntry(
        path=relative,
        entry_type=kind,
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=observed.st_mode,
        link_count=observed.st_nlink,
        size=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
    )


def _walk_snapshot(root: Path) -> tuple[StatEntry, ...]:
    rows = [_snapshot_entry(root, ".")]

    def visit(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise NvidiaUserspaceBundleError(
                f"cannot enumerate bundle directory: {directory}"
            ) from exc
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            row = _snapshot_entry(path, relative)
            rows.append(row)
            if row.entry_type == "directory":
                visit(path, relative)

    visit(root, "")
    return tuple(rows)


def _validated_root(bundle_root: Path) -> Path:
    absolute = Path(os.path.abspath(bundle_root))
    try:
        observed = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise NvidiaUserspaceBundleError(
            f"bundle root is unavailable: {absolute}"
        ) from exc
    _require(stat.S_ISDIR(observed.st_mode), "bundle root must be a directory")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise NvidiaUserspaceBundleError(
            f"cannot resolve bundle root: {absolute}"
        ) from exc
    _require(resolved == absolute, "bundle root or one of its parents is a symlink")
    return absolute


def _top_level_names(root: Path) -> frozenset[str]:
    try:
        with os.scandir(root) as iterator:
            return frozenset(entry.name for entry in iterator)
    except OSError as exc:
        raise NvidiaUserspaceBundleError(
            f"cannot enumerate bundle root: {root}"
        ) from exc


def _require_directories(root: Path, *, sealed: bool) -> tuple[Path, Path, Path]:
    expected = {"packages", "rootfs"}
    if sealed:
        expected.add(MANIFEST_NAME)
    observed = _top_level_names(root)
    _require(
        observed == expected,
        "bundle top-level closed set differs: "
        f"expected={sorted(expected)} observed={sorted(observed)}",
    )
    if sealed:
        root_value = root.stat(follow_symlinks=False)
        _require(
            stat.S_IMODE(root_value.st_mode) & 0o222 == 0,
            "sealed bundle root directory must be read-only",
        )
    packages = root / "packages"
    runtime = root / "rootfs"
    manifest = root / MANIFEST_NAME
    for path, label in ((packages, "packages"), (runtime, "rootfs")):
        value = path.stat(follow_symlinks=False)
        _require(stat.S_ISDIR(value.st_mode), f"{label} must be a real directory")
        _require(
            stat.S_IMODE(value.st_mode) & 0o222 == 0,
            f"{label} directory must be read-only",
        )
    if sealed:
        value = manifest.stat(follow_symlinks=False)
        _require(stat.S_ISREG(value.st_mode), "bundle manifest must be a regular file")
        _require(value.st_nlink == 1, "bundle manifest has a hard link")
        _require(
            stat.S_IMODE(value.st_mode) & 0o222 == 0,
            "bundle manifest must be read-only",
        )
    return packages, runtime, manifest


def _read_kernel_module_version(path: Path) -> str:
    _require(path.is_absolute(), "kernel version path must be absolute")
    try:
        raw = _stable_file_bytes(
            path,
            label="NVIDIA kernel module version source",
            maximum_size=64 * 1024,
            require_size_match=False,
        )
    except OSError as exc:
        raise NvidiaUserspaceBundleError(
            f"cannot read NVIDIA kernel module version: {path}"
        ) from exc
    matches = _KERNEL_VERSION_RE.findall(raw)
    _require(
        len(matches) == 1,
        "NVIDIA kernel module version source must contain one exact version",
    )
    return matches[0].decode("ascii")


def _resolve_dpkg_deb(executable: str | Path) -> str:
    candidate = str(executable)
    resolved = shutil.which(candidate)
    _require(resolved is not None, f"dpkg-deb executable is unavailable: {candidate}")
    path = Path(resolved)
    value = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(value.st_mode), "dpkg-deb must resolve to a regular file")
    return str(path)


def _deb_field(dpkg_deb: str, path: Path, field: str) -> str:
    try:
        completed = subprocess.run(
            [dpkg_deb, "--field", str(path), field],
            check=False,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C"},
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NvidiaUserspaceBundleError(
            f"cannot inspect Debian package metadata: {path.name}"
        ) from exc
    _require(
        completed.returncode == 0,
        f"dpkg-deb rejected package {path.name}: "
        f"{completed.stderr.decode('utf-8', errors='replace').strip()}",
    )
    try:
        value = completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise NvidiaUserspaceBundleError(
            f"Debian package {field} is not UTF-8: {path.name}"
        ) from exc
    _require(value and "\n" not in value, f"invalid Debian package {field}")
    return value


def _read_deb_metadata(dpkg_deb: str, path: Path) -> tuple[str, str, str]:
    package = _deb_field(dpkg_deb, path, "Package")
    version = _deb_field(dpkg_deb, path, "Version")
    architecture = _deb_field(dpkg_deb, path, "Architecture")
    _require(
        _PACKAGE_TOKEN_RE.fullmatch(package) is not None,
        f"invalid Debian package name: {package}",
    )
    _require(
        _PACKAGE_TOKEN_RE.fullmatch(architecture) is not None,
        f"invalid Debian architecture: {architecture}",
    )
    _require(
        version.isprintable() and not any(character.isspace() for character in version),
        f"invalid Debian package version: {version}",
    )
    return package, version, architecture


def _driver_component(package_version: str) -> str:
    without_epoch = package_version.split(":", maxsplit=1)[-1]
    return without_epoch.split("-", maxsplit=1)[0]


def _scan_packages(
    packages_root: Path, *, dpkg_deb: str, kernel_module_version: str
) -> list[dict[str, Any]]:
    try:
        with os.scandir(packages_root) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        raise NvidiaUserspaceBundleError("cannot enumerate package directory") from exc
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for entry in entries:
        path = Path(entry.path)
        value = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(value.st_mode),
            f"package tree accepts only direct regular files: {entry.name}",
        )
        _require(value.st_nlink == 1, f"package has a hard link: {entry.name}")
        _require(
            stat.S_IMODE(value.st_mode) & 0o222 == 0,
            f"package file is writable: {entry.name}",
        )
        _require(
            entry.name.endswith(".deb"),
            f"package is not a .deb: {entry.name}",
        )
        package, version, architecture = _read_deb_metadata(dpkg_deb, path)
        _require(package not in names, f"duplicate Debian package: {package}")
        names.add(package)
        _require(
            _driver_component(version) == kernel_module_version,
            f"package {package} version does not match loaded kernel module",
        )
        records.append(
            {
                "path": entry.name,
                "package": package,
                "version": version,
                "architecture": architecture,
                "mode": stat.S_IMODE(value.st_mode),
                "size": value.st_size,
                "sha256": _sha256_file(path),
            }
        )
    _require(records, "bundle must contain at least one Debian package")
    _require(
        any(record["package"].startswith("libnvidia-compute-") for record in records),
        "bundle lacks a libnvidia-compute package",
    )
    _require(
        any(record["package"].startswith("nvidia-utils-") for record in records),
        "bundle lacks an nvidia-utils package",
    )
    return records


def _archive_path(value: str, *, package: str) -> str:
    """Return one canonical rootfs-relative path from a Debian data archive."""

    raw = value
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.rstrip("/")
    if raw in {"", "."}:
        return "."
    path = Path(raw)
    _require(
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.as_posix() == raw,
        f"Debian package contains a non-canonical path: {package}:{value}",
    )
    return path.as_posix()


def _hash_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    package: str,
) -> str:
    _require(
        0 <= member.size <= MAX_ARCHIVE_ENTRY_BYTES,
        f"Debian package member is unexpectedly large: {package}:{member.name}",
    )
    stream = archive.extractfile(member)
    _require(
        stream is not None,
        f"cannot read Debian package member: {package}:{member.name}",
    )
    digest = hashlib.sha256()
    observed_size = 0
    with stream:
        while chunk := stream.read(1024 * 1024):
            observed_size += len(chunk)
            _require(
                observed_size <= member.size,
                f"Debian package member exceeds its declared size: "
                f"{package}:{member.name}",
            )
            digest.update(chunk)
    _require(
        observed_size == member.size,
        f"Debian package member size differs: {package}:{member.name}",
    )
    return digest.hexdigest()


def _archive_member_record(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    package: str,
) -> dict[str, Any]:
    path = _archive_path(member.name, package=package)
    if member.isdir():
        return {
            "path": path,
            "type": "directory",
            "mode": member.mode & 0o555,
        }
    if member.isfile():
        return {
            "path": path,
            "type": "file",
            "mode": member.mode & 0o555,
            "size": member.size,
            "sha256": _hash_archive_member(archive, member, package=package),
        }
    if member.issym():
        _require(
            isinstance(member.linkname, str) and member.linkname,
            f"Debian package symlink has an empty target: {package}:{member.name}",
        )
        return {"path": path, "type": "symlink", "target": member.linkname}
    if member.islnk():
        raise NvidiaUserspaceBundleError(
            f"Debian package contains a hard link: {package}:{member.name}"
        )
    raise NvidiaUserspaceBundleError(
        f"Debian package contains a special archive member: {package}:{member.name}"
    )


def _terminate_archive_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with suppress(OSError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _scan_deb_runtime(
    path: Path,
    *,
    package: str,
    dpkg_deb: str,
) -> list[dict[str, Any]]:
    """Stream one data archive and reconstruct its read-only extraction tree."""

    process: subprocess.Popen[bytes] | None = None
    timed_out = threading.Event()
    timer: threading.Timer | None = None
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(
                [dpkg_deb, "--fsys-tarfile", str(path)],
                stdout=subprocess.PIPE,
                stderr=errors,
                env={**os.environ, "LC_ALL": "C"},
                start_new_session=True,
            )

            def expire_scan() -> None:
                assert process is not None
                if process.poll() is None:
                    timed_out.set()
                    _terminate_archive_process(process)

            timer = threading.Timer(
                ARCHIVE_SCAN_TIMEOUT_SECONDS,
                expire_scan,
            )
            timer.daemon = True
            timer.start()
            _require(process.stdout is not None, "dpkg-deb stdout pipe is missing")
            records: list[dict[str, Any]] = []
            total_size = 0
            try:
                with (
                    process.stdout,
                    tarfile.open(
                        fileobj=process.stdout,
                        mode="r|*",
                    ) as archive,
                ):
                    for member in archive:
                        _require(
                            len(records) < MAX_ARCHIVE_ENTRIES,
                            f"Debian package has too many archive entries: {package}",
                        )
                        total_size += member.size
                        _require(
                            total_size <= MAX_ARCHIVE_TOTAL_BYTES,
                            f"Debian package expands beyond the validation limit: "
                            f"{package}",
                        )
                        records.append(
                            _archive_member_record(
                                archive,
                                member,
                                package=package,
                            )
                        )
            except (OSError, tarfile.TarError) as exc:
                if timed_out.is_set():
                    raise NvidiaUserspaceBundleError(
                        f"dpkg-deb payload scan timed out: {package}"
                    ) from exc
                raise NvidiaUserspaceBundleError(
                    f"cannot reconstruct Debian package payload: {package}"
                ) from exc
            try:
                return_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired as exc:
                raise NvidiaUserspaceBundleError(
                    f"dpkg-deb payload scan timed out: {package}"
                ) from exc
            _require(
                not timed_out.is_set(),
                f"dpkg-deb payload scan timed out: {package}",
            )
            errors.seek(0, os.SEEK_END)
            error_size = errors.tell()
            _require(
                error_size <= 1024 * 1024,
                f"dpkg-deb emitted excessive diagnostics: {package}",
            )
            errors.seek(0)
            diagnostics = errors.read().decode("utf-8", errors="replace").strip()
            _require(
                return_code == 0,
                f"dpkg-deb payload scan failed for {package}: {diagnostics}",
            )
            _require(records, f"Debian package payload is empty: {package}")
            return records
        except OSError as exc:
            raise NvidiaUserspaceBundleError(
                f"cannot launch dpkg-deb payload scan: {package}"
            ) from exc
        finally:
            if timer is not None:
                timer.cancel()
            if process is not None:
                _terminate_archive_process(process)


def _derived_runtime_from_packages(
    packages_root: Path,
    packages: Sequence[Mapping[str, Any]],
    *,
    dpkg_deb: str,
) -> list[dict[str, Any]]:
    """Reconstruct the exact union produced by extracting the sealed packages."""

    union: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    for package in packages:
        package_name = str(package["package"])
        for record in _scan_deb_runtime(
            packages_root / str(package["path"]),
            package=package_name,
            dpkg_deb=dpkg_deb,
        ):
            path = str(record["path"])
            existing = union.get(path)
            if existing is None:
                union[path] = record
                owners[path] = package_name
                continue
            _require(
                record["type"] == existing["type"] == "directory"
                and record == existing,
                "Debian package payloads overlap outside an identical directory: "
                f"{path} ({owners[path]}, {package_name})",
            )
    _require("." in union, "Debian package union lacks its root directory")
    return sorted(
        union.values(),
        key=lambda record: (
            () if record["path"] == "." else Path(str(record["path"])).parts
        ),
    )


def _runtime_mode(path: Path, value: os.stat_result, *, relative: str) -> int:
    mode = stat.S_IMODE(value.st_mode)
    _require(
        mode & 0o222 == 0,
        f"runtime entry is writable: {relative}",
    )
    return mode


def _stable_symlink_target(path: Path, *, relative: str) -> str:
    before = path.stat(follow_symlinks=False)
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise NvidiaUserspaceBundleError(
            f"cannot read runtime symlink: {relative}"
        ) from exc
    after = path.stat(follow_symlinks=False)
    _require(
        _stat_identity(before) == _stat_identity(after),
        f"runtime symlink changed while being read: {relative}",
    )
    _require(before.st_nlink == 1, f"runtime symlink has a hard link: {relative}")
    return target


def _validate_runtime_symlink(
    runtime_root: Path, path: Path, *, relative: str, target: str
) -> None:
    target_path = Path(target)
    _require(not target_path.is_absolute(), f"runtime symlink is absolute: {relative}")
    try:
        resolved = (path.parent / target_path).resolve(strict=True)
        root = runtime_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NvidiaUserspaceBundleError(
            f"runtime symlink is dangling or cyclic: {relative}"
        ) from exc
    _require(
        resolved.is_relative_to(root),
        f"runtime symlink escapes rootfs: {relative}",
    )
    resolved_value = resolved.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(resolved_value.st_mode),
        f"runtime symlink must resolve to a regular file: {relative}",
    )


def _scan_runtime(runtime_root: Path) -> list[dict[str, Any]]:
    root_value = runtime_root.stat(follow_symlinks=False)
    records: list[dict[str, Any]] = [
        {
            "path": ".",
            "type": "directory",
            "mode": _runtime_mode(runtime_root, root_value, relative="."),
        }
    ]

    def visit(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise NvidiaUserspaceBundleError(
                f"cannot enumerate runtime directory: {directory}"
            ) from exc
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            path = Path(entry.path)
            value = path.stat(follow_symlinks=False)
            kind = _entry_type(value.st_mode)
            _require(
                kind != "special",
                f"runtime tree contains a special node: {relative}",
            )
            if kind == "directory":
                records.append(
                    {
                        "path": relative,
                        "type": kind,
                        "mode": _runtime_mode(path, value, relative=relative),
                    }
                )
                visit(path, relative)
            elif kind == "file":
                _require(
                    value.st_nlink == 1,
                    f"runtime file has a hard link: {relative}",
                )
                records.append(
                    {
                        "path": relative,
                        "type": kind,
                        "mode": _runtime_mode(path, value, relative=relative),
                        "size": value.st_size,
                        "sha256": _sha256_file(path),
                    }
                )
            else:
                target = _stable_symlink_target(path, relative=relative)
                _validate_runtime_symlink(
                    runtime_root,
                    path,
                    relative=relative,
                    target=target,
                )
                records.append({"path": relative, "type": kind, "target": target})

    visit(runtime_root, "")
    return records


def _resolved_runtime_library(
    runtime_root: Path,
    path: Path,
    *,
    label: str,
    expected_name: str,
) -> Path:
    value = path.stat(follow_symlinks=False)
    _require(stat.S_ISLNK(value.st_mode), f"{label} must be a symlink")
    target = _stable_symlink_target(
        path,
        relative=path.relative_to(runtime_root).as_posix(),
    )
    _validate_runtime_symlink(
        runtime_root,
        path,
        relative=path.relative_to(runtime_root).as_posix(),
        target=target,
    )
    resolved = path.resolve(strict=True)
    _require(resolved.name == expected_name, f"{label} target version differs")
    return resolved


def _runtime_mapping(
    runtime_root: Path, *, kernel_module_version: str
) -> RuntimeMapping:
    library_directory = runtime_root / "usr/lib/x86_64-linux-gnu"
    nvidia_smi = runtime_root / "usr/bin/nvidia-smi"
    for path, label in (
        (library_directory, "NVIDIA library directory"),
        (nvidia_smi.parent, "NVIDIA binary directory"),
    ):
        value = path.stat(follow_symlinks=False)
        _require(stat.S_ISDIR(value.st_mode), f"{label} is missing")
    smi_value = nvidia_smi.stat(follow_symlinks=False)
    _require(stat.S_ISREG(smi_value.st_mode), "nvidia-smi is not a regular file")
    _require(
        stat.S_IMODE(smi_value.st_mode) & 0o111 != 0,
        "nvidia-smi is not executable",
    )
    libcuda = _resolved_runtime_library(
        runtime_root,
        library_directory / "libcuda.so.1",
        label="libcuda.so.1",
        expected_name=f"libcuda.so.{kernel_module_version}",
    )
    libnvidia_ml = _resolved_runtime_library(
        runtime_root,
        library_directory / "libnvidia-ml.so.1",
        label="libnvidia-ml.so.1",
        expected_name=f"libnvidia-ml.so.{kernel_module_version}",
    )
    unversioned = library_directory / "libcuda.so"
    unversioned_value = unversioned.stat(follow_symlinks=False)
    _require(
        stat.S_ISLNK(unversioned_value.st_mode),
        "unversioned libcuda.so must be a symlink",
    )
    _require(
        unversioned.resolve(strict=True) == libcuda,
        "unversioned libcuda.so does not map to the validated implementation",
    )
    return RuntimeMapping(
        rootfs=runtime_root,
        library_directory=library_directory,
        nvidia_smi=nvidia_smi,
        libcuda=libcuda,
        libnvidia_ml=libnvidia_ml,
    )


def runtime_environment(
    validation: BundleValidation,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an environment that resolves CUDA/NVML from the validated bundle."""

    environment = dict(base_environment or {})
    library_path = str(validation.runtime.library_directory)
    inherited = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library_path}:{inherited}" if inherited else library_path
    )
    return environment


def _content_digest(
    kernel_module_version: str,
    packages: Sequence[Mapping[str, Any]],
    runtime_derivation: str,
    runtime_tree: Sequence[Mapping[str, Any]],
) -> str:
    return _digest_json(
        {
            "domain": CONTENT_DIGEST_DOMAIN,
            "kernel_module_version": kernel_module_version,
            "packages": list(packages),
            "runtime_derivation": runtime_derivation,
            "runtime_tree": list(runtime_tree),
        }
    )


def _validate_timestamp(value: Any) -> None:
    _require(isinstance(value, str), "manifest timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise NvidiaUserspaceBundleError("manifest timestamp is invalid") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        "manifest timestamp must include a timezone",
    )
    _require(parsed <= datetime.now(UTC), "manifest timestamp is in the future")


def _validate_sha256(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return value


def _validate_package_records(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and value, "manifest packages must be nonempty")
    records: list[dict[str, Any]] = []
    previous = ""
    names: set[str] = set()
    for record in value:
        _require(isinstance(record, dict), "manifest package entry must be an object")
        _require(set(record) == PACKAGE_FIELDS, "manifest package fields drifted")
        path = record["path"]
        _require(
            isinstance(path, str)
            and path
            and Path(path).name == path
            and path.endswith(".deb"),
            "manifest package path is invalid",
        )
        _require(path > previous, "manifest packages are not strictly sorted")
        previous = path
        for field in ("package", "version", "architecture"):
            _require(
                isinstance(record[field], str) and record[field],
                f"manifest package {field} is invalid",
            )
        _require(record["package"] not in names, "manifest package name is duplicated")
        names.add(record["package"])
        for field in ("mode", "size"):
            _require(
                type(record[field]) is int and record[field] >= 0,
                f"manifest package {field} is invalid",
            )
        _require(
            record["mode"] <= 0o7777 and record["mode"] & 0o222 == 0,
            "manifest package mode is invalid or writable",
        )
        _validate_sha256(record["sha256"], label="manifest package digest")
        records.append(dict(record))
    return records


def _valid_runtime_path(value: Any) -> bool:
    if value == ".":
        return True
    if not isinstance(value, str) or not value or value.startswith("/"):
        return False
    parts = Path(value).parts
    return all(part not in {"", ".", ".."} for part in parts)


def _validate_runtime_records(value: Any) -> list[dict[str, Any]]:
    _require(
        isinstance(value, list) and value,
        "manifest runtime tree must be nonempty",
    )
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    previous: tuple[str, ...] | None = None
    for record in value:
        _require(isinstance(record, dict), "manifest runtime entry must be an object")
        kind = record.get("type")
        expected_fields = {
            "directory": RUNTIME_DIRECTORY_FIELDS,
            "file": RUNTIME_FILE_FIELDS,
            "symlink": RUNTIME_SYMLINK_FIELDS,
        }.get(kind)
        _require(expected_fields is not None, "manifest runtime type is invalid")
        _require(set(record) == expected_fields, "manifest runtime fields drifted")
        path = record["path"]
        _require(_valid_runtime_path(path), "manifest runtime path is invalid")
        _require(path not in paths, "manifest runtime path is duplicated")
        paths.add(path)
        ordering = () if path == "." else Path(path).parts
        _require(
            previous is None or ordering > previous,
            "manifest runtime tree is not strictly sorted",
        )
        previous = ordering
        if kind in {"directory", "file"}:
            _require(
                type(record["mode"]) is int
                and 0 <= record["mode"] <= 0o7777
                and record["mode"] & 0o222 == 0,
                "manifest runtime mode is invalid or writable",
            )
        if kind == "file":
            _require(
                type(record["size"]) is int and record["size"] >= 0,
                "manifest runtime file size is invalid",
            )
            _validate_sha256(record["sha256"], label="manifest runtime digest")
        if kind == "symlink":
            _require(
                isinstance(record["target"], str) and record["target"],
                "manifest runtime symlink target is invalid",
            )
        records.append(dict(record))
    _require(
        records[0].get("path") == "." and records[0].get("type") == "directory",
        "manifest runtime root entry is missing",
    )
    return records


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = _stable_file_bytes(
            path,
            label="NVIDIA userspace bundle manifest",
            maximum_size=64 * 1024 * 1024,
        )
    except OSError as exc:
        raise NvidiaUserspaceBundleError(
            f"cannot read bundle manifest: {path}"
        ) from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NvidiaUserspaceBundleError("bundle manifest is not valid JSON") from exc
    _require(isinstance(payload, dict), "bundle manifest must be a JSON object")
    _require(raw == _canonical_json(payload), "bundle manifest is not canonical JSON")
    return payload, raw


def _validate_manifest_shape(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(set(payload) == MANIFEST_FIELDS, "bundle manifest fields drifted")
    _require(payload["schema_version"] == MANIFEST_SCHEMA, "wrong bundle schema")
    _require(
        payload["bundle_type"] == "nvidia_debian_userspace_exact",
        "wrong bundle type",
    )
    _validate_timestamp(payload["created_at_utc"])
    _require(
        isinstance(payload["kernel_module_version"], str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,}", payload["kernel_module_version"]),
        "manifest kernel module version is invalid",
    )
    packages = _validate_package_records(payload["packages"])
    runtime = _validate_runtime_records(payload["runtime_tree"])
    _require(
        type(payload["package_count"]) is int
        and payload["package_count"] == len(packages),
        "manifest package count differs",
    )
    _require(
        type(payload["runtime_entry_count"]) is int
        and payload["runtime_entry_count"] == len(runtime),
        "manifest runtime entry count differs",
    )
    _require(
        payload["runtime_derivation"] == RUNTIME_DERIVATION,
        "manifest runtime derivation differs",
    )
    _require(
        payload["content_digest_algorithm"] == "sha256",
        "manifest content digest algorithm differs",
    )
    _validate_sha256(payload["content_digest"], label="manifest content digest")
    expected_digest = _content_digest(
        payload["kernel_module_version"],
        packages,
        payload["runtime_derivation"],
        runtime,
    )
    _require(
        payload["content_digest"] == expected_digest,
        "manifest content digest is internally inconsistent",
    )
    return packages, runtime


def validate_bundle(
    bundle_root: Path,
    *,
    kernel_version_path: Path = DEFAULT_KERNEL_VERSION_PATH,
    expected_manifest_sha256: str | None = None,
    dpkg_deb: str | Path = "dpkg-deb",
) -> BundleValidation:
    """Reconstruct and validate one closed NVIDIA userspace bundle."""

    root = _validated_root(bundle_root)
    packages_root, runtime_root, manifest_path = _require_directories(root, sealed=True)
    before = _walk_snapshot(root)
    payload, raw = _load_manifest(manifest_path)
    packages, runtime = _validate_manifest_shape(payload)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None:
        expected = _validate_sha256(
            expected_manifest_sha256,
            label="expected manifest digest",
        )
        _require(manifest_sha256 == expected, "bundle manifest digest differs")
    observed_kernel = _read_kernel_module_version(kernel_version_path)
    _require(
        observed_kernel == payload["kernel_module_version"],
        "loaded NVIDIA kernel module version differs from bundle",
    )
    executable = _resolve_dpkg_deb(dpkg_deb)
    observed_packages = _scan_packages(
        packages_root,
        dpkg_deb=executable,
        kernel_module_version=observed_kernel,
    )
    observed_runtime = _scan_runtime(runtime_root)
    _require(observed_packages == packages, "package closed set or content differs")
    _require(observed_runtime == runtime, "runtime closed set or content differs")
    derived_runtime = _derived_runtime_from_packages(
        packages_root,
        observed_packages,
        dpkg_deb=executable,
    )
    _require(
        derived_runtime == observed_runtime,
        "rootfs is not the normalized closed union of the sealed Debian packages",
    )
    observed_digest = _content_digest(
        observed_kernel,
        observed_packages,
        RUNTIME_DERIVATION,
        observed_runtime,
    )
    _require(observed_digest == payload["content_digest"], "bundle content differs")
    _require(
        _read_kernel_module_version(kernel_version_path) == observed_kernel,
        "loaded NVIDIA kernel module changed during validation",
    )
    after = _walk_snapshot(root)
    _require(before == after, "bundle filesystem changed during validation")
    mapping = _runtime_mapping(
        runtime_root,
        kernel_module_version=observed_kernel,
    )
    _require(after == _walk_snapshot(root), "bundle changed during path validation")
    return BundleValidation(
        manifest=payload,
        manifest_sha256=manifest_sha256,
        content_digest=observed_digest,
        kernel_module_version=observed_kernel,
        stat_snapshot=after,
        runtime=mapping,
    )


def _created_timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(UTC)
    _require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "creation timestamp must include a timezone",
    )
    timestamp = timestamp.astimezone(UTC)
    _require(timestamp <= datetime.now(UTC), "creation timestamp is in the future")
    return timestamp.isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only(path: Path, raw: bytes) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise NvidiaUserspaceBundleError(
            f"refusing to overwrite existing bundle manifest: {path}"
        ) from exc
    initial = os.fstat(descriptor)
    identity = initial.st_dev, initial.st_ino
    try:
        written = 0
        while written < len(raw):
            count = os.write(descriptor, raw[written:])
            if count == 0:
                raise OSError("zero-byte manifest write")
            written += count
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        _cleanup_created_inode(path, identity=identity)
        raise
    finally:
        with suppress(OSError):
            os.close(descriptor)
    _fsync_directory(path.parent)
    return observed.st_dev, observed.st_ino


def _cleanup_created_inode(path: Path, *, identity: tuple[int, int]) -> None:
    try:
        value = path.stat(follow_symlinks=False)
        if (
            stat.S_ISREG(value.st_mode)
            and (value.st_dev, value.st_ino) == identity
            and value.st_nlink == 1
        ):
            path.unlink()
            _fsync_directory(path.parent)
    except (NvidiaUserspaceBundleError, OSError):
        return


def _cleanup_failed_create(
    path: Path, *, identity: tuple[int, int], expected_sha256: str
) -> None:
    try:
        value = path.stat(follow_symlinks=False)
        if (
            stat.S_ISREG(value.st_mode)
            and (value.st_dev, value.st_ino) == identity
            and value.st_nlink == 1
            and _sha256_file(path) == expected_sha256
        ):
            path.unlink()
            _fsync_directory(path.parent)
    except (NvidiaUserspaceBundleError, OSError):
        return


def _open_stable_directory(path: Path, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        _require(
            (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino),
            "bundle root changed while opening its publication descriptor",
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def create_bundle_manifest(
    bundle_root: Path,
    *,
    kernel_version_path: Path = DEFAULT_KERNEL_VERSION_PATH,
    dpkg_deb: str | Path = "dpkg-deb",
    created_at: datetime | None = None,
) -> BundleValidation:
    """Publish a create-only manifest and immediately replay it from disk."""

    root = _validated_root(bundle_root)
    manifest_path = root / MANIFEST_NAME
    _require(
        not os.path.lexists(manifest_path),
        f"refusing to overwrite existing bundle manifest: {manifest_path}",
    )
    root_value = root.stat(follow_symlinks=False)
    original_root_mode = stat.S_IMODE(root_value.st_mode)
    _require(
        original_root_mode & 0o222 != 0,
        "unsealed bundle root must be writable for create-only publication",
    )
    packages_root, runtime_root, manifest_path = _require_directories(
        root, sealed=False
    )
    before = _walk_snapshot(root)
    kernel_version = _read_kernel_module_version(kernel_version_path)
    executable = _resolve_dpkg_deb(dpkg_deb)
    packages = _scan_packages(
        packages_root,
        dpkg_deb=executable,
        kernel_module_version=kernel_version,
    )
    runtime = _scan_runtime(runtime_root)
    _runtime_mapping(runtime_root, kernel_module_version=kernel_version)
    derived_runtime = _derived_runtime_from_packages(
        packages_root,
        packages,
        dpkg_deb=executable,
    )
    _require(
        derived_runtime == runtime,
        "rootfs is not the normalized closed union of the sealed Debian packages",
    )
    _require(
        _read_kernel_module_version(kernel_version_path) == kernel_version,
        "loaded NVIDIA kernel module changed during manifest creation",
    )
    _require(before == _walk_snapshot(root), "bundle changed during manifest creation")
    content_digest = _content_digest(
        kernel_version,
        packages,
        RUNTIME_DERIVATION,
        runtime,
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": _created_timestamp(created_at),
        "bundle_type": "nvidia_debian_userspace_exact",
        "kernel_module_version": kernel_version,
        "package_count": len(packages),
        "packages": packages,
        "runtime_derivation": RUNTIME_DERIVATION,
        "runtime_entry_count": len(runtime),
        "runtime_tree": runtime,
        "content_digest_algorithm": "sha256",
        "content_digest": content_digest,
    }
    raw = _canonical_json(payload)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    root_descriptor = _open_stable_directory(root, root_value)
    identity: tuple[int, int] | None = None
    try:
        identity = _write_create_only(manifest_path, raw)
        os.fchmod(root_descriptor, original_root_mode & ~0o222)
        os.fsync(root_descriptor)
        return validate_bundle(
            root,
            kernel_version_path=kernel_version_path,
            expected_manifest_sha256=raw_sha256,
            dpkg_deb=executable,
        )
    except Exception:
        if identity is not None:
            with suppress(OSError):
                os.fchmod(root_descriptor, original_root_mode)
                os.fsync(root_descriptor)
            _cleanup_failed_create(
                manifest_path,
                identity=identity,
                expected_sha256=raw_sha256,
            )
        raise
    finally:
        os.close(root_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--kernel-version-path",
        type=Path,
        default=DEFAULT_KERNEL_VERSION_PATH,
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--dpkg-deb", default="dpkg-deb")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.create and args.expected_manifest_sha256 is not None:
        parser.error("--expected-manifest-sha256 is valid only with --validate")
    try:
        if args.create:
            result = create_bundle_manifest(
                args.bundle_root,
                kernel_version_path=args.kernel_version_path,
                dpkg_deb=args.dpkg_deb,
            )
            action = "created"
        else:
            result = validate_bundle(
                args.bundle_root,
                kernel_version_path=args.kernel_version_path,
                expected_manifest_sha256=args.expected_manifest_sha256,
                dpkg_deb=args.dpkg_deb,
            )
            action = "validated"
    except (NvidiaUserspaceBundleError, OSError) as exc:
        print(f"NVIDIA userspace bundle validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"NVIDIA userspace bundle {action}: {args.bundle_root.resolve()} "
        f"manifest_sha256={result.manifest_sha256} "
        f"content_digest={result.content_digest} "
        f"kernel_module={result.kernel_module_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
