"""Fail-closed tests for the NVIDIA userspace bundle seal."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import nvidia_driver_userspace_bundle as bundle

DRIVER_VERSION = "580.159.03"
PACKAGE_VERSION = f"{DRIVER_VERSION}-1test1"


@dataclass(frozen=True, slots=True)
class BundleFixture:
    root: Path
    kernel_version: Path
    validation: bundle.BundleValidation | None


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        path = Path(directory)
        path.chmod(0o755)
        for name in (*names, *files):
            child = path / name
            if not child.is_symlink():
                child.chmod(0o755 if child.is_dir() else 0o644)


def _build_deb(source: Path, destination: Path, package: str) -> None:
    control = source / "DEBIAN"
    control.mkdir(parents=True)
    (control / "control").write_text(
        "\n".join(
            [
                f"Package: {package}",
                f"Version: {PACKAGE_VERSION}",
                "Architecture: amd64",
                "Maintainer: DAGKV Test <dagkv@example.invalid>",
                "Description: minimal NVIDIA bundle validator fixture",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "dpkg-deb",
            "--build",
            "--root-owner-group",
            str(source),
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def package_payloads(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    root = tmp_path_factory.mktemp("nvidia-debs")
    payloads: dict[str, bytes] = {}
    for package in ("libnvidia-compute-580", "nvidia-utils-580"):
        source = root / f"source-{package}"
        if package == "libnvidia-compute-580":
            library = source / "usr/lib/x86_64-linux-gnu"
            library.mkdir(parents=True)
            (library / f"libcuda.so.{DRIVER_VERSION}").write_bytes(b"libcuda\n")
            (library / f"libnvidia-ml.so.{DRIVER_VERSION}").write_bytes(b"nvml\n")
            (library / "libcuda.so.1").symlink_to(f"libcuda.so.{DRIVER_VERSION}")
            (library / "libcuda.so").symlink_to("libcuda.so.1")
            (library / "libnvidia-ml.so.1").symlink_to(
                f"libnvidia-ml.so.{DRIVER_VERSION}"
            )
        else:
            binary = source / "usr/bin"
            binary.mkdir(parents=True)
            nvidia_smi = binary / "nvidia-smi"
            nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            nvidia_smi.chmod(0o755)
        output = root / f"{package}_{PACKAGE_VERSION}_amd64.deb"
        _build_deb(source, output, package)
        payloads[output.name] = output.read_bytes()
    return payloads


@pytest.fixture
def bundle_factory(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    package_payloads: dict[str, bytes],
) -> Callable[..., BundleFixture]:
    roots: list[Path] = []

    def build(*, sealed: bool = True) -> BundleFixture:
        number = len(roots) + 1
        root = tmp_path / f"bundle-{number}"
        packages = root / "packages"
        rootfs = root / "rootfs"
        packages.mkdir(parents=True)
        rootfs.mkdir()
        for name, raw in package_payloads.items():
            path = packages / name
            path.write_bytes(raw)
            path.chmod(0o444)
            subprocess.run(
                ["dpkg-deb", "--extract", str(path), str(rootfs)],
                check=True,
                capture_output=True,
            )
        nvidia_smi = rootfs / "usr/bin/nvidia-smi"
        for directory, _, files in os.walk(rootfs, topdown=False):
            path = Path(directory)
            for name in files:
                child = path / name
                if not child.is_symlink():
                    child.chmod(0o555 if child == nvidia_smi else 0o444)
            path.chmod(0o555)
        packages.chmod(0o555)
        kernel_version = tmp_path / f"kernel-version-{number}"
        kernel_version.write_text(
            f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {DRIVER_VERSION}  Test\n",
            encoding="utf-8",
        )
        kernel_version.chmod(0o444)
        roots.append(root)
        validation = (
            bundle.create_bundle_manifest(
                root,
                kernel_version_path=kernel_version,
            )
            if sealed
            else None
        )
        return BundleFixture(root, kernel_version, validation)

    def cleanup() -> None:
        for root in roots:
            _make_tree_writable(root)

    request.addfinalizer(cleanup)
    return build


def _validate(fixture: BundleFixture, **kwargs: object) -> bundle.BundleValidation:
    return bundle.validate_bundle(
        fixture.root,
        kernel_version_path=fixture.kernel_version,
        **kwargs,
    )


def _make_runtime_directory_writable(path: Path) -> None:
    path.chmod(0o755)


def _make_runtime_directory_read_only(path: Path) -> None:
    path.chmod(0o555)


def test_create_and_fresh_validate_closed_bundle(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    assert fixture.validation is not None
    replay = _validate(
        fixture,
        expected_manifest_sha256=fixture.validation.manifest_sha256,
    )

    assert replay.content_digest == fixture.validation.content_digest
    assert replay.kernel_module_version == DRIVER_VERSION
    assert replay.manifest["package_count"] == 2
    assert replay.manifest["runtime_entry_count"] == 11
    assert replay.manifest["runtime_derivation"] == bundle.RUNTIME_DERIVATION
    assert replay.nvidia_smi_path.is_absolute()
    assert replay.nvidia_smi_path == fixture.root / "rootfs/usr/bin/nvidia-smi"
    assert replay.libcuda_path.name == f"libcuda.so.{DRIVER_VERSION}"
    assert replay.runtime.libnvidia_ml.name == f"libnvidia-ml.so.{DRIVER_VERSION}"
    assert replay.stat_snapshot[0].path == "."
    environment = bundle.runtime_environment(
        replay,
        base_environment={"LD_LIBRARY_PATH": "/existing", "TOKEN": "kept"},
    )
    assert environment == {
        "LD_LIBRARY_PATH": f"{replay.library_path}:/existing",
        "TOKEN": "kept",
    }


def test_cli_create_and_validate(
    bundle_factory: Callable[..., BundleFixture], capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = bundle_factory(sealed=False)
    assert (
        bundle.main(
            [
                "--create",
                "--bundle-root",
                str(fixture.root),
                "--kernel-version-path",
                str(fixture.kernel_version),
            ]
        )
        == 0
    )
    manifest = json.loads(
        (fixture.root / bundle.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert (
        bundle.main(
            [
                "--validate",
                "--bundle-root",
                str(fixture.root),
                "--kernel-version-path",
                str(fixture.kernel_version),
                "--expected-manifest-sha256",
                bundle.validate_bundle(
                    fixture.root,
                    kernel_version_path=fixture.kernel_version,
                ).manifest_sha256,
            ]
        )
        == 0
    )
    assert manifest["schema_version"] == bundle.MANIFEST_SCHEMA
    assert "bundle created" in capsys.readouterr().out


def test_create_is_create_only(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    manifest = fixture.root / bundle.MANIFEST_NAME
    before = manifest.read_bytes()

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="overwrite"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert manifest.read_bytes() == before


def test_rejects_package_content_tampering(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    package = next((fixture.root / "packages").iterdir())
    package.chmod(0o644)
    package.write_bytes(package.read_bytes() + b"tamper")
    package.chmod(0o444)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="package closed set"):
        _validate(fixture)


def test_rejects_kernel_module_version_drift(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    fixture.kernel_version.chmod(0o644)
    fixture.kernel_version.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.173.02  Test\n",
        encoding="utf-8",
    )
    fixture.kernel_version.chmod(0o444)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="version differs"):
        _validate(fixture)


def test_create_rejects_package_kernel_mismatch_without_manifest(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory(sealed=False)
    fixture.kernel_version.chmod(0o644)
    fixture.kernel_version.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.173.02  Test\n",
        encoding="utf-8",
    )
    fixture.kernel_version.chmod(0o444)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="does not match"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert not (fixture.root / bundle.MANIFEST_NAME).exists()


@pytest.mark.parametrize("target", ["runtime-file", "runtime-directory", "package"])
def test_rejects_writable_bundle_content(
    bundle_factory: Callable[..., BundleFixture], target: str
) -> None:
    fixture = bundle_factory(sealed=False)
    if target == "runtime-file":
        path = (
            fixture.root
            / f"rootfs/usr/lib/x86_64-linux-gnu/libcuda.so.{DRIVER_VERSION}"
        )
        path.chmod(0o644)
    elif target == "runtime-directory":
        path = fixture.root / "rootfs/usr/lib/x86_64-linux-gnu"
        path.chmod(0o755)
    else:
        path = next((fixture.root / "packages").iterdir())
        path.chmod(0o644)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="writable|read-only"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert not (fixture.root / bundle.MANIFEST_NAME).exists()


@pytest.mark.parametrize("location", ["rootfs", "top-level"])
def test_rejects_extra_files_in_closed_tree(
    bundle_factory: Callable[..., BundleFixture], location: str
) -> None:
    fixture = bundle_factory()
    if location == "rootfs":
        rootfs = fixture.root / "rootfs"
        _make_runtime_directory_writable(rootfs)
        extra = rootfs / "extra"
        extra.write_bytes(b"extra")
        extra.chmod(0o444)
        _make_runtime_directory_read_only(rootfs)
        message = "runtime closed set"
    else:
        fixture.root.chmod(0o755)
        (fixture.root / "extra").write_bytes(b"extra")
        fixture.root.chmod(0o555)
        message = "top-level closed set"

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match=message):
        _validate(fixture)


@pytest.mark.parametrize("location", ["rootfs", "packages"])
def test_rejects_hardlinks(
    bundle_factory: Callable[..., BundleFixture], location: str
) -> None:
    fixture = bundle_factory(sealed=False)
    if location == "rootfs":
        directory = fixture.root / "rootfs/usr/lib/x86_64-linux-gnu"
        source = directory / f"libcuda.so.{DRIVER_VERSION}"
    else:
        directory = fixture.root / "packages"
        source = next(directory.iterdir())
    directory.chmod(0o755)
    os.link(source, directory / "alias")
    directory.chmod(0o555)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="hard link"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )


def test_rejects_special_runtime_nodes(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory(sealed=False)
    rootfs = fixture.root / "rootfs"
    rootfs.chmod(0o755)
    os.mkfifo(rootfs / "fifo")
    rootfs.chmod(0o555)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="special node"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )


@pytest.mark.parametrize("target", ["escape", "dangling"])
def test_rejects_unsafe_runtime_symlinks(
    bundle_factory: Callable[..., BundleFixture], target: str, tmp_path: Path
) -> None:
    fixture = bundle_factory(sealed=False)
    rootfs = fixture.root / "rootfs"
    rootfs.chmod(0o755)
    if target == "escape":
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        (rootfs / "unsafe").symlink_to(os.path.relpath(outside, rootfs))
        message = "escapes"
    else:
        (rootfs / "unsafe").symlink_to("missing")
        message = "dangling"
    rootfs.chmod(0o555)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match=message):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )


def test_rejects_wrong_libcuda_mapping_before_publication(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory(sealed=False)
    library = fixture.root / "rootfs/usr/lib/x86_64-linux-gnu"
    library.chmod(0o755)
    soname = library / "libcuda.so.1"
    soname.unlink()
    soname.symlink_to(f"libnvidia-ml.so.{DRIVER_VERSION}")
    library.chmod(0o555)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="target version"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert not (fixture.root / bundle.MANIFEST_NAME).exists()


def test_rejects_rootfs_content_not_derived_from_packages(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory(sealed=False)
    library = fixture.root / "rootfs/usr/lib/x86_64-linux-gnu"
    libcuda = library / f"libcuda.so.{DRIVER_VERSION}"
    libcuda.chmod(0o644)
    libcuda.write_bytes(b"independently substituted libcuda\n")
    libcuda.chmod(0o444)

    with pytest.raises(
        bundle.NvidiaUserspaceBundleError,
        match="normalized closed union",
    ):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert not (fixture.root / bundle.MANIFEST_NAME).exists()


def test_rejects_manifest_schema_and_digest_tampering(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    manifest_path = fixture.root / bundle.MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["extra"] = "unregistered"
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="fields drifted"):
        _validate(fixture)


def test_rejects_expected_manifest_digest_mismatch(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="digest differs"):
        _validate(fixture, expected_manifest_sha256="0" * 64)


def test_failed_fresh_replay_removes_only_new_manifest(
    bundle_factory: Callable[..., BundleFixture], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = bundle_factory(sealed=False)

    def fail_replay(*_args: object, **_kwargs: object) -> bundle.BundleValidation:
        raise bundle.NvidiaUserspaceBundleError("injected fresh replay failure")

    monkeypatch.setattr(bundle, "validate_bundle", fail_replay)
    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="injected"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert not (fixture.root / bundle.MANIFEST_NAME).exists()


def test_rejects_symlinked_manifest_location(
    bundle_factory: Callable[..., BundleFixture], tmp_path: Path
) -> None:
    fixture = bundle_factory(sealed=False)
    outside = tmp_path / "outside-manifest"
    outside.write_text("preserve\n", encoding="utf-8")
    (fixture.root / bundle.MANIFEST_NAME).symlink_to(outside)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="overwrite"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )

    assert outside.read_text(encoding="utf-8") == "preserve\n"


def test_runtime_environment_does_not_implicitly_copy_process_environment(
    bundle_factory: Callable[..., BundleFixture], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = bundle_factory()
    assert fixture.validation is not None
    monkeypatch.setenv("SECRET_TEST_VALUE", "must-not-be-copied")

    environment = bundle.runtime_environment(fixture.validation)

    assert environment == {"LD_LIBRARY_PATH": str(fixture.validation.library_path)}
    assert "SECRET_TEST_VALUE" not in environment


def test_archive_scan_timeout_covers_stream_read(
    bundle_factory: Callable[..., BundleFixture],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = bundle_factory()
    fake_dpkg = tmp_path / "dpkg-deb-timeout"
    fake_dpkg.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys
import time

if sys.argv[1] == "--field":
    package_path = pathlib.Path(sys.argv[2]).name
    field = sys.argv[3]
    package = (
        "nvidia-utils-580"
        if package_path.startswith("nvidia-utils")
        else "libnvidia-compute-580"
    )
    values = {
        "Package": package,
        "Version": "580.159.03-1test1",
        "Architecture": "amd64",
    }
    print(values[field])
else:
    time.sleep(60)
""",
        encoding="utf-8",
    )
    fake_dpkg.chmod(0o755)
    monkeypatch.setattr(bundle, "ARCHIVE_SCAN_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="timed out"):
        bundle.validate_bundle(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
            dpkg_deb=fake_dpkg,
        )


def test_package_tree_rejects_directories(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory(sealed=False)
    packages = fixture.root / "packages"
    packages.chmod(0o755)
    (packages / "nested").mkdir(mode=0o555)
    packages.chmod(0o555)

    with pytest.raises(bundle.NvidiaUserspaceBundleError, match="direct regular"):
        bundle.create_bundle_manifest(
            fixture.root,
            kernel_version_path=fixture.kernel_version,
        )


def test_manifest_is_read_only_and_package_metadata_is_reconstructed(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    assert fixture.validation is not None
    manifest_path = fixture.root / bundle.MANIFEST_NAME
    assert fixture.root.stat().st_mode & 0o222 == 0
    assert manifest_path.stat().st_mode & 0o222 == 0
    assert {
        (record["package"], record["version"], record["architecture"])
        for record in fixture.validation.manifest["packages"]
    } == {
        ("libnvidia-compute-580", PACKAGE_VERSION, "amd64"),
        ("nvidia-utils-580", PACKAGE_VERSION, "amd64"),
    }


def test_cleanup_helper_handles_read_only_fixture(
    bundle_factory: Callable[..., BundleFixture],
) -> None:
    fixture = bundle_factory()
    _make_tree_writable(fixture.root)
    shutil.rmtree(fixture.root)
    assert not fixture.root.exists()
