from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from tools import run_m2_vllm_abba as runner_module
from tools.run_m2_vllm_abba import (
    CALIBRATION_COHORT_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    MIN_CALIBRATION_RUNS,
    PROTOCOL_SCHEMA,
    TOLERANCE_DERIVATION,
    TOLERANCE_SCHEMA,
    M2ValidationError,
    _transfer_params,
    build_llm_kwargs,
    compare_logit_vectors,
    dense_logits_from_logprobs,
    load_calibration_cohort,
    load_frozen_tolerance,
    read_jsonl_strict,
    resolve_native_transfer_request_id,
    validate_abba_transfer_chain,
    validate_diagnostic_trace_closed_set,
    validate_diagnostic_transfer,
    validate_native_trace_closed_set,
    validate_native_transfer,
    validate_prefetch_result,
    validate_prompt_tokens,
)


@dataclass
class _Value:
    logprob: float


@dataclass
class _Flat:
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]

    def __len__(self) -> int:
        return len(self.start_indices)


def _capture_spawn_loader(queue: object) -> None:
    queue.put(os.environ.get("LD_LIBRARY_PATH"))


def _endpoint(tier: str, slot: int, generation: int, digest: str) -> dict:
    return {
        "tier": tier,
        "physical_slot": slot,
        "allocation_generation": generation,
        "digest": digest,
    }


def _terminal(
    *,
    request_id: str,
    direction: str,
    source: dict,
    target: dict,
    job_id: int = 1,
) -> dict:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "event": "terminal",
        "status": "completed",
        "run_id": "run",
        "job_id": job_id,
        "request_id": request_id,
        "direction": direction,
        "payload_bytes": 4096,
        "reported_bytes": 4096,
        "framing": "DAGKV_PAYLOAD_V1",
        "source": source,
        "target": target,
        "failure_reason": None,
    }


def _write_parent_inputs(tmp_path: Path) -> SimpleNamespace:
    fingerprint = "f" * 64
    implementation_sha256 = "a" * 64
    runs = [
        {
            "run_id": f"run-{index}",
            "result_sha256": hashlib.sha256(f"result-{index}".encode()).hexdigest(),
            "provenance_sha256": hashlib.sha256(
                f"provenance-{index}".encode()
            ).hexdigest(),
            "sha256sums_sha256": hashlib.sha256(
                f"checksums-{index}".encode()
            ).hexdigest(),
        }
        for index in range(MIN_CALIBRATION_RUNS)
    ]
    cohort_payload = {
        "schema_version": CALIBRATION_COHORT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "pilot_excluded": True,
        "run_count": MIN_CALIBRATION_RUNS,
        "all_passed": True,
        "failures": [],
        "observed_max_abs_error": 0.109375,
        "formal_atol": 0.125,
        "formal_rtol": 0.0,
        "reproducibility_fingerprint": fingerprint,
        "runs": runs,
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort_payload), encoding="utf-8")
    tolerance_payload = {
        "schema_version": TOLERANCE_SCHEMA,
        "frozen": True,
        "frozen_at_utc": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        "atol": 0.125,
        "rtol": 0.0,
        "calibration_manifest_sha256": hashlib.sha256(
            cohort_path.read_bytes()
        ).hexdigest(),
        "reproducibility_fingerprint": fingerprint,
        "calibration_run_count": MIN_CALIBRATION_RUNS,
        "derivation": TOLERANCE_DERIVATION,
    }
    tolerance_path = tmp_path / "tolerance.json"
    tolerance_path.write_text(json.dumps(tolerance_payload), encoding="utf-8")
    return SimpleNamespace(
        cohort_path=cohort_path,
        cohort_payload=cohort_payload,
        fingerprint=fingerprint,
        implementation_sha256=implementation_sha256,
        runs=runs,
        tolerance_path=tolerance_path,
        tolerance_payload=tolerance_payload,
    )


def _mutate_parent_file(path: Path, mutation: str) -> None:
    before = path.stat()
    if mutation == "content":
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        assert mutation == "time"
        os.utime(
            path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
        )


def test_prompt_is_exactly_one_block_plus_one() -> None:
    assert validate_prompt_tokens(range(17)) == tuple(range(17))
    with pytest.raises(M2ValidationError, match="exactly 17"):
        validate_prompt_tokens(range(16))
    with pytest.raises(M2ValidationError, match="non-negative"):
        validate_prompt_tokens([*range(16), -1])


def test_rejects_loader_injection_even_when_value_is_empty() -> None:
    runner_module._reject_loader_injection({"LD_LIBRARY_PATH": "/validated"})

    with pytest.raises(M2ValidationError, match="LD_PRELOAD"):
        runner_module._reject_loader_injection({"LD_PRELOAD": ""})
    with pytest.raises(M2ValidationError, match="LD_AUDIT"):
        runner_module._reject_loader_injection({"LD_AUDIT": "/tmp/audit.so"})


def _dependency_fixture(*pairs: tuple[str, str]) -> dict[str, object]:
    packages = [
        {"name": name, "version": version} for name, version in sorted(set(pairs))
    ]
    return {
        "packages": packages,
        "manifest_sha256": runner_module._canonical_digest(packages),
    }


def test_runtime_import_boundary_restores_exact_loader_and_records_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_prefix = tmp_path / "venv"
    cv2_dir = python_prefix / "lib/python3.12/site-packages/cv2"
    cv2_dir.mkdir(parents=True)
    cv2_file = cv2_dir / "__init__.py"
    cv2_file.write_text("# fixture\n", encoding="utf-8")
    setuptools_dir = python_prefix / "lib/python3.12/site-packages/setuptools"
    vendor = setuptools_dir / "_vendor"
    vendor.mkdir(parents=True)
    setuptools_file = setuptools_dir / "__init__.py"
    setuptools_file.write_text("# fixture\n", encoding="utf-8")
    bundle_library = tmp_path / "bundle/rootfs/usr/lib/x86_64-linux-gnu"
    bundle_library.mkdir(parents=True)
    frozen = f"{bundle_library}:/usr/local/cuda/lib64"
    launch_sys_path = [str(python_prefix / "site")]
    effective_sys_path = [*launch_sys_path, str(vendor.resolve())]
    base = _dependency_fixture(("base", "1"))
    effective = _dependency_fixture(("base", "1"), ("vendored", "2"))

    monkeypatch.setattr(runner_module.sys, "prefix", str(python_prefix))
    monkeypatch.setattr(runner_module.sys, "path", effective_sys_path)
    monkeypatch.setitem(
        runner_module.sys.modules, "cv2", SimpleNamespace(__file__=str(cv2_file))
    )
    monkeypatch.setitem(
        runner_module.sys.modules,
        "setuptools",
        SimpleNamespace(__file__=str(setuptools_file)),
    )
    monkeypatch.setattr(
        runner_module,
        "_dependency_origins",
        lambda pairs: {pair: {vendor.resolve()} for pair in pairs},
    )
    expected_cv2 = runner_module._expected_cv2_loader_path()
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{expected_cv2}:{frozen}")

    capture = runner_module._restore_runtime_import_boundary(
        nvidia_bundle=SimpleNamespace(library_path=bundle_library),
        frozen_ld_library_path=frozen,
        launch_sys_path=launch_sys_path,
        base_dependencies=base,
        effective_dependencies=effective,
    )

    assert os.environ["LD_LIBRARY_PATH"] == frozen
    assert capture["loader_environment"]["observed_after_imports"] == (
        f"{expected_cv2}:{frozen}"
    )
    assert capture["loader_environment"]["restored_before_spawn"] == frozen
    assert capture["sys_path"]["added_paths"] == [str(vendor.resolve())]
    assert capture["dependencies"]["added"] == [
        {"name": "vendored", "version": "2", "origin": str(vendor.resolve())}
    ]


@pytest.mark.parametrize(
    ("frozen_prefix", "observed_prefix", "injection", "message"),
    [
        ("bundle", "/unknown", None, "outside the frozen OpenCV policy"),
        ("/host", "/host", None, "differs from the validated NVIDIA bundle"),
        ("bundle", "bundle", "LD_PRELOAD", "loader injection"),
        ("bundle", "bundle", "LD_AUDIT", "loader injection"),
    ],
)
def test_runtime_import_boundary_rejects_unfrozen_loader_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_prefix: str,
    observed_prefix: str,
    injection: str | None,
    message: str,
) -> None:
    bundle_library = tmp_path / "bundle"
    frozen = f"{bundle_library if frozen_prefix == 'bundle' else frozen_prefix}:/cuda"
    observed = (
        f"{bundle_library if observed_prefix == 'bundle' else observed_prefix}:/cuda"
    )
    dependencies = _dependency_fixture(("base", "1"))
    monkeypatch.delitem(runner_module.sys.modules, "cv2", raising=False)
    monkeypatch.delitem(runner_module.sys.modules, "setuptools", raising=False)
    monkeypatch.setattr(runner_module.sys, "path", ["/base"])
    monkeypatch.setenv("LD_LIBRARY_PATH", observed)
    if injection is not None:
        monkeypatch.setenv(injection, "/injected.so")

    with pytest.raises(M2ValidationError, match=message):
        runner_module._restore_runtime_import_boundary(
            nvidia_bundle=SimpleNamespace(library_path=bundle_library),
            frozen_ld_library_path=frozen,
            launch_sys_path=["/base"],
            base_dependencies=dependencies,
            effective_dependencies=dependencies,
        )

    assert os.environ["LD_LIBRARY_PATH"] == frozen


def test_runtime_import_loader_is_restored_when_import_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = f"{tmp_path / 'bundle'}:/cuda"
    monkeypatch.setenv("LD_LIBRARY_PATH", frozen)
    monkeypatch.setattr(runner_module, "_prepare_import_path", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "_dependency_capture",
        lambda: _dependency_fixture(("base", "1")),
    )

    def fail_import(_: str) -> object:
        os.environ["LD_LIBRARY_PATH"] = f"/mutated:{frozen}"
        raise RuntimeError("fixture import failed")

    monkeypatch.setattr(runner_module.importlib, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="fixture import failed"):
        runner_module._load_runtime_imports(
            nvidia_bundle=SimpleNamespace(library_path=tmp_path / "bundle"),
            frozen_ld_library_path=frozen,
        )
    assert os.environ["LD_LIBRARY_PATH"] == frozen


def test_prepare_import_path_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    integration = str(runner_module.INTEGRATION_ROOT)
    monkeypatch.setenv("PYTHONPATH", f"{integration}:/other:{integration}")
    monkeypatch.setattr(runner_module.sys, "path", [integration, "/other"])

    runner_module._prepare_import_path()
    first = os.environ["PYTHONPATH"]
    runner_module._prepare_import_path()

    assert first == f"{integration}:/other"
    assert os.environ["PYTHONPATH"] == first
    assert runner_module.sys.path == [integration, "/other"]


def test_create_only_output_rejects_dangling_symlink(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.symlink_to(tmp_path / "missing-target")

    with pytest.raises(M2ValidationError, match="already exists"):
        runner_module._resolve_create_only_output_dir(output)


def test_create_only_output_resolves_absent_path(tmp_path: Path) -> None:
    output = tmp_path / "run"

    assert runner_module._resolve_create_only_output_dir(output) == output.resolve()


def test_engine_spawn_boundary_rejects_late_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = "/bundle:/cuda"
    dependencies = _dependency_fixture(("base", "1"))
    monkeypatch.setenv("LD_LIBRARY_PATH", frozen)
    monkeypatch.setattr(runner_module.sys, "path", ["/frozen"])
    monkeypatch.setattr(runner_module, "_dependency_capture", lambda: dependencies)

    runner_module._require_frozen_runtime_boundary(
        frozen_ld_library_path=frozen,
        effective_sys_path=["/frozen"],
        effective_dependencies=dependencies,
        stage="before EngineCore spawn",
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/late:{frozen}")
    with pytest.raises(M2ValidationError, match="before EngineCore spawn"):
        runner_module._require_frozen_runtime_boundary(
            frozen_ld_library_path=frozen,
            effective_sys_path=["/frozen"],
            effective_dependencies=dependencies,
            stage="before EngineCore spawn",
        )


def test_spawn_child_inherits_restored_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = "/bundle:/cuda"
    monkeypatch.setenv("LD_LIBRARY_PATH", frozen)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_capture_spawn_loader, args=(queue,))

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert queue.get(timeout=1) == frozen
    queue.close()


def test_nvidia_bundle_validation_binds_digest_and_loader_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "a" * 64
    manifest_digest = "c" * 64
    driver_version = "999.888.777"
    library = tmp_path / "rootfs/usr/lib/x86_64-linux-gnu"
    validation = SimpleNamespace(
        content_digest=digest,
        manifest_sha256=manifest_digest,
        kernel_module_version=driver_version,
        library_path=library,
    )
    calls: list[tuple[Path, str]] = []

    def validate_bundle(
        path: Path, *, expected_manifest_sha256: str
    ) -> SimpleNamespace:
        calls.append((path, expected_manifest_sha256))
        return validation

    monkeypatch.setattr(runner_module, "validate_bundle", validate_bundle)
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{library}:/usr/local/cuda/lib64")

    assert (
        runner_module._validate_nvidia_bundle(
            tmp_path,
            expected_manifest_sha256=manifest_digest,
            expected_content_digest=digest,
            expected_driver_version=driver_version,
        )
        is validation
    )
    assert calls == [(tmp_path, manifest_digest)]
    with pytest.raises(M2ValidationError, match="digest differs"):
        runner_module._validate_nvidia_bundle(
            tmp_path,
            expected_manifest_sha256=manifest_digest,
            expected_content_digest="b" * 64,
            expected_driver_version=driver_version,
        )
    with pytest.raises(M2ValidationError, match="kernel module version differs"):
        runner_module._validate_nvidia_bundle(
            tmp_path,
            expected_manifest_sha256=manifest_digest,
            expected_content_digest=digest,
            expected_driver_version="111.222.333",
        )
    with pytest.raises(M2ValidationError, match="manifest SHA-256 differs"):
        runner_module._validate_nvidia_bundle(
            tmp_path,
            expected_manifest_sha256="d" * 64,
            expected_content_digest=digest,
            expected_driver_version=driver_version,
        )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/driver")
    with pytest.raises(M2ValidationError, match="must exactly match"):
        runner_module._validate_nvidia_bundle(
            tmp_path,
            expected_manifest_sha256=manifest_digest,
            expected_content_digest=digest,
            expected_driver_version=driver_version,
        )


def test_loaded_libcuda_mapping_must_resolve_inside_bundle(tmp_path: Path) -> None:
    driver_version = "999.888.777"
    rootfs = tmp_path / "rootfs"
    library = rootfs / "usr/lib/x86_64-linux-gnu"
    library.mkdir(parents=True)
    libcuda = library / f"libcuda.so.{driver_version}"
    libcuda.write_bytes(b"fixture libcuda")
    observed = libcuda.stat()
    inode = observed.st_ino
    device = f"{os.major(observed.st_dev):02x}:{os.minor(observed.st_dev):02x}"
    maps = tmp_path / "maps"
    maps.write_text(
        "\n".join(
            [
                f"1000-2000 r--p 00000000 {device} {inode} {libcuda}",
                f"2000-3000 r-xp 00001000 {device} {inode} {libcuda}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation = SimpleNamespace(
        libcuda_path=libcuda,
        runtime=SimpleNamespace(rootfs=rootfs),
    )

    capture = runner_module._capture_loaded_libcuda(validation, maps_path=maps)

    assert capture["resolved_path"] == str(libcuda)
    assert capture["mapping_count"] == 2
    assert capture["sha256"] == hashlib.sha256(b"fixture libcuda").hexdigest()

    host = tmp_path / f"host/libcuda.so.{driver_version}"
    host.parent.mkdir()
    host.write_bytes(b"host libcuda")
    maps.write_text(
        f"1000-2000 r-xp 00000000 {device} {host.stat().st_ino} {host}\n",
        encoding="utf-8",
    )
    with pytest.raises(M2ValidationError, match="does not come"):
        runner_module._capture_loaded_libcuda(validation, maps_path=maps)


@pytest.mark.skipif(np is None, reason="NumPy is required for logit helper tests")
def test_dense_logits_accepts_flat_duplicate_sampled_token() -> None:
    flat = _Flat(
        start_indices=[0],
        end_indices=[4],
        token_ids=[1, 0, 1, 2],
        logprobs=[2.0, 1.0, 2.0, -3.0],
    )
    vector = dense_logits_from_logprobs(flat, 3)
    np.testing.assert_array_equal(vector, np.array([1.0, 2.0, -3.0]))


@pytest.mark.skipif(np is None, reason="NumPy is required for logit helper tests")
def test_dense_logits_accepts_mapping_and_rejects_missing_vocab() -> None:
    vector = dense_logits_from_logprobs(
        [{0: _Value(1.0), 1: _Value(0.5)}],
        2,
    )
    np.testing.assert_array_equal(vector, np.array([1.0, 0.5]))
    with pytest.raises(M2ValidationError, match="cover"):
        dense_logits_from_logprobs([{0: _Value(1.0)}], 2)


@pytest.mark.skipif(np is None, reason="NumPy is required for logit helper tests")
def test_logit_comparison_reports_tolerance_and_token_state() -> None:
    left = np.array([1.0, 2.0])
    right = np.array([1.0, 2.0001])
    exact = compare_logit_vectors("A", 1, left, "B", 1, right, atol=0.0, rtol=0.0)
    tolerant = compare_logit_vectors("A", 1, left, "B", 2, right, atol=0.001, rtol=0.0)
    assert exact.allclose is False
    assert exact.token_equal is True
    assert tolerant.allclose is True
    assert tolerant.token_equal is False


def test_frozen_tolerance_must_predate_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _write_parent_inputs(tmp_path)

    def validate_bundle(
        path: Path,
        *,
        expected_manifest_sha256: str,
        expected_implementation_manifest_sha256: str,
        **_: object,
    ) -> tuple[dict, str, SimpleNamespace]:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected_manifest_sha256:
            raise runner_module.CalibrationEvidenceError(
                "calibration manifest SHA-256 differs from the expected digest"
            )
        assert expected_implementation_manifest_sha256 == parent.implementation_sha256
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return manifest, observed, SimpleNamespace(runs=tuple(parent.runs))

    monkeypatch.setattr(
        runner_module,
        "validate_published_calibration_bundle",
        validate_bundle,
    )
    run_started_ns = time.time_ns() + 1_000_000_000
    frozen = load_frozen_tolerance(
        parent.tolerance_path,
        run_started_ns=run_started_ns,
    )
    assert frozen.atol == 0.125
    assert (
        frozen.calibration_manifest_sha256
        == (parent.tolerance_payload["calibration_manifest_sha256"])
    )
    cohort = load_calibration_cohort(
        parent.cohort_path,
        frozen_tolerance=frozen,
        run_started_ns=run_started_ns,
        expected_implementation_manifest_sha256=parent.implementation_sha256,
    )
    assert cohort["run_count"] == MIN_CALIBRATION_RUNS

    with pytest.raises(M2ValidationError, match="before runner startup"):
        load_frozen_tolerance(
            parent.tolerance_path,
            run_started_ns=parent.tolerance_path.stat().st_mtime_ns,
        )

    parent.cohort_payload["observed_max_abs_error"] = 0.25
    parent.cohort_path.write_text(json.dumps(parent.cohort_payload), encoding="utf-8")
    with pytest.raises(M2ValidationError, match="SHA-256 differs"):
        load_calibration_cohort(
            parent.cohort_path,
            frozen_tolerance=frozen,
            run_started_ns=time.time_ns() + 1_000_000_000,
            expected_implementation_manifest_sha256=parent.implementation_sha256,
        )


def test_parent_evidence_inputs_reject_symlinks(tmp_path: Path) -> None:
    parent = _write_parent_inputs(tmp_path)
    run_started_ns = time.time_ns() + 1_000_000_000
    tolerance_link = tmp_path / "tolerance-link.json"
    tolerance_link.symlink_to(parent.tolerance_path)

    with pytest.raises(M2ValidationError, match="must be a regular file"):
        load_frozen_tolerance(tolerance_link, run_started_ns=run_started_ns)

    frozen = load_frozen_tolerance(
        parent.tolerance_path,
        run_started_ns=run_started_ns,
    )
    cohort_link = tmp_path / "cohort-link.json"
    cohort_link.symlink_to(parent.cohort_path)
    with pytest.raises(M2ValidationError, match="must be a regular file"):
        load_calibration_cohort(
            cohort_link,
            frozen_tolerance=frozen,
            run_started_ns=run_started_ns,
            expected_implementation_manifest_sha256=parent.implementation_sha256,
        )


@pytest.mark.parametrize("mutation", ["content", "time"])
def test_tolerance_rejects_mutation_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    parent = _write_parent_inputs(tmp_path)
    original_read = runner_module.read_stable_bytes

    def mutate_after_read(path: Path, *, label: str) -> bytes:
        raw = original_read(path, label=label)
        if path == parent.tolerance_path:
            _mutate_parent_file(path, mutation)
        return raw

    monkeypatch.setattr(runner_module, "read_stable_bytes", mutate_after_read)
    with pytest.raises(M2ValidationError, match="changed during validation"):
        load_frozen_tolerance(
            parent.tolerance_path,
            run_started_ns=time.time_ns() + 1_000_000_000,
        )


@pytest.mark.parametrize("mutation", ["content", "time"])
def test_cohort_rejects_mutation_during_bundle_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    parent = _write_parent_inputs(tmp_path)
    run_started_ns = time.time_ns() + 1_000_000_000
    frozen = load_frozen_tolerance(
        parent.tolerance_path,
        run_started_ns=run_started_ns,
    )

    def mutating_bundle(path: Path, **_: object) -> tuple[dict, str, SimpleNamespace]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        _mutate_parent_file(path, mutation)
        return payload, observed, SimpleNamespace(runs=tuple(parent.runs))

    monkeypatch.setattr(
        runner_module,
        "validate_published_calibration_bundle",
        mutating_bundle,
    )
    with pytest.raises(M2ValidationError, match="changed during validation"):
        load_calibration_cohort(
            parent.cohort_path,
            frozen_tolerance=frozen,
            run_started_ns=run_started_ns,
            expected_implementation_manifest_sha256=parent.implementation_sha256,
        )


def test_jsonl_reader_rejects_partial_rows(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"event":"ok"}', encoding="utf-8")
    with pytest.raises(M2ValidationError, match="unterminated"):
        read_jsonl_strict(path)


def test_prefetch_requires_one_external_block() -> None:
    validate_prefetch_result(
        {
            "started": True,
            "completed": True,
            "reason": "completed",
            "local_gpu_hit_tokens": 0,
            "external_hit_tokens": 16,
            "loaded_tokens": 16,
        }
    )
    with pytest.raises(M2ValidationError, match="exactly one external"):
        validate_prefetch_result(
            {
                "started": True,
                "completed": True,
                "reason": "completed",
                "local_gpu_hit_tokens": 0,
                "external_hit_tokens": 0,
                "loaded_tokens": 0,
            }
        )


def test_prefetch_derives_external_tokens_after_deferred_lookup() -> None:
    validate_prefetch_result(
        {
            "started": True,
            "lookup_pending": True,
            "completed": True,
            "reason": "completed",
            "local_gpu_hit_tokens": 0,
            "loaded_tokens": 16,
        }
    )

    with pytest.raises(M2ValidationError, match="exactly one external"):
        validate_prefetch_result(
            {
                "started": True,
                "completed": True,
                "reason": "completed",
                "local_gpu_hit_tokens": 0,
                "loaded_tokens": 16,
            }
        )


def test_diagnostic_transfer_requires_equal_endpoint_digests() -> None:
    digest = "a" * 64
    submitted = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "event": "submitted",
        "run_id": "run",
        "job_id": 1,
        "request_id": "req",
        "direction": "D2H",
    }
    terminal = _terminal(
        request_id="req",
        direction="D2H",
        source=_endpoint("GPU", 3, 7, digest),
        target=_endpoint("CPU", 9, 2, digest),
    )
    assert (
        validate_diagnostic_transfer(
            [submitted, terminal],
            request_id="req",
            direction="D2H",
            run_id="run",
        )
        == terminal
    )

    terminal["target"]["digest"] = "b" * 64
    with pytest.raises(M2ValidationError, match="digest changed"):
        validate_diagnostic_transfer(
            [submitted, terminal],
            request_id="req",
            direction="D2H",
            run_id="run",
        )


def test_diagnostic_trace_rejects_extra_or_failed_dma() -> None:
    digest = "d" * 64
    submitted = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "event": "submitted",
        "status": "in_flight",
        "run_id": "run",
        "phase": "ABBA",
        "job_id": 1,
        "request_id": "req",
        "direction": "D2H",
        "framing": "DAGKV_PAYLOAD_V1",
    }
    terminal = _terminal(
        request_id="req",
        direction="D2H",
        source=_endpoint("GPU", 3, 7, digest),
        target=_endpoint("CPU", 9, 2, digest),
    )
    terminal["phase"] = "ABBA"
    validate_diagnostic_trace_closed_set(
        [submitted, terminal],
        run_id="run",
        expected_transfers=(("req", "D2H"),),
    )

    failed = dict(terminal, status="failed")
    with pytest.raises(M2ValidationError, match="did not complete"):
        validate_diagnostic_trace_closed_set(
            [submitted, failed],
            run_id="run",
            expected_transfers=(("req", "D2H"),),
        )

    extra = dict(submitted, job_id=2, request_id="extra")
    with pytest.raises(M2ValidationError, match="exactly submitted/terminal"):
        validate_diagnostic_trace_closed_set(
            [submitted, terminal, extra],
            run_id="run",
            expected_transfers=(("req", "D2H"),),
        )


def test_native_transfer_requires_scheduler_and_lifecycle_rows() -> None:
    rows = [
        {
            "event": "store_scheduled",
            "trace_id": "trace",
            "native_store_bytes": 4096,
        },
        {
            "event": "store_complete",
            "trace_id": "trace",
            "native_store_bytes": 4096,
        },
        {
            "event": "kv_lifecycle",
            "trace_id": "trace",
            "action": "save",
            "status": "scheduled",
            "event_id": "save:scheduled",
            "request_id": "engine-request",
        },
        {
            "event": "kv_lifecycle",
            "trace_id": "trace",
            "action": "save",
            "status": "completed",
            "parent_event_id": "save:scheduled",
            "observed_byte_count": 4096,
        },
    ]
    assert validate_native_transfer(rows, trace_id="trace", direction="D2H") == 4096
    assert (
        resolve_native_transfer_request_id(rows, trace_id="trace", direction="D2H")
        == "engine-request"
    )
    for row in rows:
        row["trace_id"] = "run:trace"
        if row.get("event") == "kv_lifecycle":
            row["run_id"] = "run"
    validate_native_trace_closed_set(
        rows,
        run_id="run",
        expected_transfers=(("run:trace", "D2H"),),
    )

    rows.append(dict(rows[-1], status="failed"))
    with pytest.raises(M2ValidationError, match="failed event"):
        validate_native_trace_closed_set(
            rows,
            run_id="run",
            expected_transfers=(("run:trace", "D2H"),),
        )


def test_abba_transfer_chain_requires_fresh_generations() -> None:
    digest = "c" * 64
    b1_d2h = _terminal(
        request_id="b1-producer",
        direction="D2H",
        source=_endpoint("GPU", 4, 1, digest),
        target=_endpoint("CPU", 8, 1, digest),
    )
    b1_h2d = _terminal(
        request_id="b1-prefetch",
        direction="H2D",
        source=_endpoint("CPU", 8, 1, digest),
        target=_endpoint("GPU", 4, 2, digest),
    )
    b2_d2h = _terminal(
        request_id="b2-producer",
        direction="D2H",
        source=_endpoint("GPU", 5, 1, digest),
        target=_endpoint("CPU", 8, 2, digest),
    )
    b2_h2d = _terminal(
        request_id="b2-prefetch",
        direction="H2D",
        source=_endpoint("CPU", 8, 2, digest),
        target=_endpoint("GPU", 5, 2, digest),
    )
    validate_abba_transfer_chain(b1_d2h, b1_h2d, b2_d2h, b2_h2d)

    b2_d2h["target"]["allocation_generation"] = 1
    b2_h2d["source"]["allocation_generation"] = 1
    with pytest.raises(M2ValidationError, match="reused B1 CPU"):
        validate_abba_transfer_chain(b1_d2h, b1_h2d, b2_d2h, b2_h2d)


def test_llm_kwargs_forbid_convenience_offload_size() -> None:
    sentinel = object()
    kwargs = build_llm_kwargs(Path("/model"), sentinel)
    assert kwargs["kv_transfer_config"] is sentinel
    assert kwargs["block_size"] == 16
    assert kwargs["enable_chunked_prefill"] is True
    assert kwargs["max_logprobs"] == -1
    assert kwargs["logprobs_mode"] == "raw_logits"
    assert kwargs["attention_config"] == {
        "backend": "FLASH_ATTN",
        "flash_attn_version": 2,
    }
    assert "kv_offloading_size" not in kwargs


def test_a1_g_control_shares_lifecycle_phase_with_distinct_trace() -> None:
    params = _transfer_params(
        run_id="run",
        phase="A1_G",
        trace_id="run:G:measurement",
        native_trace=Path("/tmp/native.jsonl"),
        role="measurement",
    )

    assert params["offload_phase"] == "A1_G"
    assert params["offload_trace_id"] == "run:G:measurement"
    assert params["max_offload_tokens"] == 0


def test_git_capture_sorts_porcelain_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = {
        "rev-parse": b"a" * 40 + b"\n",
        "status": b"?? z.py\n M a.py\nD  middle.py\n",
        "diff": b"",
        "ls-files": b"",
    }

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=responses[command[3]])

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    capture = runner_module._git_capture(
        tmp_path / "source",
        output_dir=tmp_path / "evidence",
        label="fixture",
    )

    assert capture["status_short"] == sorted(capture["status_short"])
    assert capture["status_short"] == [" M a.py", "?? z.py", "D  middle.py"]
