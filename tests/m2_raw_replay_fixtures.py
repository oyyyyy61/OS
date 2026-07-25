"""Reusable raw M2 run fixtures and their focused validator tests."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tools.m2_raw_replay import FROZEN_QWEN3_8B_VOCAB_SIZE
from tools.nvidia_driver_userspace_bundle import (
    CONTENT_DIGEST_DOMAIN,
    RUNTIME_DERIVATION,
)
from tools.nvidia_driver_userspace_bundle import (
    MANIFEST_SCHEMA as NVIDIA_MANIFEST_SCHEMA,
)

RUN_ID = "m2-fixture-raw-replay"
DIGEST = hashlib.sha256(b"canonical kv").hexdigest()
NVIDIA_KERNEL_VERSION = "999.888.777"
BYTES = 4096
PHASES = ("A1", "G", "B1", "B2", "A2")
PAIRS = (
    ("A1", "G"),
    ("A1", "B1"),
    ("A1", "B2"),
    ("A1", "A2"),
    ("G", "B1"),
    ("G", "B2"),
    ("B1", "B2"),
)


@dataclass(frozen=True, slots=True)
class RawRunFixture:
    run_dir: Path
    run_id: str
    implementation_manifest_sha256: str
    reproducibility_fingerprint: str
    nvidia_bundle_root: str
    nvidia_driver_version: str
    nvidia_manifest_sha256: str
    nvidia_content_digest: str


def _canonical(payload: Any) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        )
    )


def _nvidia_capture(
    *,
    bundle_root: Path | None = None,
    driver_version: str = NVIDIA_KERNEL_VERSION,
) -> dict[str, Any]:
    kernel_version = driver_version
    bundle_root = bundle_root or Path("/fixture/nvidia")
    if not bundle_root.is_absolute():
        raise ValueError("fixture NVIDIA bundle root must be absolute")
    rootfs = bundle_root / "rootfs"
    library = rootfs / "usr/lib/x86_64-linux-gnu"
    nvidia_smi = rootfs / "usr/bin/nvidia-smi"
    libcuda = library / f"libcuda.so.{kernel_version}"
    libnvidia_ml = library / f"libnvidia-ml.so.{kernel_version}"
    libcuda_payload = b"fixture libcuda"
    packages = [
        {
            "path": "libnvidia-compute-fixture_amd64.deb",
            "package": "libnvidia-compute-fixture",
            "version": f"{kernel_version}-fixture",
            "architecture": "amd64",
            "mode": 0o444,
            "size": 1,
            "sha256": hashlib.sha256(b"compute deb").hexdigest(),
        },
        {
            "path": "nvidia-utils-fixture_amd64.deb",
            "package": "nvidia-utils-fixture",
            "version": f"{kernel_version}-fixture",
            "architecture": "amd64",
            "mode": 0o444,
            "size": 1,
            "sha256": hashlib.sha256(b"utils deb").hexdigest(),
        },
    ]
    runtime_tree = [
        {"path": ".", "type": "directory", "mode": 0o555},
        {"path": "usr", "type": "directory", "mode": 0o555},
        {"path": "usr/bin", "type": "directory", "mode": 0o555},
        {
            "path": "usr/bin/nvidia-smi",
            "type": "file",
            "mode": 0o555,
            "size": 1,
            "sha256": hashlib.sha256(b"nvidia-smi").hexdigest(),
        },
        {"path": "usr/lib", "type": "directory", "mode": 0o555},
        {
            "path": "usr/lib/x86_64-linux-gnu",
            "type": "directory",
            "mode": 0o555,
        },
        {
            "path": "usr/lib/x86_64-linux-gnu/libcuda.so",
            "type": "symlink",
            "target": "libcuda.so.1",
        },
        {
            "path": "usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "type": "symlink",
            "target": libcuda.name,
        },
        {
            "path": libcuda.relative_to(rootfs).as_posix(),
            "type": "file",
            "mode": 0o444,
            "size": len(libcuda_payload),
            "sha256": hashlib.sha256(libcuda_payload).hexdigest(),
        },
        {
            "path": "usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "type": "symlink",
            "target": libnvidia_ml.name,
        },
        {
            "path": libnvidia_ml.relative_to(rootfs).as_posix(),
            "type": "file",
            "mode": 0o444,
            "size": 1,
            "sha256": hashlib.sha256(b"fixture nvml").hexdigest(),
        },
    ]
    content_digest = _canonical(
        {
            "domain": CONTENT_DIGEST_DOMAIN,
            "kernel_module_version": kernel_version,
            "packages": packages,
            "runtime_derivation": RUNTIME_DERIVATION,
            "runtime_tree": runtime_tree,
        }
    )
    manifest = {
        "schema_version": NVIDIA_MANIFEST_SCHEMA,
        "created_at_utc": "2026-07-24T00:00:00+00:00",
        "bundle_type": "nvidia_debian_userspace_exact",
        "kernel_module_version": kernel_version,
        "package_count": len(packages),
        "packages": packages,
        "runtime_derivation": RUNTIME_DERIVATION,
        "runtime_entry_count": len(runtime_tree),
        "runtime_tree": runtime_tree,
        "content_digest_algorithm": "sha256",
        "content_digest": content_digest,
    }
    manifest_bytes = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "root": str(bundle_root),
        "expected_manifest_sha256": manifest_sha256,
        "expected_content_digest": content_digest,
        "expected_driver_version": kernel_version,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "content_digest": content_digest,
        "kernel_module_version": kernel_version,
        "runtime": {
            "rootfs": str(rootfs),
            "library_directory": str(library),
            "nvidia_smi": str(nvidia_smi),
            "libcuda": str(libcuda),
            "libnvidia_ml": str(libnvidia_ml),
        },
        "libcuda_mapping": {
            "path": str(libcuda),
            "resolved_path": str(libcuda),
            "rootfs_relative_path": libcuda.relative_to(rootfs).as_posix(),
            "device": 1,
            "inode": 12,
            "size": len(libcuda_payload),
            "sha256": hashlib.sha256(libcuda_payload).hexdigest(),
            "mapping_count": 5,
        },
    }


def _reseal(run_dir: Path) -> None:
    checksum = run_dir / "SHA256SUMS"
    if checksum.exists():
        checksum.unlink()
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    checksum.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(run_dir).as_posix()}\n" for path in files
        ),
        encoding="ascii",
    )


def _endpoint(tier: str, slot: int, generation: int) -> dict[str, Any]:
    return {
        "tier": tier,
        "physical_slot": slot,
        "allocation_generation": generation,
        "digest": DIGEST,
    }


def _diagnostic_pair(
    *,
    run_id: str,
    request_id: str,
    direction: str,
    job_id: int,
    source: dict[str, Any],
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    capture_ns = 100 + job_id * 10
    common = {
        "schema_version": "dagkv.vllm_m2.transfer_probe.v1",
        "run_id": run_id,
        "phase": "ABBA",
        "request_id": request_id,
        "direction": direction,
        "job_id": job_id,
        "framing": "DAGKV_PAYLOAD_V1",
        "payload_bytes": BYTES,
        "capture_ns": capture_ns,
        "source": source,
    }
    return [
        {
            **common,
            "event": "submitted",
            "status": "in_flight",
            "submit_ns": capture_ns + 1,
            "target": None,
        },
        {
            **common,
            "event": "terminal",
            "status": "completed",
            "failure_reason": None,
            "reported_bytes": BYTES,
            "terminal_ns": capture_ns + 2,
            "target": target,
        },
    ]


def _lookup(trace_id: str, request_id: str, gpu: int, native: int) -> dict[str, Any]:
    return {
        "event": "lookup",
        "trace_id": trace_id,
        "request_id": request_id,
        "gpu_hit_tokens": gpu,
        "native_load_hit_tokens": native,
        "skip_reading_prefix_cache": False,
        "dag_preload_event": "m2_explicit_prefetch" if native else None,
        "dag_preload_sources": ["m2_frozen_abba"] if native else [],
    }


def _native_transfer(
    *,
    run_id: str,
    label: str,
    direction: str,
    request_id: str,
    trace_id: str,
    job_id: int,
    cpu_generation: int,
) -> list[dict[str, Any]]:
    phase = label[:2]
    prefix = "store" if direction == "D2H" else "load"
    action = "save" if direction == "D2H" else "prefetch"
    source_tier, target_tier = ("gpu", "cpu") if direction == "D2H" else ("cpu", "gpu")
    operation_id = f"{request_id}:job:{job_id}"
    event_id = f"{operation_id}:scheduled"
    block = {
        "allocation_generation": cpu_generation,
        "block_id": "cpu:0",
        "byte_count": BYTES,
        "identity_kind": "physical_slot",
    }
    common = {
        "schema_version": "kv_lifecycle_event_v2",
        "run_id": run_id,
        "event": "kv_lifecycle",
        "action": action,
        "phase": phase,
        "request_id": request_id,
        "trace_id": trace_id,
        "accounting_trace_id": trace_id,
        "workflow_id": f"m2-abba:{run_id}",
        "source": "vllm.offloading.scheduler",
        "source_tier": source_tier,
        "target_tier": target_tier,
        "block_count": 1,
        "byte_count": BYTES,
        "blocks": [block],
        "operation_id": operation_id,
    }
    byte_key = f"native_{prefix}_bytes"
    return [
        {
            **common,
            "status": "scheduled",
            "event_id": event_id,
            "parent_event_id": None,
            "observed_byte_count": None,
        },
        {
            "event": f"{prefix}_scheduled",
            "trace_id": trace_id,
            "request_id": request_id,
            "job_id": job_id,
            "time": 10.0 + job_id,
            byte_key: BYTES,
        },
        {
            **common,
            "status": "completed",
            "event_id": f"{operation_id}:completed",
            "parent_event_id": event_id,
            "observed_byte_count": BYTES,
        },
        {
            "event": f"{prefix}_complete",
            "trace_id": trace_id,
            "request_id": request_id,
            "job_id": job_id,
            "time": 10.5 + job_id,
            byte_key: BYTES,
        },
    ]


def _manifest_entry(
    path: str, payload: bytes, *, kind: str | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if kind is not None:
        entry.update({"kind": kind, "mtime_ns": 1, "inode": len(path)})
    return entry


def _write_sparse_logits(np: Any, path: Path, *, hot: bool) -> Any:
    vector = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(FROZEN_QWEN3_8B_VOCAB_SIZE,),
    )
    vector[931] = 1.0
    vector[932] = 2.0
    if hot:
        vector[10] = 0.1
    vector.flush()
    del vector
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _git_capture(run_dir: Path, label: str) -> dict[str, Any]:
    state = run_dir / "source_state"
    state.mkdir(exist_ok=True)
    patch = state / f"{label}.tracked.patch"
    archive = state / f"{label}.untracked.tar"
    patch.write_bytes(b"")
    with tarfile.open(archive, "w"):
        pass
    head = hashlib.sha1(label.encode()).hexdigest()
    snapshot = {
        "head": head,
        "tracked_diff_sha256": _sha(patch),
        "tracked_diff_bytes": 0,
        "untracked": [],
    }
    return {
        "root": f"/fixture/{label}",
        "head": head,
        "dirty": False,
        "status_short": [],
        "tracked_patch": f"source_state/{label}.tracked.patch",
        "tracked_patch_sha256": _sha(patch),
        "untracked_archive": f"source_state/{label}.untracked.tar",
        "untracked_archive_sha256": _sha(archive),
        "untracked_files": [],
        "snapshot_sha256": _canonical(snapshot),
    }


def build_raw_run(
    run_dir: Path,
    *,
    run_id: str = RUN_ID,
    implementation: dict[str, Any] | None = None,
    expected_implementation_manifest_sha256: str | None = None,
    fingerprint_seed: str = "fixture",
    model_root: Path | None = None,
    cpu_bytes: int = 1024,
    protocol_payload: bytes | None = None,
    python_executable: Path | None = None,
    nvidia_bundle_root: Path | None = None,
    nvidia_driver_version: str = NVIDIA_KERNEL_VERSION,
) -> RawRunFixture:
    np = pytest.importorskip("numpy")
    run_dir.mkdir()
    protocol = protocol_payload or b"# Frozen M2 protocol\n"
    (run_dir / "protocol.md").write_bytes(protocol)

    vectors = {
        phase: _write_sparse_logits(
            np,
            run_dir / f"logits_{phase}.npy",
            hot=phase in {"G", "B1", "B2"},
        )
        for phase in PHASES
    }

    api = {"A1": "0", "G": "1", "B1": "3", "B2": "5", "A2": "6"}
    measurements: dict[str, dict[str, Any]] = {}
    for phase, vector in vectors.items():
        top = int(np.argmax(vector))
        margin = float(vector[top] - np.partition(vector, -2)[-2])
        measurements[phase] = {
            "request_id": api[phase],
            "trace_id": f"{run_id}:{phase}:measurement",
            "token_id": top,
            "num_cached_tokens": 0 if phase in {"A1", "A2"} else 16,
            "elapsed_ms": 1.0,
            "top1_margin": margin,
            "logits_file": f"logits_{phase}.npy",
            "logits_sha256": _sha(run_dir / f"logits_{phase}.npy"),
        }
    comparisons = []
    for left, right in PAIRS:
        absolute = np.abs(vectors[left] - vectors[right])
        denominator = np.maximum(np.abs(vectors[left]), np.abs(vectors[right]))
        relative = np.divide(
            absolute,
            denominator,
            out=np.zeros_like(absolute),
            where=denominator != 0,
        )
        comparisons.append(
            {
                "left": left,
                "right": right,
                "token_equal": True,
                "allclose": bool(
                    np.allclose(vectors[left], vectors[right], atol=0.125, rtol=0.0)
                ),
                "max_abs_error": float(absolute.max(initial=0.0)),
                "max_rel_error": float(relative.max(initial=0.0)),
            }
        )

    prefetch_ids = {"B1": "m2-prefetch-b1", "B2": "m2-prefetch-b2"}
    prefetch = {
        phase: {
            "allocated_block_ids": [[3]],
            "completed": True,
            "external_hit_tokens": 16,
            "loaded_tokens": 16,
            "local_gpu_hit_tokens": 0,
            "lookup_pending": False,
            "reason": "completed",
            "request_id": prefetch_ids[phase],
            "started": True,
            "steps": 2,
            "wall_ms": 1.0,
        }
        for phase in ("B1", "B2")
    }
    transfers = {
        "B1_D2H": {
            "request_id": "2",
            "engine_request_id": "2-engine",
            "trace_id": f"{run_id}:B1:producer",
        },
        "B1_H2D": {
            "request_id": prefetch_ids["B1"],
            "engine_request_id": prefetch_ids["B1"],
            "trace_id": f"{run_id}:B1:prefetch",
        },
        "B2_D2H": {
            "request_id": "4",
            "engine_request_id": "4-engine",
            "trace_id": f"{run_id}:B2:producer",
        },
        "B2_H2D": {
            "request_id": prefetch_ids["B2"],
            "engine_request_id": prefetch_ids["B2"],
            "trace_id": f"{run_id}:B2:prefetch",
        },
    }
    execution = {
        "schema_version": "dagkv.m2.vllm_abba.v3",
        "run_id": run_id,
        "measurements": {
            phase: {
                "request_id": api[phase],
                "trace_id": f"{run_id}:{phase}:measurement",
            }
            for phase in PHASES
        },
        "transfers": transfers,
    }
    _write_json(run_dir / "execution_ids.json", execution)

    endpoints = {
        "B1_D2H": (_endpoint("GPU", 2, 3), _endpoint("CPU", 0, 1)),
        "B1_H2D": (_endpoint("CPU", 0, 1), _endpoint("GPU", 3, 2)),
        "B2_D2H": (_endpoint("GPU", 4, 2), _endpoint("CPU", 0, 2)),
        "B2_H2D": (_endpoint("CPU", 0, 2), _endpoint("GPU", 3, 3)),
    }
    diagnostic: list[dict[str, Any]] = []
    for job_id, (label, direction) in enumerate(
        (("B1_D2H", "D2H"), ("B1_H2D", "H2D"), ("B2_D2H", "D2H"), ("B2_H2D", "H2D"))
    ):
        source, target = endpoints[label]
        diagnostic.extend(
            _diagnostic_pair(
                run_id=run_id,
                request_id=transfers[label]["engine_request_id"],
                direction=direction,
                job_id=job_id,
                source=source,
                target=target,
            )
        )
    _write_jsonl(run_dir / "diagnostic_transfers.jsonl", diagnostic)

    native = [
        _lookup(
            f"{run_id}:{phase}:measurement",
            f"{api[phase]}-engine",
            0 if phase in {"A1", "A2"} else 16,
            0,
        )
        for phase in PHASES
    ]
    for job_id, (label, direction) in enumerate(
        (("B1_D2H", "D2H"), ("B1_H2D", "H2D"), ("B2_D2H", "D2H"), ("B2_H2D", "H2D"))
    ):
        transfer = transfers[label]
        if direction == "D2H":
            native.append(
                _lookup(transfer["trace_id"], transfer["engine_request_id"], 0, 0)
            )
        else:
            native.append(
                _lookup(transfer["trace_id"], transfer["engine_request_id"], 0, 16)
            )
        native.extend(
            _native_transfer(
                run_id=run_id,
                label=label,
                direction=direction,
                request_id=transfer["engine_request_id"],
                trace_id=transfer["trace_id"],
                job_id=job_id,
                cpu_generation=1 if label.startswith("B1") else 2,
            )
        )
    _write_jsonl(run_dir / "native_lifecycle.jsonl", native)

    dagkv_git = _git_capture(run_dir, "dagkv")
    vllm_git = _git_capture(run_dir, "vllm")
    if implementation is None:
        implementation_files = sorted(
            [
                _manifest_entry("integrations/vllm_m2/dagkv_vllm_m2/spec.py", b"spec"),
                _manifest_entry("research/REFERENCES.md", b"references"),
                _manifest_entry("research/imported/RELATED_WORK_MATRIX.md", b"related"),
                _manifest_entry(
                    "research/protocols/M2_VLLM_REPLAY_PROTOCOL.md", protocol
                ),
                _manifest_entry("tools/m2_raw_replay.py", b"raw replay"),
                _manifest_entry(
                    "tools/nvidia_driver_userspace_bundle.py", b"nvidia bundle"
                ),
                _manifest_entry("tools/run_m2_vllm_abba.py", b"runner"),
            ],
            key=lambda item: item["path"],
        )
        implementation = {
            "files": implementation_files,
            "manifest_sha256": _canonical(implementation_files),
        }
    else:
        implementation = json.loads(json.dumps(implementation))
        implementation_files = implementation["files"]
    if expected_implementation_manifest_sha256 is not None:
        assert implementation["manifest_sha256"] == (
            expected_implementation_manifest_sha256
        )
    model_files = sorted(
        [
            _manifest_entry("config.json", b"config", kind="metadata"),
            _manifest_entry("model-00001.safetensors", b"weights", kind="weight"),
            _manifest_entry("model.safetensors.index.json", b"index", kind="metadata"),
        ],
        key=lambda item: item["path"],
    )
    model_content = [
        {key: item[key] for key in ("path", "size", "kind", "sha256")}
        for item in model_files
    ]
    model = {
        "root": str(model_root or Path("/fixture/model")),
        "full_hashes": True,
        "files": model_files,
        "manifest_sha256": _canonical(model_content),
    }
    extension = _manifest_entry("vllm/_C.abi3.so", b"extension")
    extension.update({"mtime_ns": 1, "inode": 10})
    executable_path = python_executable or Path("/fixture/python")
    runtime_python_path = executable_path.resolve(strict=False)
    python_entry = {
        "path": str(runtime_python_path),
        "size": 6,
        "sha256": hashlib.sha256(b"python").hexdigest(),
        "mtime_ns": 1,
        "inode": 11,
    }
    runtime_content = {
        "vllm_extensions": [
            {key: extension[key] for key in ("path", "size", "sha256")}
        ],
        "python_executable": {
            key: python_entry[key] for key in ("path", "size", "sha256")
        },
    }
    runtime = {
        "root": "/fixture/vllm",
        "full_hashes": True,
        "vllm_extensions": [extension],
        "python_executable": python_entry,
        "manifest_sha256": _canonical(runtime_content),
    }
    packages = [{"name": "numpy", "version": "fixture"}]
    dependencies = {"packages": packages, "manifest_sha256": _canonical(packages)}
    engine = {
        "async_scheduling": False,
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2},
        "block_size": 16,
        "data_parallel_size": 1,
        "disable_hybrid_kv_cache_manager": True,
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.82,
        "logprobs_mode": "raw_logits",
        "max_logprobs": -1,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 1,
        "model": model["root"],
        "pipeline_parallel_size": 1,
        "scheduling_policy": "fcfs",
        "seed": 20260724,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
    }
    connector = {
        "cpu_bytes_to_use": cpu_bytes,
        "dagkv_diagnostic_phase": "ABBA",
        "dagkv_diagnostic_run_id": run_id,
        "dagkv_diagnostic_trace_file": str(run_dir / "diagnostic_transfers.jsonl"),
        "fanout_layerwise_load": False,
        "lifecycle_accounting_enabled": True,
        "spec_module_path": "dagkv_vllm_m2.spec",
        "spec_name": "DAGKVDiagnosticCPUOffloadingSpec",
    }
    nvidia = _nvidia_capture(
        bundle_root=nvidia_bundle_root,
        driver_version=nvidia_driver_version,
    )
    system = {
        "frozen_environment": fingerprint_seed,
        "gpu": {
            "name": "fixture",
            "nvidia_smi": (
                f"GPU-fixture, {nvidia['kernel_module_version']}, "
                "00000000:01:00.0, Fixture GPU, 24564"
            ),
            "nvidia_smi_executable": nvidia["runtime"]["nvidia_smi"],
        },
        "environment": {"LD_LIBRARY_PATH": nvidia["runtime"]["library_directory"]},
    }
    components = {
        "implementation_manifest_sha256": implementation["manifest_sha256"],
        "vllm_snapshot_sha256": vllm_git["snapshot_sha256"],
        "model_manifest_sha256": model["manifest_sha256"],
        "runtime_binary_manifest_sha256": runtime["manifest_sha256"],
        "dependency_manifest_sha256": dependencies["manifest_sha256"],
        "nvidia_driver_userspace_content_digest": nvidia["content_digest"],
        "system": system,
        "prompt_token_ids": list(range(1000, 1017)),
        "block_size": 16,
        "cpu_bytes": cpu_bytes,
        "engine_config": engine,
        "connector_config": {
            key: value
            for key, value in connector.items()
            if key not in {"dagkv_diagnostic_trace_file", "dagkv_diagnostic_run_id"}
        },
    }
    fingerprint = _canonical(components)
    provenance = {
        "schema_version": "dagkv.m2.vllm_abba.v3",
        "run_id": run_id,
        "mode": "calibration",
        "started_at_utc": "2026-07-25T00:00:00+00:00",
        "argv": [
            "tools/run_m2_vllm_abba.py",
            "--full-provenance",
            "--nvidia-userspace-bundle-root",
            nvidia["root"],
            "--expected-nvidia-userspace-bundle-manifest-sha256",
            nvidia["manifest_sha256"],
            "--expected-nvidia-userspace-bundle-content-digest",
            nvidia["content_digest"],
            "--expected-nvidia-driver-version",
            nvidia["kernel_module_version"],
        ],
        "python": "3.12",
        "executable": str(executable_path),
        "prompt_token_ids": list(range(1000, 1017)),
        "block_size": 16,
        "cpu_bytes": cpu_bytes,
        "tolerance": {"atol": 0.125, "rtol": 0.0},
        "frozen_tolerance": None,
        "calibration_cohort": None,
        "full_provenance": True,
        "preflight": {
            "cuda_version": "13.0",
            "gpu_count": 1,
            "gpu_name": "fixture",
            "model_architectures": ["Qwen3ForCausalLM"],
            "model_config_sha256": next(
                item["sha256"] for item in model_files if item["path"] == "config.json"
            ),
            "torch_version": "2.0",
            "vllm_module": "/fixture/vllm/vllm/__init__.py",
            "vllm_version": "fixture",
        },
        "implementation": implementation,
        "dagkv_git": dagkv_git,
        "vllm_git": vllm_git,
        "model": model,
        "runtime_binaries": runtime,
        "dependencies": dependencies,
        "nvidia_driver_userspace": nvidia,
        "system": system,
        "reproducibility_components": components,
        "reproducibility_fingerprint": fingerprint,
        "engine_config": engine,
        "connector_config": connector,
        "postflight": {
            "completed_at_utc": "2026-07-25T00:00:01+00:00",
            "dagkv_git_snapshot_sha256": dagkv_git["snapshot_sha256"],
            "vllm_git_snapshot_sha256": vllm_git["snapshot_sha256"],
            "implementation_manifest_sha256": implementation["manifest_sha256"],
            "model_file_stats_unchanged": True,
            "runtime_binary_stats_unchanged": True,
            "nvidia_driver_userspace_content_digest": nvidia["content_digest"],
            "nvidia_driver_userspace_manifest_sha256": nvidia["manifest_sha256"],
            "nvidia_driver_userspace_unchanged": True,
            "libcuda_mapping_unchanged": True,
        },
    }
    _write_json(run_dir / "provenance.json", provenance)
    result = {
        "schema_version": "dagkv.m2.vllm_abba.v3",
        "run_id": run_id,
        "mode": "calibration",
        "gate_status": "CALIBRATED_NOT_ACCEPTED",
        "m2_accepted": False,
        "m2_item8_accepted": False,
        "formal_run_passed": False,
        "within_requested_tolerance": True,
        "minimum_top1_margin": 1.0,
        "reproducibility_fingerprint": fingerprint,
        "completed_at_utc": "2026-07-25T00:00:02+00:00",
        "tolerance": {"atol": 0.125, "rtol": 0.0},
        "measurements": measurements,
        "comparisons": comparisons,
        "prefetch": prefetch,
        "native_bytes": {label: BYTES for label in transfers},
        "diagnostic_bytes": {label: BYTES for label in transfers},
        "transfer_digests": {"B1": DIGEST, "B2": DIGEST},
        "artifacts": {
            "native_trace": "native_lifecycle.jsonl",
            "diagnostic_trace": "diagnostic_transfers.jsonl",
            "protocol": "protocol.md",
            "provenance": "provenance.json",
        },
    }
    _write_json(run_dir / "result.json", result)
    _reseal(run_dir)
    return RawRunFixture(
        run_dir=run_dir,
        run_id=run_id,
        implementation_manifest_sha256=implementation["manifest_sha256"],
        reproducibility_fingerprint=fingerprint,
        nvidia_bundle_root=nvidia["root"],
        nvidia_driver_version=nvidia["kernel_module_version"],
        nvidia_manifest_sha256=nvidia["manifest_sha256"],
        nvidia_content_digest=nvidia["content_digest"],
    )
