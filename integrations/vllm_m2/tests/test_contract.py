"""CPU-only checks for the vLLM diagnostic integration contract."""

from __future__ import annotations

import json
import pickle
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dagkv_vllm_m2.connector import DAGKVDiagnosticOffloadingConnector
from dagkv_vllm_m2.framing import PAYLOAD_FRAMING, observe_payload
from dagkv_vllm_m2.metadata import ProbedGPULoadStoreSpec
from dagkv_vllm_m2.sink import DurableJSONLSink
from dagkv_vllm_m2.spec import DAGKVDiagnosticCPUOffloadingWorker
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadAllocationRecord,
    ReqContextSnapshot,
    TransferResult,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

_PAGE_BYTES = 8
_GPU_BLOCK = 1
_CPU_BLOCK = 2


class _FakeDiagnosticWorker(DAGKVDiagnosticCPUOffloadingWorker):
    """CPU-only backend with explicit completion visibility controls."""

    def __init__(
        self,
        trace_file: Path,
        *,
        completion: str = "immediate",
        corrupt_target: bool = False,
        reported_bytes: int = _PAGE_BYTES,
    ) -> None:
        gpu = torch.arange(32, dtype=torch.int8).view(4, _PAGE_BYTES)
        kv_caches = SimpleNamespace(
            tensors=[SimpleNamespace(tensor=gpu, page_size_bytes=_PAGE_BYTES)],
            group_data_refs=[
                [SimpleNamespace(tensor_idx=0, page_size_bytes=_PAGE_BYTES)]
            ],
        )
        self.cpu_tensors = [torch.zeros((4, _PAGE_BYTES), dtype=torch.int8)]
        self._fake_completion = completion
        self._fake_corrupt_target = corrupt_target
        self._fake_reported_bytes = reported_bytes
        self._fake_ready: list[TransferResult] = []
        self._fake_deferred: list[TransferResult] = []
        self.waited_job_ids: set[int] = set()
        self.backend_shutdown_calls = 0
        self._dagkv_initialize(
            kv_caches,
            str(trace_file),
            run_id="unit-run",
            phase="unit-phase",
        )

    def _complete_copy(
        self,
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        cpu_spec: CPULoadStoreSpec,
        *,
        direction: str,
    ) -> bool:
        gpu_row = self._dagkv_gpu_tensors[0][int(gpu_spec.block_ids[0])]
        cpu_row = self.cpu_tensors[0][int(cpu_spec.block_ids[0])]
        source, target = (
            (gpu_row, cpu_row) if direction == "D2H" else (cpu_row, gpu_row)
        )
        target.copy_(source)
        if self._fake_corrupt_target:
            target[0] = target[0] + 1

        result = TransferResult(
            job_id=job_id,
            success=True,
            transfer_size=self._fake_reported_bytes,
        )
        if self._fake_completion == "immediate":
            self._fake_ready.append(result)
        elif self._fake_completion == "wait":
            self._fake_deferred.append(result)
        elif self._fake_completion != "never":
            raise AssertionError(
                f"unknown fake completion mode: {self._fake_completion}"
            )
        return True

    def _backend_submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: CPULoadStoreSpec
    ) -> bool:
        return self._complete_copy(
            job_id,
            src_spec,
            dst_spec,
            direction="D2H",
        )

    def _backend_submit_load(
        self, job_id: int, src_spec: CPULoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._complete_copy(
            job_id,
            dst_spec,
            src_spec,
            direction="H2D",
        )

    def _backend_get_finished(self) -> list[TransferResult]:
        ready, self._fake_ready = self._fake_ready, []
        return ready

    def _backend_wait(self, job_ids: set[int]) -> None:
        self.waited_job_ids.update(job_ids)
        if self._fake_completion == "wait":
            self._fake_ready.extend(self._fake_deferred)
            self._fake_deferred.clear()

    def _backend_shutdown(self) -> None:
        self.backend_shutdown_calls += 1


def _transfer_specs(
    direction: str,
    *,
    request_id: str,
) -> tuple[ProbedGPULoadStoreSpec, CPULoadStoreSpec]:
    gpu = ProbedGPULoadStoreSpec(
        [_GPU_BLOCK],
        [1],
        [0],
        allocation_generations=[7],
        request_id=request_id,
        direction=direction,
        capture_ns=11,
    )
    record = OffloadAllocationRecord(
        key=make_offload_key(request_id.encode(), 0),
        physical_slot=_CPU_BLOCK,
        allocation_generation=5,
        group_idx=0,
        payload_bytes=_PAGE_BYTES,
        allocated_bytes=_PAGE_BYTES,
        producer=ReqContextSnapshot(req_id=request_id),
    )
    return gpu, CPULoadStoreSpec([_CPU_BLOCK], [record])


def _submit(
    worker: _FakeDiagnosticWorker,
    job_id: int,
    direction: str,
) -> None:
    gpu, cpu = _transfer_specs(direction, request_id=f"request-{job_id}")
    accepted = (
        worker.submit_store(job_id, gpu, cpu)
        if direction == "D2H"
        else worker.submit_load(job_id, cpu, gpu)
    )
    assert accepted


def _trace_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_probed_gpu_spec_is_pickle_stable() -> None:
    spec = ProbedGPULoadStoreSpec(
        [4],
        [1],
        [0],
        allocation_generations=[7],
        request_id="request-1",
        direction="D2H",
        capture_ns=11,
    )

    restored = pickle.loads(pickle.dumps(spec))
    assert restored.block_ids.tolist() == [4]
    assert restored.allocation_generations == (7,)
    assert restored.request_id == "request-1"
    assert restored.direction == "D2H"


def test_connector_captures_allocator_generation() -> None:
    connector = object.__new__(DAGKVDiagnosticOffloadingConnector)
    connector._dagkv_gpu_block_pool = SimpleNamespace(allocation_generations=(0, 3, 9))
    native = GPULoadStoreSpec([2], [1], [0])

    probed = connector._probe_gpu_spec(
        native,
        request_id="request-2",
        direction="H2D",
    )
    assert probed.allocation_generations == (9,)
    assert probed.block_ids.tolist() == [2]
    assert probed.capture_ns > 0

    connector._dagkv_gpu_block_pool = SimpleNamespace(allocation_generations=(0, 0, 0))
    with pytest.raises(RuntimeError, match="unallocated"):
        connector._probe_gpu_spec(
            native,
            request_id="request-2",
            direction="H2D",
        )


def test_payload_framing_excludes_padding_and_preserves_order() -> None:
    tensor = torch.arange(24, dtype=torch.int8).view(2, 12)
    refs = [[SimpleNamespace(tensor_idx=0, page_size_bytes=8)]]

    first = observe_payload([tensor], refs, 0)
    assert first.payload_bytes == 8
    assert PAYLOAD_FRAMING == "DAGKV_PAYLOAD_V1"

    tensor[0, 8:] = -1
    assert observe_payload([tensor], refs, 0) == first
    tensor[0, 0] = -1
    assert observe_payload([tensor], refs, 0).digest != first.digest


def test_payload_framing_is_domain_separated() -> None:
    tensor = torch.arange(8, dtype=torch.int8).view(1, 8)
    refs = [[SimpleNamespace(tensor_idx=0, page_size_bytes=8)]]
    observation = observe_payload([tensor], refs, 0)
    assert observation.digest != sha256(tensor.numpy().tobytes()).hexdigest()


def test_cpu_record_is_required_for_exact_generation() -> None:
    gpu = ProbedGPULoadStoreSpec(
        [4],
        [1],
        [0],
        allocation_generations=[7],
        request_id="request-3",
        direction="D2H",
        capture_ns=11,
    )
    with pytest.raises(ValueError, match="allocation record"):
        DAGKVDiagnosticCPUOffloadingWorker._validate_specs(
            1,
            gpu,
            CPULoadStoreSpec([8]),
            "D2H",
        )

    record = OffloadAllocationRecord(
        key=make_offload_key(b"content", 0),
        physical_slot=8,
        allocation_generation=5,
        group_idx=0,
        payload_bytes=4096,
        allocated_bytes=4096,
        producer=ReqContextSnapshot(req_id="request-3"),
    )
    concrete_gpu, concrete_cpu = DAGKVDiagnosticCPUOffloadingWorker._validate_specs(
        1,
        gpu,
        CPULoadStoreSpec([8], [record]),
        "D2H",
    )
    assert concrete_gpu.allocation_generations == (7,)
    assert concrete_cpu.allocation_records[0].allocation_generation == 5


def test_sink_appends_complete_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "probe.jsonl"
    sink = DurableJSONLSink(path)
    sink.append({"schema_version": "test", "value": 1})
    sink.append({"schema_version": "test", "value": 2})

    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert [row["value"] for row in rows] == [1, 2]


def test_sink_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        DurableJSONLSink("relative.jsonl")


@pytest.mark.parametrize("direction", ["D2H", "H2D"])
def test_diagnostic_worker_verifies_successful_transfer(
    tmp_path: Path,
    direction: str,
) -> None:
    trace = tmp_path / f"{direction.lower()}.jsonl"
    worker = _FakeDiagnosticWorker(trace)
    if direction == "H2D":
        worker.cpu_tensors[0][_CPU_BLOCK] = torch.arange(
            40,
            40 + _PAGE_BYTES,
            dtype=torch.int8,
        )

    _submit(worker, 10, direction)
    results = worker.get_finished()

    assert results == [
        TransferResult(job_id=10, success=True, transfer_size=_PAGE_BYTES)
    ]
    rows = _trace_rows(trace)
    assert [row["event"] for row in rows] == ["submitted", "terminal"]
    terminal = rows[-1]
    assert terminal["status"] == "completed"
    assert terminal["reported_bytes"] == _PAGE_BYTES
    assert terminal["payload_bytes"] == _PAGE_BYTES
    assert terminal["failure_reason"] is None
    assert terminal["source"]["digest"] == terminal["target"]["digest"]
    worker.shutdown()


@pytest.mark.parametrize(
    ("corrupt_target", "reported_bytes", "failure_reason"),
    [
        (True, _PAGE_BYTES, "endpoint_digest_mismatch"),
        (False, _PAGE_BYTES - 1, "reported_bytes_mismatch"),
    ],
)
def test_diagnostic_worker_fails_closed_on_payload_mismatch(
    tmp_path: Path,
    corrupt_target: bool,
    reported_bytes: int,
    failure_reason: str,
) -> None:
    trace = tmp_path / f"failure-{failure_reason}.jsonl"
    worker = _FakeDiagnosticWorker(
        trace,
        corrupt_target=corrupt_target,
        reported_bytes=reported_bytes,
    )

    _submit(worker, 20, "D2H")
    results = worker.get_finished()

    assert results == [
        TransferResult(job_id=20, success=False, transfer_size=reported_bytes)
    ]
    terminal = _trace_rows(trace)[-1]
    assert terminal["status"] == "failed"
    assert failure_reason in terminal["failure_reason"]
    assert terminal["reported_bytes"] == reported_bytes
    worker.shutdown()


def test_shutdown_drains_completed_pending_transfer(tmp_path: Path) -> None:
    trace = tmp_path / "shutdown-completed.jsonl"
    worker = _FakeDiagnosticWorker(trace, completion="wait")
    _submit(worker, 30, "D2H")

    assert [row["event"] for row in _trace_rows(trace)] == ["submitted"]
    worker.shutdown()

    rows = _trace_rows(trace)
    assert [row["event"] for row in rows] == ["submitted", "terminal"]
    assert rows[-1]["status"] == "completed"
    assert worker.waited_job_ids == {30}
    assert worker._dagkv_pending == {}
    assert worker.backend_shutdown_calls == 1

    worker.shutdown()
    assert _trace_rows(trace) == rows
    assert worker.backend_shutdown_calls == 1


def test_shutdown_fails_unverifiable_pending_transfer(tmp_path: Path) -> None:
    trace = tmp_path / "shutdown-unverifiable.jsonl"
    worker = _FakeDiagnosticWorker(trace, completion="never")
    _submit(worker, 40, "H2D")

    worker.shutdown()

    rows = _trace_rows(trace)
    assert [row["event"] for row in rows] == ["submitted", "terminal"]
    terminal = rows[-1]
    assert terminal["status"] == "failed"
    assert terminal["failure_reason"] == "shutdown_pending_unverifiable"
    assert terminal["reported_bytes"] is None
    assert terminal["source"]["tier"] == "CPU"
    assert terminal["target"] is None
    assert worker.waited_job_ids == {40}
    assert worker._dagkv_pending == {}
    assert worker._dagkv_immediate == []
    assert worker.backend_shutdown_calls == 1
