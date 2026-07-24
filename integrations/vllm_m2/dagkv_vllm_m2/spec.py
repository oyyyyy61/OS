"""Single-block CPU offloading worker with end-to-end payload probes."""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from .framing import PAYLOAD_FRAMING, PayloadObservation, observe_payload
from .metadata import ProbedGPULoadStoreSpec, TransferDirection
from .sink import DurableJSONLSink

logger = logging.getLogger(__name__)
_SCHEMA = "dagkv.vllm_m2.transfer_probe.v1"


@dataclass(frozen=True, slots=True)
class _PendingProbe:
    direction: TransferDirection
    gpu_spec: ProbedGPULoadStoreSpec
    cpu_spec: CPULoadStoreSpec
    source: PayloadObservation


class DAGKVDiagnosticCPUOffloadingWorker(CPUOffloadingWorker):
    """CPU worker that verifies every one-block DMA at both endpoints."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        *,
        trace_file: str,
        run_id: str | None = None,
        phase: str | None = None,
    ) -> None:
        if block_size_factor != 1:
            raise ValueError("M2 diagnostics require block_size_factor=1")
        if len(kv_caches.group_data_refs) != 1:
            raise ValueError("M2 diagnostics require exactly one KV cache group")
        super().__init__(kv_caches, block_size_factor, num_cpu_blocks)
        self._dagkv_initialize(kv_caches, trace_file, run_id=run_id, phase=phase)

    def _dagkv_initialize(
        self,
        kv_caches: CanonicalKVCaches,
        trace_file: str,
        *,
        run_id: str | None,
        phase: str | None,
    ) -> None:
        self._dagkv_gpu_tensors = [
            item.tensor.view(torch.int8).view(-1, int(item.page_size_bytes))
            for item in kv_caches.tensors
        ]
        self._dagkv_group_refs = kv_caches.group_data_refs
        self._dagkv_sink = DurableJSONLSink(trace_file)
        self._dagkv_run_id = run_id
        self._dagkv_phase = phase
        self._dagkv_pending: dict[int, _PendingProbe] = {}
        self._dagkv_immediate: list[TransferResult] = []
        self._dagkv_lock = threading.RLock()
        self._dagkv_shutdown = False

    def _backend_submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return super().submit_store(job_id, src_spec, dst_spec)

    def _backend_submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return super().submit_load(job_id, src_spec, dst_spec)

    def _backend_get_finished(self) -> list[TransferResult]:
        return super().get_finished()

    def _backend_wait(self, job_ids: set[int]) -> None:
        super().wait(job_ids)

    def _backend_shutdown(self) -> None:
        super().shutdown()

    def _observe_gpu(self, spec: ProbedGPULoadStoreSpec) -> PayloadObservation:
        return observe_payload(
            self._dagkv_gpu_tensors,
            self._dagkv_group_refs,
            int(spec.block_ids[0]),
        )

    def _observe_cpu(self, spec: CPULoadStoreSpec) -> PayloadObservation:
        return observe_payload(
            self.cpu_tensors,
            self._dagkv_group_refs,
            int(spec.block_ids[0]),
        )

    @staticmethod
    def _validate_specs(
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        cpu_spec: LoadStoreSpec,
        direction: TransferDirection,
    ) -> tuple[ProbedGPULoadStoreSpec, CPULoadStoreSpec]:
        if type(job_id) is not int or job_id < 0:
            raise ValueError("job_id must be a non-negative integer")
        if not isinstance(gpu_spec, ProbedGPULoadStoreSpec):
            raise ValueError("GPU transfer spec is missing allocation generations")
        if not isinstance(cpu_spec, CPULoadStoreSpec):
            raise ValueError("CPU transfer spec must be CPULoadStoreSpec")
        if gpu_spec.direction != direction:
            raise ValueError("GPU transfer direction does not match submit method")
        if len(gpu_spec.block_ids) != 1 or tuple(gpu_spec.group_sizes) != (1,):
            raise ValueError("M2 diagnostics admit exactly one complete GPU block")
        if len(gpu_spec.block_indices) != 1:
            raise ValueError("M2 diagnostics require one GPU block index")
        if len(gpu_spec.allocation_generations) != 1:
            raise ValueError("M2 diagnostics require one GPU allocation generation")
        if len(cpu_spec.block_ids) != 1:
            raise ValueError("M2 diagnostics admit exactly one complete CPU block")
        if len(cpu_spec.allocation_records) != 1:
            raise ValueError("CPU allocation record is required for exact generation")
        record = cpu_spec.allocation_records[0]
        if int(cpu_spec.block_ids[0]) != record.physical_slot:
            raise ValueError("CPU allocation record slot differs from transfer spec")
        if type(record.allocation_generation) is not int or (
            record.allocation_generation <= 0
        ):
            raise ValueError("CPU allocation generation must be positive")
        if record.group_idx != 0:
            raise ValueError("M2 diagnostics require CPU allocation group zero")
        return gpu_spec, cpu_spec

    @staticmethod
    def _endpoint(
        tier: str,
        physical_slot: int,
        allocation_generation: int,
        digest: str,
    ) -> dict[str, Any]:
        return {
            "tier": tier,
            "physical_slot": physical_slot,
            "allocation_generation": allocation_generation,
            "digest": digest,
        }

    def _base_row(
        self,
        job_id: int,
        direction: TransferDirection | None,
        request_id: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA,
            "job_id": job_id,
            "request_id": request_id,
            "direction": direction,
            "run_id": self._dagkv_run_id,
            "phase": self._dagkv_phase,
            "framing": PAYLOAD_FRAMING,
        }

    def _endpoints(
        self,
        pending: _PendingProbe,
        target: PayloadObservation,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        gpu_slot = int(pending.gpu_spec.block_ids[0])
        gpu_generation = pending.gpu_spec.allocation_generations[0]
        record = pending.cpu_spec.allocation_records[0]
        gpu = self._endpoint(
            "GPU",
            gpu_slot,
            gpu_generation,
            pending.source.digest if pending.direction == "D2H" else target.digest,
        )
        cpu = self._endpoint(
            "CPU",
            record.physical_slot,
            record.allocation_generation,
            pending.source.digest if pending.direction == "H2D" else target.digest,
        )
        return (gpu, cpu) if pending.direction == "D2H" else (cpu, gpu)

    def _queue_failure(
        self,
        job_id: int,
        reason: str,
        *,
        direction: TransferDirection | None = None,
        request_id: str | None = None,
        payload_bytes: int | None = None,
        source: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
        reported_bytes: int | None = 0,
    ) -> None:
        row = self._base_row(job_id, direction, request_id)
        row.update(
            {
                "event": "terminal",
                "status": "failed",
                "terminal_ns": time.time_ns(),
                "payload_bytes": payload_bytes,
                "reported_bytes": reported_bytes,
                "source": source,
                "target": target,
                "failure_reason": reason[:240],
            }
        )
        try:
            self._dagkv_sink.append(row)
        except Exception:
            logger.exception("failed to durably append DAGKV diagnostic failure")
        self._dagkv_immediate.append(
            TransferResult(job_id=job_id, success=False, transfer_size=reported_bytes)
        )

    def _submit(
        self,
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        cpu_spec: LoadStoreSpec,
        direction: TransferDirection,
    ) -> bool:
        with self._dagkv_lock:
            if job_id in self._dagkv_pending or any(
                result.job_id == job_id for result in self._dagkv_immediate
            ):
                self._queue_failure(job_id, "duplicate_job_id", direction=direction)
                return True
            try:
                probed_gpu, concrete_cpu = self._validate_specs(
                    job_id, gpu_spec, cpu_spec, direction
                )
                source = (
                    self._observe_gpu(probed_gpu)
                    if direction == "D2H"
                    else self._observe_cpu(concrete_cpu)
                )
                record_payload_bytes = concrete_cpu.allocation_records[0].payload_bytes
                if record_payload_bytes != source.payload_bytes:
                    raise ValueError(
                        "CPU allocation record payload bytes differ from "
                        "captured payload"
                    )
                pending = _PendingProbe(direction, probed_gpu, concrete_cpu, source)
                source_endpoint, _ = self._endpoints(pending, source)
                submitted = self._base_row(job_id, direction, probed_gpu.request_id)
                submitted.update(
                    {
                        "event": "submitted",
                        "status": "in_flight",
                        "capture_ns": probed_gpu.capture_ns,
                        "submit_ns": time.time_ns(),
                        "payload_bytes": source.payload_bytes,
                        "source": source_endpoint,
                        "target": None,
                    }
                )
                self._dagkv_sink.append(submitted)
                accepted = (
                    self._backend_submit_store(job_id, probed_gpu, concrete_cpu)
                    if direction == "D2H"
                    else self._backend_submit_load(job_id, concrete_cpu, probed_gpu)
                )
                if not accepted:
                    raise RuntimeError("backend_rejected_transfer")
                self._dagkv_pending[job_id] = pending
                return True
            except Exception as exc:
                self._queue_failure(
                    job_id,
                    f"submit_probe_failed:{type(exc).__name__}:{exc}",
                    direction=direction,
                    request_id=getattr(gpu_spec, "request_id", None),
                )
                return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(job_id, src_spec, dst_spec, "D2H")

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(job_id, dst_spec, src_spec, "H2D")

    def _terminalize_backend_results(
        self, results: list[TransferResult]
    ) -> list[TransferResult]:
        verified: list[TransferResult] = []
        for result in results:
            pending = self._dagkv_pending.pop(result.job_id, None)
            if pending is None:
                self._queue_failure(
                    result.job_id,
                    "backend_returned_unknown_job",
                    reported_bytes=result.transfer_size,
                )
                continue
            try:
                target = (
                    self._observe_cpu(pending.cpu_spec)
                    if pending.direction == "D2H"
                    else self._observe_gpu(pending.gpu_spec)
                )
                source_endpoint, target_endpoint = self._endpoints(pending, target)
                mismatches: list[str] = []
                if not result.success:
                    mismatches.append("backend_reported_failure")
                if result.transfer_size != pending.source.payload_bytes:
                    mismatches.append("reported_bytes_mismatch")
                if target.payload_bytes != pending.source.payload_bytes:
                    mismatches.append("target_payload_bytes_mismatch")
                record_bytes = pending.cpu_spec.allocation_records[0].payload_bytes
                if record_bytes != pending.source.payload_bytes:
                    mismatches.append("record_payload_bytes_mismatch")
                if target.digest != pending.source.digest:
                    mismatches.append("endpoint_digest_mismatch")
                if mismatches:
                    self._queue_failure(
                        result.job_id,
                        ",".join(mismatches),
                        direction=pending.direction,
                        request_id=pending.gpu_spec.request_id,
                        payload_bytes=pending.source.payload_bytes,
                        source=source_endpoint,
                        target=target_endpoint,
                        reported_bytes=result.transfer_size,
                    )
                    continue
                row = self._base_row(
                    result.job_id,
                    pending.direction,
                    pending.gpu_spec.request_id,
                )
                row.update(
                    {
                        "event": "terminal",
                        "status": "completed",
                        "capture_ns": pending.gpu_spec.capture_ns,
                        "terminal_ns": time.time_ns(),
                        "payload_bytes": pending.source.payload_bytes,
                        "reported_bytes": result.transfer_size,
                        "source": source_endpoint,
                        "target": target_endpoint,
                        "failure_reason": None,
                    }
                )
                self._dagkv_sink.append(row)
                verified.append(result)
            except Exception as exc:
                self._queue_failure(
                    result.job_id,
                    f"terminal_probe_failed:{type(exc).__name__}:{exc}",
                    direction=pending.direction,
                    request_id=pending.gpu_spec.request_id,
                    payload_bytes=pending.source.payload_bytes,
                    reported_bytes=result.transfer_size,
                )
        return verified

    def get_finished(self) -> list[TransferResult]:
        with self._dagkv_lock:
            immediate, self._dagkv_immediate = self._dagkv_immediate, []
            verified = self._terminalize_backend_results(self._backend_get_finished())
            queued, self._dagkv_immediate = self._dagkv_immediate, []
            return [*immediate, *verified, *queued]

    def shutdown(self) -> None:
        """Synchronize and terminalize every diagnostic job before teardown."""

        with self._dagkv_lock:
            if self._dagkv_shutdown:
                return

            shutdown_failures: list[str] = []
            pending_ids = set(self._dagkv_pending)
            if pending_ids:
                try:
                    self._backend_wait(pending_ids)
                except Exception as exc:
                    shutdown_failures.append(f"wait_failed:{type(exc).__name__}:{exc}")

            try:
                finished = self._backend_get_finished()
            except Exception as exc:
                shutdown_failures.append(f"drain_failed:{type(exc).__name__}:{exc}")
            else:
                self._terminalize_backend_results(finished)

            reason = "shutdown_pending_unverifiable"
            if shutdown_failures:
                reason = f"{reason}:{';'.join(shutdown_failures)}"
            for job_id, pending in list(self._dagkv_pending.items()):
                self._dagkv_pending.pop(job_id)
                source_endpoint, _ = self._endpoints(pending, pending.source)
                self._queue_failure(
                    job_id,
                    reason,
                    direction=pending.direction,
                    request_id=pending.gpu_spec.request_id,
                    payload_bytes=pending.source.payload_bytes,
                    source=source_endpoint,
                    reported_bytes=None,
                )

            # Results cannot be consumed after the backend is torn down. Their
            # durable terminal rows above are the shutdown contract.
            self._dagkv_immediate.clear()
            self._backend_shutdown()
            self._dagkv_shutdown = True


class DAGKVDiagnosticCPUOffloadingSpec(CPUOffloadingSpec):
    """CPU spec restricted to the reproducible single-rank M2 experiment."""

    def __init__(self, vllm_config: Any, kv_cache_config: Any) -> None:
        super().__init__(vllm_config, kv_cache_config)
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("M2 diagnostics require world_size=1")
        if self.block_size_factor != 1:
            raise ValueError("M2 diagnostics require block_size_factor=1")
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("M2 diagnostics require exactly one KV cache group")
        if self.extra_config.get("fanout_layerwise_load") is not False:
            raise ValueError("M2 diagnostics require fanout_layerwise_load=false")
        trace_file = self.extra_config.get("dagkv_diagnostic_trace_file")
        if not isinstance(trace_file, str) or not trace_file:
            raise ValueError("dagkv_diagnostic_trace_file must be a non-empty string")
        self._dagkv_trace_file = trace_file
        self._dagkv_run_id = self._optional_label("dagkv_diagnostic_run_id")
        self._dagkv_phase = self._optional_label("dagkv_diagnostic_phase")

    def _optional_label(self, key: str) -> str | None:
        value = self.extra_config.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a non-empty string when provided")
        return value

    def create_worker(
        self, kv_caches: CanonicalKVCaches
    ) -> DAGKVDiagnosticCPUOffloadingWorker:
        return DAGKVDiagnosticCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            trace_file=self._dagkv_trace_file,
            run_id=self._dagkv_run_id,
            phase=self._dagkv_phase,
        )
