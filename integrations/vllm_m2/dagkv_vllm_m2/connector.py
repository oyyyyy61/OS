"""External vLLM connector that captures real GPU allocation generations."""

import time
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import GPULoadStoreSpec

from .metadata import ProbedGPULoadStoreSpec, TransferDirection

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class DAGKVDiagnosticOffloadingConnector(OffloadingConnector):
    """OffloadingConnector with scheduler-time GPU generation capture."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self._dagkv_gpu_block_pool: BlockPool | None = None
        super().__init__(vllm_config, role, kv_cache_config)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._dagkv_gpu_block_pool = gpu_block_pool

    def _probe_gpu_spec(
        self,
        spec: GPULoadStoreSpec,
        *,
        request_id: str,
        direction: TransferDirection,
    ) -> ProbedGPULoadStoreSpec:
        pool = self._dagkv_gpu_block_pool
        if pool is None:
            raise RuntimeError("GPU block pool was not bound before metadata capture")
        block_ids = [int(value) for value in spec.block_ids]
        all_generations = pool.allocation_generations
        generations: list[int] = []
        for block_id in block_ids:
            if block_id < 0 or block_id >= len(all_generations):
                raise RuntimeError("GPU transfer references a slot outside the pool")
            generation = all_generations[block_id]
            if type(generation) is not int or generation <= 0:
                raise RuntimeError("GPU transfer references an unallocated slot")
            generations.append(generation)
        return ProbedGPULoadStoreSpec(
            block_ids,
            tuple(int(value) for value in spec.group_sizes),
            tuple(int(value) for value in spec.block_indices),
            allocation_generations=generations,
            request_id=request_id,
            direction=direction,
            capture_ns=time.time_ns(),
        )

    def _probe_job(self, job: TransferJob, direction: TransferDirection) -> None:
        if direction == "D2H":
            if not isinstance(job.src_spec, GPULoadStoreSpec):
                raise TypeError("D2H job source must be a GPU load/store spec")
            job.src_spec = self._probe_gpu_spec(
                job.src_spec,
                request_id=job.req_id,
                direction=direction,
            )
            return
        if not isinstance(job.dst_spec, GPULoadStoreSpec):
            raise TypeError("H2D job target must be a GPU load/store spec")
        job.dst_spec = self._probe_gpu_spec(
            job.dst_spec,
            request_id=job.req_id,
            direction=direction,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        if not isinstance(meta, OffloadingConnectorMetadata):
            raise TypeError("offloading connector returned unexpected metadata")
        for job in meta.store_jobs.values():
            self._probe_job(job, "D2H")
        for job in meta.load_jobs.values():
            self._probe_job(job, "H2D")
        return meta
