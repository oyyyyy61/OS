"""Pickle-stable scheduler-to-worker transfer identity."""

from collections.abc import Sequence
from typing import Literal

from vllm.v1.kv_offload.base import GPULoadStoreSpec

TransferDirection = Literal["D2H", "H2D"]


class ProbedGPULoadStoreSpec(GPULoadStoreSpec):
    """GPU transfer spec augmented with scheduler-observed slot generations."""

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int],
        *,
        allocation_generations: Sequence[int],
        request_id: str,
        direction: TransferDirection,
        capture_ns: int,
    ) -> None:
        super().__init__(block_ids, group_sizes, block_indices)
        generations = tuple(allocation_generations)
        if len(generations) != len(block_ids):
            raise ValueError(
                "allocation_generations must match the number of GPU block IDs"
            )
        if any(type(value) is not int or value <= 0 for value in generations):
            raise ValueError("GPU allocation generations must be positive integers")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if direction not in ("D2H", "H2D"):
            raise ValueError("direction must be D2H or H2D")
        if type(capture_ns) is not int or capture_ns <= 0:
            raise ValueError("capture_ns must be a positive integer")
        self.allocation_generations = generations
        self.request_id = request_id
        self.direction: TransferDirection = direction
        self.capture_ns = capture_ns
