"""Engine observations may close transfers only through canonical APIs."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from dagkv.domain import (
    BlockKey,
    LedgerAction,
    ReplicaId,
    Tier,
    TransferDirection,
    TransferIntegrityError,
    TransferState,
)
from dagkv.engine_adapter import (
    DMATerminalObservation,
    EngineFingerprint,
    PhysicalEndpoint,
    commit_transfer_observation,
)
from dagkv.orchestrator import LifecycleOrchestrator


def _fingerprint(digest: Callable[[str], str]) -> EngineFingerprint:
    return EngineFingerprint(
        engine="vllm",
        version="0.1.dev",
        source_revision="2e5d72f",
        source_diff_digest=digest("vllm-diff"),
        model_fingerprint="qwen3-8b-local",
    )


def _d2h(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> tuple[LifecycleOrchestrator, object, str]:
    payload_digest = digest("payload")
    runtime = LifecycleOrchestrator(run_id="adapter-test")
    gpu = ReplicaId(Tier.GPU, "cuda:0", "gpu-slot-3", 1)
    cpu = ReplicaId(Tier.CPU, "host:0", "cpu-slot-7", 1)
    runtime.register_gpu_block(
        block_key,
        gpu,
        byte_capacity=4096,
        payload_size=4096,
        payload_digest=payload_digest,
        timestamp_ns=1,
    )
    command = runtime.begin_d2h(
        block_key,
        cpu,
        transfer_id="d2h-1",
        timestamp_ns=2,
    )
    assert command is not None
    return runtime, command, payload_digest


def _observation(
    command: object,
    payload_digest: str,
    fingerprint: EngineFingerprint,
    *,
    success: bool = True,
    target_digest: str | None = None,
) -> DMATerminalObservation:
    submitted_ns = 2 if command.direction == TransferDirection.D2H else 5
    return DMATerminalObservation(
        engine_job_id="job-11",
        transfer_id=command.transfer_id,
        direction=command.direction,
        source=PhysicalEndpoint.from_replica(
            command.source_replica,
            generation_source="allocator-source",
        ),
        target=PhysicalEndpoint.from_replica(
            command.target_replica,
            generation_source="allocator-target",
        ),
        payload_bytes=command.byte_count,
        reported_bytes=command.byte_count if success else 0,
        source_digest=payload_digest,
        target_digest=(payload_digest if target_digest is None else target_digest)
        if success
        else target_digest,
        submitted_ns=submitted_ns,
        terminal_ns=submitted_ns + 1,
        engine_success=success,
        error=None if success else "worker copy failed",
        fingerprint=fingerprint,
    )


def test_successful_observation_commits_and_replays_exactly(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, command, payload_digest = _d2h(block_key, digest)
    observation = _observation(command, payload_digest, _fingerprint(digest))

    assert commit_transfer_observation(runtime, command, observation) is True
    assert commit_transfer_observation(runtime, command, observation) is False
    transfer = runtime.transfer_snapshot(command.transfer_id)
    assert transfer.state == TransferState.COMPLETED
    assert transfer.observed_digest == payload_digest


def test_successful_h2d_uses_same_projection(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, d2h, payload_digest = _d2h(block_key, digest)
    fingerprint = _fingerprint(digest)
    commit_transfer_observation(
        runtime,
        d2h,
        _observation(d2h, payload_digest, fingerprint),
    )
    snapshot = runtime.block_snapshot(block_key)
    runtime.drop_gpu(
        block_key,
        expected_gpu=d2h.source_replica,
        expected_location_version=snapshot.location_version,
        timestamp_ns=4,
    )
    target = ReplicaId(Tier.GPU, "cuda:0", "gpu-slot-3", 2)
    h2d = runtime.ensure_h2d(
        block_key,
        target,
        (),
        transfer_id="h2d-1",
        timestamp_ns=5,
        action=LedgerAction.PREFETCH,
    )
    assert h2d is not None
    observation = _observation(h2d, payload_digest, fingerprint)
    assert observation.direction == TransferDirection.H2D
    assert commit_transfer_observation(runtime, h2d, observation) is True


def test_digest_mismatch_terminalizes_before_raising(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, command, payload_digest = _d2h(block_key, digest)
    observation = _observation(
        command,
        payload_digest,
        _fingerprint(digest),
        target_digest=digest("corrupt"),
    )

    with pytest.raises(TransferIntegrityError, match="target payload digest"):
        commit_transfer_observation(runtime, command, observation)
    assert runtime.transfer_snapshot(command.transfer_id).state == TransferState.FAILED
    assert Tier.CPU not in runtime.block_snapshot(block_key).replicas


def test_generation_mismatch_terminalizes_before_raising(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, command, payload_digest = _d2h(block_key, digest)
    original = _observation(command, payload_digest, _fingerprint(digest))
    row = original.to_dict()
    row["target"]["generation"] = 2
    observation = DMATerminalObservation.from_dict(row)

    with pytest.raises(TransferIntegrityError, match="target allocation generation"):
        commit_transfer_observation(runtime, command, observation)
    assert runtime.transfer_snapshot(command.transfer_id).state == TransferState.FAILED


def test_explicit_worker_failure_closes_reservation(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, command, payload_digest = _d2h(block_key, digest)
    observation = _observation(
        command,
        payload_digest,
        _fingerprint(digest),
        success=False,
    )

    assert commit_transfer_observation(runtime, command, observation) is True
    assert commit_transfer_observation(runtime, command, observation) is False
    assert runtime.transfer_snapshot(command.transfer_id).state == TransferState.FAILED


def test_json_roundtrip_rejects_unknown_or_tampered_fields(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    _, command, payload_digest = _d2h(block_key, digest)
    observation = _observation(command, payload_digest, _fingerprint(digest))
    assert DMATerminalObservation.from_dict(observation.to_dict()) == observation

    unknown = deepcopy(observation.to_dict())
    unknown["repaired_offline"] = True
    with pytest.raises(ValueError, match="fields differ"):
        DMATerminalObservation.from_dict(unknown)

    bad_schema = deepcopy(observation.to_dict())
    bad_schema["schema_version"] = "future"
    with pytest.raises(ValueError, match="unsupported"):
        DMATerminalObservation.from_dict(bad_schema)
