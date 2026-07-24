"""End-to-end component lifecycle tests for shared DAG owners."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dagkv.domain import (
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    ExecutionRef,
    IdentityError,
    LedgerAction,
    ReplicaId,
    StateTransitionError,
    Tier,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)
from dagkv.orchestrator import LifecycleOrchestrator


def _workflow(workflow_id: str, node_id: str) -> WorkflowSpec:
    return WorkflowSpec(
        WorkflowKey(workflow_id, 0),
        (WorkflowNode(node_id),),
    )


def _request_binding(
    runtime: LifecycleOrchestrator,
    workflow: WorkflowSpec,
    node_id: str,
    binding_id: str,
    block_key: BlockKey,
    *,
    timestamp_ns: int,
) -> BindingHandle:
    handle = BindingHandle(workflow.key, f"request-{binding_id}", binding_id)
    runtime.bind_owner(
        handle,
        node_id=node_id,
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.REQUIRED,
        execution_ref=ExecutionRef(
            workflow.key,
            handle.request_id,
            f"sequence-{binding_id}",
            0,
        ),
        timestamp_ns=timestamp_ns,
    )
    return handle


def _drop_gpu(
    runtime: LifecycleOrchestrator,
    block_key: BlockKey,
    *,
    timestamp_ns: int,
) -> bool:
    block = runtime.block_snapshot(block_key)
    return runtime.drop_gpu(
        block_key,
        expected_gpu=block.replicas[Tier.GPU].replica_id,
        expected_location_version=block.location_version,
        timestamp_ns=timestamp_ns,
    )


def _reclaim(
    runtime: LifecycleOrchestrator,
    block_key: BlockKey,
    *,
    timestamp_ns: int,
) -> bool:
    block = runtime.block_snapshot(block_key)
    return runtime.reclaim(
        block_key,
        expected_replicas=tuple(
            replica.replica_id for replica in block.replicas.values()
        ),
        expected_location_version=block.location_version,
        timestamp_ns=timestamp_ns,
    )


def test_two_owner_offload_readmission_and_reclaim(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Shared owners survive one D2H/H2D and release independently."""

    runtime = LifecycleOrchestrator(run_id="shared-lifecycle")
    workflow_a = _workflow("workflow-a", "agent-a")
    workflow_b = _workflow("workflow-b", "agent-b")
    runtime.register_workflow(workflow_a)
    runtime.register_workflow(workflow_b)
    payload_digest = digest("kv-payload")
    gpu_v1 = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1)
    cpu_v1 = ReplicaId(Tier.CPU, "numa:0", "slot-0", 1)
    gpu_v2 = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    runtime.register_gpu_block(
        block_key,
        gpu_v1,
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=payload_digest,
        timestamp_ns=1,
    )
    runtime.start_node(workflow_a.key, "agent-a", timestamp_ns=2)
    runtime.start_node(workflow_b.key, "agent-b", timestamp_ns=3)
    owner_a = _request_binding(
        runtime,
        workflow_a,
        "agent-a",
        "owner-a",
        block_key,
        timestamp_ns=4,
    )
    owner_b = _request_binding(
        runtime,
        workflow_b,
        "agent-b",
        "owner-b",
        block_key,
        timestamp_ns=5,
    )
    retention = BindingHandle(workflow_a.key, "retention-a", "retention-a")
    runtime.bind_owner(
        retention,
        node_id="agent-a",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=6,
    )
    runtime.open_lease(
        retention,
        "lease-a",
        registered_ns=7,
        deadline_ns=10,
        reason="predicted_future_fanout",
    )
    with pytest.raises(StateTransitionError, match="active leases"):
        runtime.begin_d2h(
            block_key,
            cpu_v1,
            transfer_id="save-block",
            timestamp_ns=8,
        )
    assert runtime.expire_leases(timestamp_ns=10) == ("lease-a",)
    assert runtime.release_binding(retention, timestamp_ns=11)

    save = runtime.begin_d2h(
        block_key,
        cpu_v1,
        transfer_id="save-block",
        timestamp_ns=12,
    )
    assert save is not None and save.action == LedgerAction.SAVE
    assert runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=13,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    runtime.set_binding_state(owner_a, BindingState.RETAINED, timestamp_ns=14)
    runtime.set_binding_state(owner_b, BindingState.RETAINED, timestamp_ns=15)
    assert _drop_gpu(runtime, block_key, timestamp_ns=16)
    assert not runtime.is_ready(owner_a)
    assert not runtime.is_ready(owner_b)

    load = runtime.ensure_h2d(
        block_key,
        gpu_v2,
        (owner_a,),
        transfer_id="load-block-a",
        timestamp_ns=17,
    )
    joined = runtime.ensure_h2d(
        block_key,
        gpu_v2,
        (owner_b,),
        transfer_id="load-block-b",
        timestamp_ns=18,
    )
    assert load is not None
    assert joined == load
    assert runtime.complete_transfer(
        load.transfer_id,
        timestamp_ns=19,
        observed_bytes=load.byte_count,
        observed_digest=load.payload_digest,
    )
    assert runtime.is_ready(owner_a)
    assert runtime.is_ready(owner_b)

    assert runtime.release_binding(owner_a, timestamp_ns=20)
    assert not runtime.release_binding(owner_a, timestamp_ns=21)
    assert runtime.is_ready(owner_b)
    with pytest.raises(StateTransitionError, match="live owner"):
        _reclaim(runtime, block_key, timestamp_ns=21)

    wrong_owner = BindingHandle(workflow_a.key, owner_b.request_id, owner_b.binding_id)
    with pytest.raises(IdentityError, match="capability"):
        runtime.release_binding(wrong_owner, timestamp_ns=22)
    assert runtime.release_binding(owner_b, timestamp_ns=22)
    runtime.complete_node(workflow_a.key, "agent-a", timestamp_ns=23)
    runtime.complete_node(workflow_b.key, "agent-b", timestamp_ns=24)
    runtime.finish_workflow(workflow_a.key, timestamp_ns=25)
    runtime.finish_workflow(workflow_b.key, timestamp_ns=26)
    assert not runtime.complete_node(workflow_a.key, "agent-a", timestamp_ns=23)
    assert _reclaim(runtime, block_key, timestamp_ns=27)

    report = runtime.audit(require_quiescent=True)
    assert report.passed, report.issues
    assert report.live_replicas == 0
    assert report.active_bindings == 0
    assert report.inflight_transfers == 0


def test_cross_owner_release_stays_rejected_after_real_owner_release(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A released binding remains owner-qualified for duplicate calls."""

    runtime = LifecycleOrchestrator(run_id="owner-capability")
    first = _workflow("first", "agent")
    second = _workflow("second", "agent")
    runtime.register_workflow(first)
    runtime.register_workflow(second)
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=1024,
        payload_size=1024,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    owner = BindingHandle(first.key, "request", "binding")
    runtime.bind_owner(
        owner,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(first.key, "request", "sequence", 0),
        timestamp_ns=2,
    )
    assert runtime.release_binding(owner, timestamp_ns=3)
    forged = BindingHandle(second.key, "request", "binding")
    with pytest.raises(IdentityError, match="capability"):
        runtime.release_binding(forged, timestamp_ns=4)
