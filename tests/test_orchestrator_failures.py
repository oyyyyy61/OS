"""Failure, generation, DAG, and concurrency regressions."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from dagkv.domain import (
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    ExecutionRef,
    IdentityError,
    LeaseState,
    LedgerAction,
    NodeStatus,
    ReplicaId,
    StateTransitionError,
    Tier,
    TransferIntegrityError,
    TransferState,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
    WorkflowStatus,
)
from dagkv.orchestrator import LifecycleOrchestrator


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


def test_integrity_mismatch_cleans_reservation_and_rejects_stale_completion(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A mismatched DMA terminal consumes its generation and fails closed."""

    runtime = LifecycleOrchestrator(run_id="integrity-failure")
    gpu = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    cpu_v1 = ReplicaId(Tier.CPU, "numa:0", "slot", 1)
    runtime.register_gpu_block(
        block_key,
        gpu,
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    command = runtime.begin_d2h(
        block_key,
        cpu_v1,
        transfer_id="save-bad",
        timestamp_ns=2,
    )
    assert command is not None
    with pytest.raises(TransferIntegrityError, match="integrity failure"):
        runtime.complete_transfer(
            command.transfer_id,
            timestamp_ns=3,
            observed_bytes=command.byte_count - 1,
            observed_digest=command.payload_digest,
        )
    failed = runtime.transfer_snapshot(command.transfer_id)
    assert failed.state == TransferState.FAILED
    report = runtime.audit()
    assert not report.passed
    assert report.reservations == 0
    assert report.inflight_transfers == 0
    with pytest.raises(StateTransitionError, match="conflicting transfer"):
        runtime.complete_transfer(
            command.transfer_id,
            timestamp_ns=4,
            observed_bytes=command.byte_count,
            observed_digest=command.payload_digest,
        )

    retry = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 2),
        transfer_id="save-retry",
        timestamp_ns=4,
    )
    assert retry is not None
    runtime.complete_transfer(
        retry.transfer_id,
        timestamp_ns=5,
        observed_bytes=retry.byte_count,
        observed_digest=retry.payload_digest,
    )
    other_block = replace(block_key, content_digest=digest("other-content"))
    with pytest.raises(StateTransitionError, match="stale or skipped"):
        runtime.register_gpu_block(
            other_block,
            ReplicaId(Tier.GPU, "cuda:1", "new-slot", 2),
            byte_capacity=2048,
            payload_size=1536,
            payload_digest=digest("payload"),
            timestamp_ns=6,
        )


def test_stale_drop_and_reclaim_cannot_delete_new_generation(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Old policy commands cannot affect a re-admitted block generation."""

    runtime = LifecycleOrchestrator(run_id="delete-aba")
    payload_digest = digest("payload")
    gpu_v1 = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    runtime.register_gpu_block(
        block_key,
        gpu_v1,
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=payload_digest,
        timestamp_ns=1,
    )
    first = runtime.block_snapshot(block_key)
    runtime.drop_gpu(
        block_key,
        expected_gpu=gpu_v1,
        expected_location_version=first.location_version,
        timestamp_ns=2,
    )
    gpu_v2 = ReplicaId(Tier.GPU, "cuda:0", "slot", 2)
    runtime.register_gpu_block(
        block_key,
        gpu_v2,
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=payload_digest,
        timestamp_ns=3,
    )
    with pytest.raises(StateTransitionError, match="stale GPU drop"):
        runtime.drop_gpu(
            block_key,
            expected_gpu=gpu_v1,
            expected_location_version=first.location_version,
            timestamp_ns=4,
        )
    second = runtime.block_snapshot(block_key)
    assert second.replicas[Tier.GPU].replica_id == gpu_v2
    runtime.reclaim(
        block_key,
        expected_replicas=(gpu_v2,),
        expected_location_version=second.location_version,
        timestamp_ns=4,
    )
    gpu_v3 = ReplicaId(Tier.GPU, "cuda:0", "slot", 3)
    runtime.register_gpu_block(
        block_key,
        gpu_v3,
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=payload_digest,
        timestamp_ns=5,
    )
    with pytest.raises(StateTransitionError, match="stale reclaim"):
        runtime.reclaim(
            block_key,
            expected_replicas=(gpu_v2,),
            expected_location_version=second.location_version,
            timestamp_ns=6,
        )
    assert runtime.block_snapshot(block_key).replicas[Tier.GPU].replica_id == gpu_v3
    _reclaim(runtime, block_key, timestamp_ns=6)
    assert runtime.audit(require_quiescent=True).passed


def test_h2d_reservation_blocks_side_publish_and_payload_conflict(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A private target cannot be overwritten or paired with another payload."""

    runtime = LifecycleOrchestrator(run_id="reservation-isolation")
    payload_digest = digest("payload")
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=payload_digest,
        timestamp_ns=1,
    )
    save = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 1),
        transfer_id="save",
        timestamp_ns=2,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=3,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    _drop_gpu(runtime, block_key, timestamp_ns=4)
    load = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 2),
        (),
        transfer_id="prefetch",
        timestamp_ns=5,
        action=LedgerAction.PREFETCH,
    )
    assert load is not None
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="owns a reservation"):
        runtime.register_gpu_block(
            block_key,
            ReplicaId(Tier.GPU, "cuda:1", "other-slot", 1),
            byte_capacity=2048,
            payload_size=1536,
            payload_digest=payload_digest,
            timestamp_ns=6,
        )
    assert len(runtime.events) == event_count
    runtime.cancel_transfer(
        load.transfer_id,
        timestamp_ns=6,
        observed_bytes=0,
    )
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="publish GPU through H2D"):
        runtime.register_gpu_block(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot", 3),
            byte_capacity=2048,
            payload_size=1536,
            payload_digest=payload_digest,
            timestamp_ns=7,
        )
    assert len(runtime.events) == event_count
    retry = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 3),
        (),
        transfer_id="prefetch-retry",
        timestamp_ns=7,
        action=LedgerAction.PREFETCH,
    )
    assert retry is not None
    runtime.complete_transfer(
        retry.transfer_id,
        timestamp_ns=8,
        observed_bytes=retry.byte_count,
        observed_digest=retry.payload_digest,
    )
    _reclaim(runtime, block_key, timestamp_ns=9)
    with pytest.raises(TransferIntegrityError, match="canonical block payload"):
        runtime.register_gpu_block(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot", 4),
            byte_capacity=2048,
            payload_size=1536,
            payload_digest=digest("conflicting-payload"),
            timestamp_ns=10,
        )
    assert runtime.audit(require_quiescent=True).passed


def test_binding_initial_state_execution_owner_and_lease_id_are_strict(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Hidden waiting state, duplicate refs, and terminal lease reuse fail."""

    runtime = LifecycleOrchestrator(run_id="binding-strictness")
    key = WorkflowKey("strict", 0)
    runtime.register_workflow(WorkflowSpec(key, (WorkflowNode("agent"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    execution_ref = ExecutionRef(key, "request", "sequence", 0)
    first = BindingHandle(key, "request", "first")
    runtime.bind_owner(
        first,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=execution_ref,
        timestamp_ns=2,
    )
    with pytest.raises(StateTransitionError, match="already has an owner"):
        runtime.bind_owner(
            BindingHandle(key, "request", "second"),
            node_id="agent",
            block_key=block_key,
            kind=BindingKind.REQUEST,
            state=BindingState.RETAINED,
            execution_ref=execution_ref,
            timestamp_ns=3,
        )
    with pytest.raises(StateTransitionError, match="must start"):
        runtime.bind_owner(
            BindingHandle(key, "waiting", "waiting"),
            node_id="agent",
            block_key=block_key,
            kind=BindingKind.REQUEST,
            state=BindingState.WAITING,
            execution_ref=ExecutionRef(key, "waiting", "sequence", 0),
            timestamp_ns=3,
        )
    with pytest.raises(StateTransitionError, match="must start"):
        runtime.bind_owner(
            BindingHandle(key, "released", "released"),
            node_id="agent",
            block_key=block_key,
            kind=BindingKind.WORKFLOW_RETENTION,
            state=BindingState.RELEASED,
            execution_ref=None,
            timestamp_ns=3,
        )
    retention = BindingHandle(key, "retention", "retention")
    runtime.bind_owner(
        retention,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=3,
    )
    runtime.open_lease(
        retention,
        "lease",
        registered_ns=4,
        deadline_ns=5,
        reason="future reuse",
    )
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="before its deadline"):
        runtime.terminate_lease(
            retention,
            "lease",
            LeaseState.EXPIRED,
            timestamp_ns=4,
        )
    assert len(runtime.events) == event_count
    runtime.expire_leases(timestamp_ns=5)
    with pytest.raises(IdentityError, match="lease ID already used"):
        runtime.open_lease(
            retention,
            "lease",
            registered_ns=4,
            deadline_ns=5,
            reason="future reuse",
        )
    runtime.fail_workflow(key, timestamp_ns=6, error="test cleanup")
    _reclaim(runtime, block_key, timestamp_ns=7)
    assert runtime.audit(require_quiescent=True).passed


def test_invalid_waiter_cannot_leave_half_scheduled_h2d(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Waiter validation precedes allocation, ledger, and transfer mutation."""

    runtime = LifecycleOrchestrator(run_id="waiter-atomicity")
    key = WorkflowKey("waiter", 0)
    runtime.register_workflow(WorkflowSpec(key, (WorkflowNode("agent"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=1024,
        payload_size=1024,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(key, "agent", timestamp_ns=2)
    owner = BindingHandle(key, "request", "owner")
    runtime.bind_owner(
        owner,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(key, "request", "sequence", 0),
        timestamp_ns=2,
    )
    runtime._bindings[owner.binding_id].state = BindingState.WAITING
    event_count = len(runtime.events)
    with pytest.raises(
        StateTransitionError, match="invalid GPU fast-path waiter state"
    ):
        runtime.ensure_h2d(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
            (owner,),
            transfer_id="invalid-fast-path",
            timestamp_ns=2,
        )
    assert len(runtime.events) == event_count
    runtime._bindings[owner.binding_id].state = BindingState.RETAINED
    save = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 1),
        transfer_id="save",
        timestamp_ns=3,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=4,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    _drop_gpu(runtime, block_key, timestamp_ns=5)

    # Fault injection models a corrupt adapter callback with no hidden public API.
    runtime._bindings[owner.binding_id].state = BindingState.REQUIRED
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="invalid H2D waiter state"):
        runtime.ensure_h2d(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot", 2),
            (owner,),
            transfer_id="load",
            timestamp_ns=6,
        )
    assert len(runtime.events) == event_count
    block = runtime.block_snapshot(block_key)
    assert block.inflight_transfer_id is None
    assert runtime.audit().reservations == 0

    runtime._bindings[owner.binding_id].state = BindingState.RETAINED
    runtime.fail_workflow(key, timestamp_ns=6, error="test cleanup")
    _reclaim(runtime, block_key, timestamp_ns=7)
    assert runtime.audit(require_quiescent=True).passed


def test_early_workflow_failure_does_not_partially_mutate_state(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """All failure cleanup validation completes before any state transition."""

    runtime = LifecycleOrchestrator(run_id="atomic-workflow-failure")
    key = WorkflowKey("atomic", 0)
    runtime.register_workflow(WorkflowSpec(key, (WorkflowNode("agent"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=1024,
        payload_size=1024,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    owner = BindingHandle(key, "request", "owner")
    runtime.bind_owner(
        owner,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(key, "request", "sequence", 0),
        timestamp_ns=10,
    )
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="predates ledger"):
        runtime.fail_workflow(key, timestamp_ns=5, error="old callback")
    workflow = runtime.workflow_snapshot(key)
    assert workflow.status == WorkflowStatus.ACTIVE
    assert workflow.nodes["agent"].status == NodeStatus.READY
    assert runtime.binding_snapshot(owner).active
    assert len(runtime.events) == event_count
    assert runtime.audit().passed
    runtime.fail_workflow(key, timestamp_ns=11, error="valid callback")
    _reclaim(runtime, block_key, timestamp_ns=12)
    assert runtime.audit(require_quiescent=True).passed


def test_dag_failure_cancels_parallel_node_and_skips_descendants() -> None:
    """A failed fanout branch closes active siblings and downstream nodes."""

    runtime = LifecycleOrchestrator(run_id="dag-failure")
    key = WorkflowKey("diamond", 0)
    spec = WorkflowSpec(
        key,
        (
            WorkflowNode("root"),
            WorkflowNode("left", ("root",)),
            WorkflowNode("right", ("root",)),
            WorkflowNode("join", ("left", "right")),
        ),
    )
    runtime.register_workflow(spec)
    with pytest.raises(StateTransitionError, match="not ready"):
        runtime.start_node(key, "left", timestamp_ns=1)
    runtime.start_node(key, "root", timestamp_ns=1)
    runtime.complete_node(key, "root", timestamp_ns=2)
    runtime.start_node(key, "left", timestamp_ns=3)
    runtime.start_node(key, "right", timestamp_ns=4)
    runtime.fail_node(key, "left", timestamp_ns=5, error="agent exception")

    workflow = runtime.workflow_snapshot(key)
    assert workflow.status == WorkflowStatus.FAILED
    assert workflow.nodes["left"].status == NodeStatus.FAILED
    assert workflow.nodes["right"].status == NodeStatus.SKIPPED
    assert workflow.nodes["join"].status == NodeStatus.SKIPPED
    assert not runtime.fail_node(key, "left", timestamp_ns=5, error="agent exception")
    with pytest.raises(StateTransitionError, match="conflicting node failure"):
        runtime.fail_node(key, "left", timestamp_ns=6, error="agent exception")
    report = runtime.audit(require_quiescent=True)
    assert report.passed, report.issues


def test_required_binding_respects_the_dag_running_gate(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A child cannot become executable before its predecessors and start event."""

    runtime = LifecycleOrchestrator(run_id="dag-binding-gate")
    key = WorkflowKey("chain", 0)
    runtime.register_workflow(
        WorkflowSpec(
            key,
            (
                WorkflowNode("root"),
                WorkflowNode("child", ("root",)),
            ),
        )
    )
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=1024,
        payload_size=1024,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )

    rejected = BindingHandle(key, "early", "early-required")
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="running DAG node"):
        runtime.bind_owner(
            rejected,
            node_id="child",
            block_key=block_key,
            kind=BindingKind.REQUEST,
            state=BindingState.REQUIRED,
            execution_ref=ExecutionRef(key, "early", "sequence", 0),
            timestamp_ns=2,
        )
    assert len(runtime.events) == event_count

    owner = BindingHandle(key, "child-request", "child-retained")
    runtime.bind_owner(
        owner,
        node_id="child",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(key, "child-request", "sequence", 0),
        timestamp_ns=2,
    )
    assert not runtime.is_ready(owner)
    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="running DAG node"):
        runtime.ensure_h2d(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
            (owner,),
            transfer_id="invalid-fast-path",
            timestamp_ns=3,
        )
    assert len(runtime.events) == event_count
    assert runtime.binding_snapshot(owner).state == BindingState.RETAINED
    with pytest.raises(StateTransitionError, match="running DAG node"):
        runtime.set_binding_state(owner, BindingState.REQUIRED, timestamp_ns=3)

    runtime.start_node(key, "root", timestamp_ns=3)
    runtime.complete_node(key, "root", timestamp_ns=4)
    assert runtime.workflow_snapshot(key).nodes["child"].status == NodeStatus.READY
    assert not runtime.is_ready(owner)
    with pytest.raises(StateTransitionError, match="running DAG node"):
        runtime.set_binding_state(owner, BindingState.REQUIRED, timestamp_ns=5)

    runtime.start_node(key, "child", timestamp_ns=5)
    runtime.set_binding_state(owner, BindingState.REQUIRED, timestamp_ns=6)
    assert runtime.is_ready(owner)
    with pytest.raises(StateTransitionError, match="live execution mappings"):
        runtime.complete_node(key, "child", timestamp_ns=7)
    assert runtime.workflow_snapshot(key).nodes["child"].status == NodeStatus.RUNNING
    runtime.release_binding(owner, timestamp_ns=7)
    runtime.complete_node(key, "child", timestamp_ns=8)
    runtime.finish_workflow(key, timestamp_ns=9)
    _reclaim(runtime, block_key, timestamp_ns=10)
    assert runtime.audit(require_quiescent=True).passed


def test_execution_reference_is_single_use_across_binding_lifetimes(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A delayed engine reference cannot alias a later binding or block."""

    runtime = LifecycleOrchestrator(run_id="execution-ref-history")
    key = WorkflowKey("history", 0)
    other_block = replace(block_key, content_digest=digest("other-content"))
    runtime.register_workflow(WorkflowSpec(key, (WorkflowNode("agent"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-a", 1),
        byte_capacity=1024,
        payload_size=1024,
        payload_digest=digest("payload-a"),
        timestamp_ns=1,
    )
    runtime.register_gpu_block(
        other_block,
        ReplicaId(Tier.GPU, "cuda:0", "slot-b", 1),
        byte_capacity=1024,
        payload_size=1024,
        payload_digest=digest("payload-b"),
        timestamp_ns=2,
    )
    runtime.start_node(key, "agent", timestamp_ns=3)

    execution_ref = ExecutionRef(key, "request", "sequence", 0)
    first = BindingHandle(key, "request", "first-owner")
    runtime.bind_owner(
        first,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.REQUIRED,
        execution_ref=execution_ref,
        timestamp_ns=4,
    )
    runtime.set_binding_state(first, BindingState.RETAINED, timestamp_ns=5)
    runtime.set_binding_state(first, BindingState.REQUIRED, timestamp_ns=6)
    mapping_ids = [
        event.mapping_id
        for event in runtime.events
        if event.action == LedgerAction.EXEC_MAP
    ]
    assert len(mapping_ids) == len(set(mapping_ids)) == 2
    runtime.release_binding(first, timestamp_ns=7)

    event_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="already used"):
        runtime.bind_owner(
            BindingHandle(key, "request", "second-owner"),
            node_id="agent",
            block_key=other_block,
            kind=BindingKind.REQUEST,
            state=BindingState.REQUIRED,
            execution_ref=execution_ref,
            timestamp_ns=8,
        )
    assert len(runtime.events) == event_count

    runtime.complete_node(key, "agent", timestamp_ns=8)
    runtime.finish_workflow(key, timestamp_ns=9)
    _reclaim(runtime, block_key, timestamp_ns=10)
    _reclaim(runtime, other_block, timestamp_ns=10)
    assert runtime.audit(require_quiescent=True).passed


def test_workflow_failure_closes_bindings_leases_and_execution_maps(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Injected workflow failure leaves no owner-side live resource."""

    runtime = LifecycleOrchestrator(run_id="workflow-cleanup")
    key = WorkflowKey("cleanup", 0)
    spec = WorkflowSpec(key, (WorkflowNode("agent"),))
    runtime.register_workflow(spec)
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(key, "agent", timestamp_ns=2)
    request = BindingHandle(key, "request", "request-owner")
    runtime.bind_owner(
        request,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.REQUIRED,
        execution_ref=ExecutionRef(key, "request", "sequence", 0),
        timestamp_ns=3,
    )
    retention = BindingHandle(key, "retention", "retention-owner")
    runtime.bind_owner(
        retention,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=4,
    )
    runtime.open_lease(
        retention,
        "retention-lease",
        registered_ns=5,
        deadline_ns=100,
        reason="future reuse",
    )
    runtime.fail_workflow(key, timestamp_ns=6, error="injected workflow error")

    workflow = runtime.workflow_snapshot(key)
    assert workflow.status == WorkflowStatus.FAILED
    assert workflow.binding_ids == set()
    assert not runtime.is_ready(request)
    report = runtime.audit()
    assert report.passed, report.issues
    assert report.active_bindings == 0
    assert report.active_leases == 0
    assert report.execution_mappings == 0
    _reclaim(runtime, block_key, timestamp_ns=7)
    assert runtime.audit(require_quiescent=True).passed


def test_released_h2d_waiter_is_not_published_after_completion(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Release racing with DMA completion cannot create a use-after-free map."""

    runtime = LifecycleOrchestrator(run_id="released-waiter")
    key = WorkflowKey("release-race", 0)
    spec = WorkflowSpec(key, (WorkflowNode("agent"),))
    runtime.register_workflow(spec)
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(key, "agent", timestamp_ns=2)
    owner = BindingHandle(key, "request", "owner")
    runtime.bind_owner(
        owner,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.REQUIRED,
        execution_ref=ExecutionRef(key, "request", "sequence", 0),
        timestamp_ns=3,
    )
    runtime.set_binding_state(owner, BindingState.RETAINED, timestamp_ns=4)
    save = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 1),
        transfer_id="save",
        timestamp_ns=5,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=6,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    _drop_gpu(runtime, block_key, timestamp_ns=7)
    load = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 2),
        (owner,),
        transfer_id="load",
        timestamp_ns=8,
    )
    assert load is not None
    runtime.release_binding(owner, timestamp_ns=9)
    assert runtime.complete_transfer(
        load.transfer_id,
        timestamp_ns=10,
        observed_bytes=load.byte_count,
        observed_digest=load.payload_digest,
    )
    assert not runtime.complete_transfer(
        load.transfer_id,
        timestamp_ns=10,
        observed_bytes=load.byte_count,
        observed_digest=load.payload_digest,
    )
    assert not runtime.is_ready(owner)
    report = runtime.audit()
    assert report.passed, report.issues
    assert report.execution_mappings == 0
    runtime.complete_node(key, "agent", timestamp_ns=11)
    runtime.finish_workflow(key, timestamp_ns=12)
    _reclaim(runtime, block_key, timestamp_ns=13)
    assert runtime.audit(require_quiescent=True).passed


def test_h2d_completion_does_not_publish_a_terminal_node_waiter(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A node completing during DMA loses execution publication eligibility."""

    runtime = LifecycleOrchestrator(run_id="terminal-node-waiter")
    key = WorkflowKey("completion-race", 0)
    runtime.register_workflow(WorkflowSpec(key, (WorkflowNode("agent"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(key, "agent", timestamp_ns=2)
    owner = BindingHandle(key, "request", "owner")
    runtime.bind_owner(
        owner,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.REQUIRED,
        execution_ref=ExecutionRef(key, "request", "sequence", 0),
        timestamp_ns=3,
    )
    runtime.set_binding_state(owner, BindingState.RETAINED, timestamp_ns=4)
    save = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 1),
        transfer_id="save",
        timestamp_ns=5,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=6,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    _drop_gpu(runtime, block_key, timestamp_ns=7)
    load = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 2),
        (owner,),
        transfer_id="load",
        timestamp_ns=8,
    )
    assert load is not None
    assert runtime.binding_snapshot(owner).state == BindingState.WAITING

    runtime.complete_node(key, "agent", timestamp_ns=9)
    runtime.complete_transfer(
        load.transfer_id,
        timestamp_ns=10,
        observed_bytes=load.byte_count,
        observed_digest=load.payload_digest,
    )
    assert runtime.binding_snapshot(owner).state == BindingState.RETAINED
    assert not runtime.is_ready(owner)
    report = runtime.audit()
    assert report.passed, report.issues
    assert report.execution_mappings == 0

    runtime.finish_workflow(key, timestamp_ns=11)
    _reclaim(runtime, block_key, timestamp_ns=12)
    assert runtime.audit(require_quiescent=True).passed


def test_concurrent_h2d_waiters_share_one_physical_transfer(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Concurrent callers attach to a single target reservation and DMA."""

    runtime = LifecycleOrchestrator(run_id="single-flight")
    key = WorkflowKey("fanout", 0)
    spec = WorkflowSpec(key, (WorkflowNode("agent"),))
    runtime.register_workflow(spec)
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(key, "agent", timestamp_ns=2)
    owners: list[BindingHandle] = []
    for index in range(8):
        handle = BindingHandle(key, f"request-{index}", f"owner-{index}")
        runtime.bind_owner(
            handle,
            node_id="agent",
            block_key=block_key,
            kind=BindingKind.REQUEST,
            state=BindingState.REQUIRED,
            execution_ref=ExecutionRef(key, handle.request_id, f"seq-{index}", 0),
            timestamp_ns=3 + index,
        )
        owners.append(handle)
    for index, handle in enumerate(owners):
        runtime.set_binding_state(
            handle,
            BindingState.RETAINED,
            timestamp_ns=20 + index,
        )
    save = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 1),
        transfer_id="save",
        timestamp_ns=30,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=31,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    _drop_gpu(runtime, block_key, timestamp_ns=32)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot", 2)

    def ensure(index: int):
        return runtime.ensure_h2d(
            block_key,
            target,
            (owners[index],),
            transfer_id=f"load-{index}",
            timestamp_ns=33,
            action=LedgerAction.LOAD,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        commands = list(executor.map(ensure, range(len(owners))))
    transfer_ids = {command.transfer_id for command in commands if command}
    assert len(transfer_ids) == 1
    transfer_id = transfer_ids.pop()
    command = next(command for command in commands if command is not None)
    scheduled_loads = [
        event
        for event in runtime.events
        if event.action == LedgerAction.LOAD and event.status.value == "scheduled"
    ]
    assert len(scheduled_loads) == 1
    runtime.complete_transfer(
        transfer_id,
        timestamp_ns=34,
        observed_bytes=command.byte_count,
        observed_digest=command.payload_digest,
    )
    assert all(runtime.is_ready(owner) for owner in owners)

    for owner in owners:
        runtime.release_binding(owner, timestamp_ns=35)
    runtime.complete_node(key, "agent", timestamp_ns=36)
    runtime.finish_workflow(key, timestamp_ns=37)
    _reclaim(runtime, block_key, timestamp_ns=38)
    report = runtime.audit(require_quiescent=True)
    assert report.passed, report.issues


def test_workflow_failure_racing_h2d_completion_is_serializable(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A failed owner cannot be remapped by a concurrent DMA callback."""

    runtime = LifecycleOrchestrator(run_id="failure-completion-race")
    key_a = WorkflowKey("workflow-a", 0)
    key_b = WorkflowKey("workflow-b", 0)
    runtime.register_workflow(WorkflowSpec(key_a, (WorkflowNode("agent"),)))
    runtime.register_workflow(WorkflowSpec(key_b, (WorkflowNode("agent"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 1),
        byte_capacity=2048,
        payload_size=1536,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(key_a, "agent", timestamp_ns=2)
    runtime.start_node(key_b, "agent", timestamp_ns=3)
    owners: list[BindingHandle] = []
    for index, key in enumerate((key_a, key_b)):
        handle = BindingHandle(key, f"request-{index}", f"owner-{index}")
        runtime.bind_owner(
            handle,
            node_id="agent",
            block_key=block_key,
            kind=BindingKind.REQUEST,
            state=BindingState.REQUIRED,
            execution_ref=ExecutionRef(key, handle.request_id, "sequence", 0),
            timestamp_ns=4 + index,
        )
        owners.append(handle)
    runtime.set_binding_state(owners[0], BindingState.RETAINED, timestamp_ns=6)
    runtime.set_binding_state(owners[1], BindingState.RETAINED, timestamp_ns=7)
    save = runtime.begin_d2h(
        block_key,
        ReplicaId(Tier.CPU, "numa:0", "slot", 1),
        transfer_id="save",
        timestamp_ns=8,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=9,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    _drop_gpu(runtime, block_key, timestamp_ns=10)
    load = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot", 2),
        (owners[0],),
        transfer_id="load",
        timestamp_ns=11,
    )
    assert load is not None
    assert (
        runtime.ensure_h2d(
            block_key,
            load.target_replica,
            (owners[1],),
            transfer_id="joined-load",
            timestamp_ns=11,
        )
        == load
    )

    barrier = Barrier(3)

    def fail_first_workflow() -> bool:
        barrier.wait()
        return runtime.fail_workflow(
            key_a,
            timestamp_ns=12,
            error="injected failure",
        )

    def finish_dma() -> bool:
        barrier.wait()
        return runtime.complete_transfer(
            load.transfer_id,
            timestamp_ns=12,
            observed_bytes=load.byte_count,
            observed_digest=load.payload_digest,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        failed = executor.submit(fail_first_workflow)
        completed = executor.submit(finish_dma)
        barrier.wait()
        assert failed.result()
        assert completed.result()

    assert not runtime.is_ready(owners[0])
    assert runtime.is_ready(owners[1])
    report = runtime.audit()
    assert report.passed, report.issues
    assert report.execution_mappings == 1
    runtime.release_binding(owners[1], timestamp_ns=13)
    runtime.fail_workflow(key_b, timestamp_ns=13, error="test cleanup")
    _reclaim(runtime, block_key, timestamp_ns=14)
    assert runtime.audit(require_quiescent=True).passed
