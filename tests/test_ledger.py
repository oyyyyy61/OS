"""Transactional ledger validation tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from dagkv.domain import (
    BindingKind,
    BlockKey,
    ExecutionRef,
    LedgerAction,
    LedgerStatus,
    ReplicaId,
    StateTransitionError,
    Tier,
    WorkflowKey,
)
from dagkv.ledger import EventDraft, EventLedger


def test_invalid_batch_rolls_back_all_rows() -> None:
    """A bad child row cannot leave its allocation parent behind."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    drafts = (
        EventDraft(
            action=LedgerAction.NODE,
            status=LedgerStatus.SCHEDULED,
            reason="start",
            timestamp_ns=1,
            operation_id="node:workflow:0:a",
            local_id="start",
            node_id="a",
        ),
        EventDraft(
            action=LedgerAction.NODE,
            status=LedgerStatus.COMPLETED,
            reason="done",
            timestamp_ns=2,
            operation_id="node:workflow:0:a",
            parent_local_id="start",
            node_id="a",
        ),
    )
    with pytest.raises(StateTransitionError, match="requires workflow"):
        ledger.append_batch(drafts)
    assert ledger.events == ()


def test_ledger_rejects_global_time_reversal() -> None:
    """Event order cannot silently repair a regressing clock."""

    from dagkv.domain import WorkflowKey

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    workflow = WorkflowKey("workflow", 0)
    ledger.append(
        EventDraft(
            action=LedgerAction.NODE,
            status=LedgerStatus.SCHEDULED,
            reason="start",
            timestamp_ns=2,
            operation_id="node:workflow:0:a",
            workflow=workflow,
            node_id="a",
        )
    )
    with pytest.raises(StateTransitionError, match="non-decreasing"):
        ledger.append(
            EventDraft(
                action=LedgerAction.NODE,
                status=LedgerStatus.SCHEDULED,
                reason="late row with old clock",
                timestamp_ns=1,
                operation_id="node:workflow:0:b",
                workflow=workflow,
                node_id="b",
            )
        )
    assert len(ledger.events) == 1


def test_audit_rejects_tampered_event_envelope_and_unknown_parent() -> None:
    """Frozen evidence cannot rewrite its ledger identity or parent graph."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    workflow = WorkflowKey("workflow", 0)
    started = ledger.append(
        EventDraft(
            action=LedgerAction.NODE,
            status=LedgerStatus.SCHEDULED,
            reason="start",
            timestamp_ns=1,
            operation_id="node:workflow:0:agent",
            workflow=workflow,
            node_id="agent",
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.NODE,
            status=LedgerStatus.COMPLETED,
            reason="done",
            timestamp_ns=2,
            operation_id="node:workflow:0:agent",
            parent_event_id=started.event_id,
            workflow=workflow,
            node_id="agent",
        )
    )
    assert ledger.audit(require_quiescent=True) == ()

    ledger._events[0] = replace(
        started,
        schema_version="tampered-schema",
        event_id="tampered-event",
        parent_event_id="ghost-parent",
        run_id="other-run",
        phase="other-phase",
        source="other-source",
    )
    issues = ledger.audit(require_quiescent=True)
    assert any("schema_version changed" in issue for issue in issues)
    assert any("event ID mismatch" in issue for issue in issues)
    assert any("unknown parent event" in issue for issue in issues)
    assert any("run_id changed" in issue for issue in issues)
    assert any("phase changed" in issue for issue in issues)
    assert any("source changed" in issue for issue in issues)


def test_transfer_terminal_cannot_change_frozen_geometry(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Independent replay rejects a terminal with rewritten tiers or payload."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    gpu = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    cpu = ReplicaId(Tier.CPU, "numa:0", "slot", 1)
    payload_a = digest("payload-a")
    payload_b = digest("payload-b")
    source_allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="source allocation",
            timestamp_ns=0,
            operation_id="allocation:gpu",
            block_key=block_key,
            blocks=(gpu,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_a,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.MAP,
            status=LedgerStatus.COMPLETED,
            reason="source published",
            timestamp_ns=0,
            operation_id="content-map:gpu",
            parent_event_id=source_allocation.event_id,
            block_key=block_key,
            blocks=(gpu,),
            mapping_id="content-map:gpu",
            payload_size=32,
            payload_digest=payload_a,
        )
    )
    allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="target reservation",
            timestamp_ns=1,
            operation_id="allocation:cpu",
            block_key=block_key,
            blocks=(cpu,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_a,
        )
    )
    scheduled = ledger.append(
        EventDraft(
            action=LedgerAction.SAVE,
            status=LedgerStatus.SCHEDULED,
            reason="save",
            timestamp_ns=2,
            operation_id="save",
            parent_event_id=allocation.event_id,
            block_key=block_key,
            blocks=(cpu,),
            transfer_id="save",
            source_tier=gpu.tier,
            target_tier=cpu.tier,
            byte_count=32,
            payload_digest=payload_a,
        )
    )
    with pytest.raises(StateTransitionError, match="geometry changed"):
        ledger.append(
            EventDraft(
                action=LedgerAction.SAVE,
                status=LedgerStatus.COMPLETED,
                reason="tampered terminal",
                timestamp_ns=3,
                operation_id="save",
                parent_event_id=scheduled.event_id,
                block_key=block_key,
                blocks=(cpu,),
                transfer_id="save",
                source_tier=gpu.tier,
                target_tier=cpu.tier,
                byte_count=1,
                observed_byte_count=1,
                payload_digest=payload_b,
                observed_digest=payload_b,
            )
        )
    assert len(ledger.events) == 4


def test_binding_parent_freezes_execution_identity(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Execution maps cannot substitute another engine reference."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    workflow = WorkflowKey("workflow", 0)
    replica = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    ref_a = ExecutionRef(workflow, "request", "sequence-a", 0)
    ref_b = ExecutionRef(workflow, "request", "sequence-b", 0)
    payload_digest = digest("payload")
    allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="allocate",
            timestamp_ns=1,
            operation_id="allocation",
            block_key=block_key,
            blocks=(replica,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    content_map = ledger.append(
        EventDraft(
            action=LedgerAction.MAP,
            status=LedgerStatus.COMPLETED,
            reason="publish",
            timestamp_ns=1,
            operation_id="content-map",
            parent_event_id=allocation.event_id,
            block_key=block_key,
            blocks=(replica,),
            mapping_id="content-map",
            payload_size=32,
            payload_digest=payload_digest,
        )
    )
    binding = ledger.append(
        EventDraft(
            action=LedgerAction.BIND,
            status=LedgerStatus.COMPLETED,
            reason="bind",
            timestamp_ns=2,
            operation_id="binding",
            workflow=workflow,
            request_id="request",
            node_id="agent",
            block_key=block_key,
            binding_id="binding",
            binding_kind=BindingKind.REQUEST,
            execution_ref=ref_a,
        )
    )
    invalid = EventDraft(
        action=LedgerAction.EXEC_MAP,
        status=LedgerStatus.COMPLETED,
        reason="map",
        timestamp_ns=3,
        operation_id="execution-map",
        parent_event_id=binding.event_id,
        workflow=workflow,
        request_id="request",
        node_id="agent",
        block_key=block_key,
        blocks=(replica,),
        binding_id="binding",
        binding_kind=BindingKind.REQUEST,
        mapping_id="execution-map",
        execution_ref=ref_b,
    )
    with pytest.raises(StateTransitionError, match="binding lineage changed"):
        ledger.append(invalid)
    assert len(ledger.events) == 3
    with pytest.raises(StateTransitionError, match="running node lifecycle"):
        ledger.append(replace(invalid, execution_ref=ref_a))
    node_started = ledger.append(
        EventDraft(
            action=LedgerAction.NODE,
            status=LedgerStatus.SCHEDULED,
            reason="start",
            timestamp_ns=3,
            operation_id="node:workflow:0:agent",
            workflow=workflow,
            node_id="agent",
        )
    )
    valid_map = ledger.append(replace(invalid, execution_ref=ref_a))
    assert ledger.audit() == ()
    with pytest.raises(StateTransitionError, match="live execution mapping"):
        ledger.append(
            EventDraft(
                action=LedgerAction.NODE,
                status=LedgerStatus.COMPLETED,
                reason="premature terminal",
                timestamp_ns=4,
                operation_id="node:workflow:0:agent",
                parent_event_id=node_started.event_id,
                workflow=workflow,
                node_id="agent",
            )
        )
    with pytest.raises(StateTransitionError, match="live execution mapping"):
        ledger.append(
            EventDraft(
                action=LedgerAction.UNMAP,
                status=LedgerStatus.COMPLETED,
                reason="early unmap",
                timestamp_ns=4,
                operation_id="content-map",
                parent_event_id=content_map.event_id,
                block_key=block_key,
                blocks=(replica,),
                mapping_id="content-map",
                payload_size=32,
                payload_digest=payload_digest,
            )
        )
    with pytest.raises(StateTransitionError, match="live execution mapping"):
        ledger.append(
            EventDraft(
                action=LedgerAction.RELEASE,
                status=LedgerStatus.COMPLETED,
                reason="early release",
                timestamp_ns=4,
                operation_id="binding",
                parent_event_id=binding.event_id,
                workflow=workflow,
                request_id="request",
                node_id="agent",
                block_key=block_key,
                binding_id="binding",
                binding_kind=BindingKind.REQUEST,
                execution_ref=ref_a,
            )
        )
    assert len(ledger.events) == 5

    # Simulate a corrupted persisted row to verify independent replay as well.
    missing_replica = ReplicaId(Tier.GPU, "cuda:1", "missing", 1)
    ledger._events[valid_map.sequence] = replace(
        valid_map,
        blocks=(missing_replica,),
    )
    assert any("live physical content mapping" in issue for issue in ledger.audit())
    ledger._events[valid_map.sequence] = replace(valid_map, execution_ref=ref_b)
    assert any("binding lineage changed" in issue for issue in ledger.audit())


def test_binding_requires_a_live_content_location(block_key: BlockKey) -> None:
    """A logical owner cannot attach to content that never existed."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    workflow = WorkflowKey("workflow", 0)
    with pytest.raises(StateTransitionError, match="no live content location"):
        ledger.append(
            EventDraft(
                action=LedgerAction.BIND,
                status=LedgerStatus.COMPLETED,
                reason="dangling owner",
                timestamp_ns=1,
                operation_id="retention",
                workflow=workflow,
                request_id="retention-request",
                node_id="agent",
                block_key=block_key,
                binding_id="retention",
                binding_kind=BindingKind.WORKFLOW_RETENTION,
            )
        )
    assert ledger.events == ()


def test_lease_identity_cannot_reopen_and_audit_detects_tampering(
    block_key: BlockKey,
) -> None:
    """A lease ID names exactly one open-to-terminal lifecycle."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    workflow = WorkflowKey("workflow", 0)
    replica = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="allocate",
            timestamp_ns=0,
            operation_id="allocation",
            block_key=block_key,
            blocks=(replica,),
            payload_size=32,
            byte_count=64,
            payload_digest=block_key.content_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.MAP,
            status=LedgerStatus.COMPLETED,
            reason="publish",
            timestamp_ns=0,
            operation_id="content-map",
            parent_event_id=allocation.event_id,
            block_key=block_key,
            blocks=(replica,),
            mapping_id="content-map",
            payload_size=32,
            payload_digest=block_key.content_digest,
        )
    )
    binding = ledger.append(
        EventDraft(
            action=LedgerAction.BIND,
            status=LedgerStatus.COMPLETED,
            reason="retain",
            timestamp_ns=1,
            operation_id="retention",
            workflow=workflow,
            request_id="retention-request",
            node_id="agent",
            block_key=block_key,
            binding_id="retention",
            binding_kind=BindingKind.WORKFLOW_RETENTION,
        )
    )

    def open_lease(lease_id: str, timestamp_ns: int):
        return ledger.append(
            EventDraft(
                action=LedgerAction.LEASE,
                status=LedgerStatus.SCHEDULED,
                reason="future reuse",
                timestamp_ns=timestamp_ns,
                operation_id=lease_id,
                parent_event_id=binding.event_id,
                workflow=workflow,
                request_id="retention-request",
                node_id="agent",
                block_key=block_key,
                binding_id="retention",
                binding_kind=BindingKind.WORKFLOW_RETENTION,
                lease_id=lease_id,
                lease_deadline_ns=timestamp_ns + 1,
            )
        )

    def close_lease(
        lease_id: str,
        parent_event_id: str,
        timestamp_ns: int,
        deadline_ns: int,
    ):
        return ledger.append(
            EventDraft(
                action=LedgerAction.LEASE,
                status=LedgerStatus.COMPLETED,
                reason="expired",
                timestamp_ns=timestamp_ns,
                operation_id=lease_id,
                parent_event_id=parent_event_id,
                workflow=workflow,
                request_id="retention-request",
                node_id="agent",
                block_key=block_key,
                binding_id="retention",
                binding_kind=BindingKind.WORKFLOW_RETENTION,
                lease_id=lease_id,
                lease_deadline_ns=deadline_ns,
            )
        )

    first_open = open_lease("lease-one", 2)
    with pytest.raises(StateTransitionError, match="expired before its deadline"):
        close_lease("lease-one", first_open.event_id, 2, 3)
    close_lease("lease-one", first_open.event_id, 3, 3)
    with pytest.raises(StateTransitionError, match="lease identity already used"):
        open_lease("lease-one", 4)

    second_open = open_lease("lease-two", 4)
    second_close = close_lease("lease-two", second_open.event_id, 5, 5)
    assert ledger.audit() == ()

    ledger._events[second_open.sequence] = replace(
        second_open,
        operation_id="lease-one",
        lease_id="lease-one",
    )
    ledger._events[second_close.sequence] = replace(
        second_close,
        operation_id="lease-one",
        lease_id="lease-one",
    )
    assert any("lease identity already used" in issue for issue in ledger.audit())


def test_cross_family_references_block_early_free_and_failed_publish(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Bindings and transfer outcomes constrain physical content lifetimes."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    workflow = WorkflowKey("workflow", 0)
    payload_digest = digest("payload")
    gpu = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    cpu = ReplicaId(Tier.CPU, "numa:0", "slot", 1)
    gpu_allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="allocate source",
            timestamp_ns=1,
            operation_id="gpu-allocation",
            block_key=block_key,
            blocks=(gpu,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    gpu_map = ledger.append(
        EventDraft(
            action=LedgerAction.MAP,
            status=LedgerStatus.COMPLETED,
            reason="publish source",
            timestamp_ns=1,
            operation_id="gpu-map",
            parent_event_id=gpu_allocation.event_id,
            block_key=block_key,
            blocks=(gpu,),
            mapping_id="gpu-map",
            payload_size=32,
            payload_digest=payload_digest,
        )
    )
    binding = ledger.append(
        EventDraft(
            action=LedgerAction.BIND,
            status=LedgerStatus.COMPLETED,
            reason="retain source",
            timestamp_ns=2,
            operation_id="retention",
            workflow=workflow,
            request_id="retention-request",
            node_id="agent",
            block_key=block_key,
            binding_id="retention",
            binding_kind=BindingKind.WORKFLOW_RETENTION,
        )
    )
    early_unmap = EventDraft(
        action=LedgerAction.UNMAP,
        status=LedgerStatus.COMPLETED,
        reason="early free",
        timestamp_ns=3,
        operation_id="gpu-map",
        parent_event_id=gpu_map.event_id,
        block_key=block_key,
        blocks=(gpu,),
        mapping_id="gpu-map",
        payload_size=32,
        payload_digest=payload_digest,
    )
    with pytest.raises(StateTransitionError, match="live binding"):
        ledger.append(early_unmap)
    ledger.append(
        EventDraft(
            action=LedgerAction.RELEASE,
            status=LedgerStatus.COMPLETED,
            reason="release retention",
            timestamp_ns=3,
            operation_id="retention",
            parent_event_id=binding.event_id,
            workflow=workflow,
            request_id="retention-request",
            node_id="agent",
            block_key=block_key,
            binding_id="retention",
            binding_kind=BindingKind.WORKFLOW_RETENTION,
        )
    )

    conflicting_payload = digest("conflicting payload")
    target_allocation_draft = EventDraft(
        action=LedgerAction.ALLOCATE,
        status=LedgerStatus.COMPLETED,
        reason="reserve target",
        timestamp_ns=4,
        operation_id="cpu-allocation",
        block_key=block_key,
        blocks=(cpu,),
        payload_size=32,
        byte_count=64,
        payload_digest=conflicting_payload,
    )
    with pytest.raises(StateTransitionError, match="payload digest changed"):
        ledger.append(target_allocation_draft)
    cpu_allocation = ledger.append(
        replace(target_allocation_draft, payload_digest=payload_digest)
    )
    transfer = ledger.append(
        EventDraft(
            action=LedgerAction.SAVE,
            status=LedgerStatus.SCHEDULED,
            reason="save",
            timestamp_ns=5,
            operation_id="save",
            parent_event_id=cpu_allocation.event_id,
            block_key=block_key,
            blocks=(cpu,),
            transfer_id="save",
            source_tier=Tier.GPU,
            target_tier=Tier.CPU,
            byte_count=32,
            payload_digest=payload_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.SAVE,
            status=LedgerStatus.FAILED,
            reason="DMA failure",
            timestamp_ns=6,
            operation_id="save",
            parent_event_id=transfer.event_id,
            block_key=block_key,
            blocks=(cpu,),
            transfer_id="save",
            source_tier=Tier.GPU,
            target_tier=Tier.CPU,
            byte_count=32,
            payload_digest=payload_digest,
            error="DMA failure",
        )
    )
    with pytest.raises(StateTransitionError, match="cannot publish content"):
        ledger.append(
            EventDraft(
                action=LedgerAction.MAP,
                status=LedgerStatus.COMPLETED,
                reason="invalid publication",
                timestamp_ns=7,
                operation_id="cpu-map",
                parent_event_id=cpu_allocation.event_id,
                block_key=block_key,
                blocks=(cpu,),
                mapping_id="cpu-map",
                payload_size=32,
                payload_digest=payload_digest,
            )
        )
    ledger.append(
        EventDraft(
            action=LedgerAction.EVICT,
            status=LedgerStatus.COMPLETED,
            reason="failed target cleanup",
            timestamp_ns=7,
            operation_id="cpu-allocation",
            parent_event_id=cpu_allocation.event_id,
            block_key=block_key,
            blocks=(cpu,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    ledger.append(replace(early_unmap, timestamp_ns=8, reason="final unmap"))
    ledger.append(
        EventDraft(
            action=LedgerAction.EVICT,
            status=LedgerStatus.COMPLETED,
            reason="final eviction",
            timestamp_ns=8,
            operation_id="gpu-allocation",
            parent_event_id=gpu_allocation.event_id,
            block_key=block_key,
            blocks=(gpu,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    assert ledger.audit(require_quiescent=True) == ()


def test_cpu_only_gpu_reentry_requires_completed_h2d(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """A later GPU generation cannot publish around the transfer lifecycle."""

    ledger = EventLedger(run_id="run", phase="test", source="pytest")
    payload_digest = digest("payload")
    gpu_v1 = ReplicaId(Tier.GPU, "cuda:0", "slot", 1)
    gpu_v2 = ReplicaId(Tier.GPU, "cuda:0", "slot", 2)
    cpu = ReplicaId(Tier.CPU, "numa:0", "slot", 1)

    gpu_allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="producer allocation",
            timestamp_ns=0,
            operation_id="gpu-v1",
            block_key=block_key,
            blocks=(gpu_v1,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    gpu_map = ledger.append(
        EventDraft(
            action=LedgerAction.MAP,
            status=LedgerStatus.COMPLETED,
            reason="producer publication",
            timestamp_ns=0,
            operation_id="gpu-map-v1",
            parent_event_id=gpu_allocation.event_id,
            block_key=block_key,
            blocks=(gpu_v1,),
            mapping_id="gpu-map-v1",
            payload_size=32,
            payload_digest=payload_digest,
        )
    )
    cpu_allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="save target",
            timestamp_ns=1,
            operation_id="cpu-v1",
            block_key=block_key,
            blocks=(cpu,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    save = ledger.append(
        EventDraft(
            action=LedgerAction.SAVE,
            status=LedgerStatus.SCHEDULED,
            reason="save",
            timestamp_ns=2,
            operation_id="save",
            parent_event_id=cpu_allocation.event_id,
            block_key=block_key,
            blocks=(cpu,),
            transfer_id="save",
            source_tier=Tier.GPU,
            target_tier=Tier.CPU,
            byte_count=32,
            payload_digest=payload_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.SAVE,
            status=LedgerStatus.COMPLETED,
            reason="save completed",
            timestamp_ns=3,
            operation_id="save",
            parent_event_id=save.event_id,
            block_key=block_key,
            blocks=(cpu,),
            transfer_id="save",
            source_tier=Tier.GPU,
            target_tier=Tier.CPU,
            byte_count=32,
            observed_byte_count=32,
            payload_digest=payload_digest,
            observed_digest=payload_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.MAP,
            status=LedgerStatus.COMPLETED,
            reason="saved content published",
            timestamp_ns=3,
            operation_id="cpu-map",
            parent_event_id=cpu_allocation.event_id,
            block_key=block_key,
            blocks=(cpu,),
            mapping_id="cpu-map",
            payload_size=32,
            payload_digest=payload_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.UNMAP,
            status=LedgerStatus.COMPLETED,
            reason="drop old GPU map",
            timestamp_ns=4,
            operation_id="gpu-map-v1",
            parent_event_id=gpu_map.event_id,
            block_key=block_key,
            blocks=(gpu_v1,),
            mapping_id="gpu-map-v1",
            payload_size=32,
            payload_digest=payload_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.EVICT,
            status=LedgerStatus.COMPLETED,
            reason="drop old GPU allocation",
            timestamp_ns=4,
            operation_id="gpu-v1",
            parent_event_id=gpu_allocation.event_id,
            block_key=block_key,
            blocks=(gpu_v1,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )

    gpu_v2_allocation = ledger.append(
        EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason="load target",
            timestamp_ns=5,
            operation_id="gpu-v2",
            block_key=block_key,
            blocks=(gpu_v2,),
            payload_size=32,
            byte_count=64,
            payload_digest=payload_digest,
        )
    )
    gpu_v2_map = EventDraft(
        action=LedgerAction.MAP,
        status=LedgerStatus.COMPLETED,
        reason="GPU publication",
        timestamp_ns=6,
        operation_id="gpu-map-v2",
        parent_event_id=gpu_v2_allocation.event_id,
        block_key=block_key,
        blocks=(gpu_v2,),
        mapping_id="gpu-map-v2",
        payload_size=32,
        payload_digest=payload_digest,
    )
    with pytest.raises(StateTransitionError, match="completed H2D"):
        ledger.append(gpu_v2_map)

    load = ledger.append(
        EventDraft(
            action=LedgerAction.LOAD,
            status=LedgerStatus.SCHEDULED,
            reason="load",
            timestamp_ns=6,
            operation_id="load",
            parent_event_id=gpu_v2_allocation.event_id,
            block_key=block_key,
            blocks=(gpu_v2,),
            transfer_id="load",
            source_tier=Tier.CPU,
            target_tier=Tier.GPU,
            byte_count=32,
            payload_digest=payload_digest,
        )
    )
    ledger.append(
        EventDraft(
            action=LedgerAction.LOAD,
            status=LedgerStatus.COMPLETED,
            reason="load completed",
            timestamp_ns=7,
            operation_id="load",
            parent_event_id=load.event_id,
            block_key=block_key,
            blocks=(gpu_v2,),
            transfer_id="load",
            source_tier=Tier.CPU,
            target_tier=Tier.GPU,
            byte_count=32,
            observed_byte_count=32,
            payload_digest=payload_digest,
            observed_digest=payload_digest,
        )
    )
    ledger.append(replace(gpu_v2_map, timestamp_ns=7))
    assert ledger.audit() == ()
