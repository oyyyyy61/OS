"""Focused and adversarial tests for canonical C1 lifecycle evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from dagkv.c1_lifecycle import (
    LIFECYCLE_CLOCK_DOMAIN,
    LIFECYCLE_SIDECAR_SCHEMA_VERSION,
    CanonicalLifecycleEvidenceGate,
    ClosedLifecycleArtifact,
    _project_lifecycle,
    load_lifecycle_artifact,
    make_lifecycle_closure,
    write_lifecycle_artifact,
)
from dagkv.c1_trace import (
    AbstainedAttemptPayload,
    AtomicCutoffView,
    CutoffPayload,
    DemandIntentPayload,
    EvidenceRole,
    H2DExecMapService,
    H2DFailedService,
    ObservationTerminalPayload,
    RequestCancelledService,
    ServiceDisposition,
    TraceHeaderPayload,
    TraceRecordType,
    TraceValidationError,
    validate_trace,
)
from dagkv.domain import (
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    ExecutionRef,
    LedgerAction,
    LedgerStatus,
    ReplicaId,
    StateTransitionError,
    Tier,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)
from dagkv.orchestrator import LifecycleOrchestrator
from tests.test_c1_trace import _digest, _trace
from tests.test_c1_trace_runtime import (
    _drop_to_cpu,
    _runtime_with_retained_waiter,
)


def _runtime(block_key: BlockKey) -> tuple[LifecycleOrchestrator, WorkflowSpec]:
    runtime = LifecycleOrchestrator(run_id="run-1", phase="m3_c1b")
    workflow = WorkflowSpec(
        key=WorkflowKey("workflow-1", 0),
        nodes=(WorkflowNode("root"), WorkflowNode("next", ("root",))),
    )
    runtime.register_workflow(workflow)
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1),
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=_digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(workflow.key, "root", timestamp_ns=2)
    runtime.bind_owner(
        BindingHandle(workflow.key, "retention", "retention-1"),
        node_id="root",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=3,
    )
    return runtime, workflow


def _artifact(
    runtime: LifecycleOrchestrator,
    trace_pair_id: str,
) -> ClosedLifecycleArtifact:
    events = runtime.events
    return ClosedLifecycleArtifact(
        schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
        artifact_id="lifecycle-artifact-1",
        trace_pair_id=trace_pair_id,
        run_id="run-1",
        phase="m3_c1b",
        source="dagkv.orchestrator",
        clock_domain=LIFECYCLE_CLOCK_DOMAIN,
        implementation_digest=_digest("implementation"),
        environment_digest=_digest("environment"),
        events=events,
        closure=make_lifecycle_closure(events),
    )


def _complete_trace(
    runtime: LifecycleOrchestrator,
    workflow: WorkflowSpec,
    block_key: BlockKey,
):
    records = list(_trace(block_key, predicted=False, service="none"))
    snapshot = runtime.shared_lease_policy_snapshot(block_key)
    view = AtomicCutoffView(
        snapshot=snapshot,
        owner_specs=(workflow,),
        lifecycle_prefix=runtime.events,
        cutoff_ns=10,
        horizon_duration_ns=20,
        deadline_ns=30,
    )
    cutoff_record = next(
        record for record in records if record.record_type == TraceRecordType.CUTOFF
    )
    cutoff = cutoff_record.payload
    assert isinstance(cutoff, CutoffPayload)
    last = runtime.events[-1]
    records[cutoff_record.sequence] = replace(
        cutoff_record,
        payload=replace(
            cutoff,
            snapshot=snapshot,
            lifecycle_event_count=len(runtime.events),
            last_event_id=last.event_id,
            last_event_timestamp_ns=last.timestamp_ns,
            atomic_cutoff_view_digest=view.view_digest,
        ),
    )
    attempt_record = next(
        record
        for record in records
        if record.record_type == TraceRecordType.FORECAST_ATTEMPT
    )
    attempt = attempt_record.payload
    assert isinstance(attempt, AbstainedAttemptPayload)
    records[attempt_record.sequence] = replace(
        attempt_record,
        payload=replace(
            attempt,
            context=replace(
                attempt.context,
                information_cutoff_digest=view.view_digest,
            ),
        ),
    )
    runtime.seal_lifecycle()
    terminal_last = runtime.events[-1]
    terminal_record = next(
        record
        for record in records
        if record.record_type == TraceRecordType.OBSERVATION_TERMINAL
    )
    terminal = terminal_record.payload
    assert isinstance(terminal, ObservationTerminalPayload)
    records[terminal_record.sequence] = replace(
        terminal_record,
        payload=replace(
            terminal,
            last_verified_event_count=len(runtime.events),
            last_verified_event_id=terminal_last.event_id,
            last_verified_event_timestamp_ns=terminal_last.timestamp_ns,
        ),
    )
    return tuple(records)


def test_lifecycle_artifact_round_trip_and_complete_gate(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    runtime, workflow = _runtime(block_key)
    records = _complete_trace(runtime, workflow, block_key)
    header = records[0].payload
    assert isinstance(header, TraceHeaderPayload)
    artifact = _artifact(runtime, header.trace_pair_id)
    path = (tmp_path / "lifecycle.json").resolve()

    digest = write_lifecycle_artifact(path, artifact)
    loaded = load_lifecycle_artifact(path)
    assert loaded.artifact == artifact
    assert loaded.digest == digest

    receipt = CanonicalLifecycleEvidenceGate(
        path,
        digest,
        _digest("lifecycle-verifier"),
    ).verify_lifecycle(validate_trace(records))
    assert receipt.role == EvidenceRole.LIFECYCLE
    assert receipt.artifact_digest == digest
    assert receipt.verified_observation_ids == ("observation-1",)

    with pytest.raises(TraceValidationError, match="create-only"):
        write_lifecycle_artifact(path, artifact)


def test_lifecycle_gate_recomputes_cutoff_digest(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    runtime, workflow = _runtime(block_key)
    records = list(_complete_trace(runtime, workflow, block_key))
    forged_digest = _digest("forged-cutoff-view")
    cutoff_record = next(
        record for record in records if record.record_type == TraceRecordType.CUTOFF
    )
    cutoff = cutoff_record.payload
    assert isinstance(cutoff, CutoffPayload)
    records[cutoff_record.sequence] = replace(
        cutoff_record,
        payload=replace(cutoff, atomic_cutoff_view_digest=forged_digest),
    )
    attempt_record = next(
        record
        for record in records
        if record.record_type == TraceRecordType.FORECAST_ATTEMPT
    )
    attempt = attempt_record.payload
    assert isinstance(attempt, AbstainedAttemptPayload)
    records[attempt_record.sequence] = replace(
        attempt_record,
        payload=replace(
            attempt,
            context=replace(
                attempt.context,
                information_cutoff_digest=forged_digest,
            ),
        ),
    )
    validated = validate_trace(tuple(records))
    header = records[0].payload
    assert isinstance(header, TraceHeaderPayload)
    artifact = _artifact(runtime, header.trace_pair_id)
    path = (tmp_path / "forged-cutoff-lifecycle.json").resolve()
    digest = write_lifecycle_artifact(path, artifact)

    with pytest.raises(TraceValidationError, match="cutoff digest"):
        CanonicalLifecycleEvidenceGate(
            path,
            digest,
            _digest("lifecycle-verifier"),
        ).verify_lifecycle(validated)


def test_lifecycle_closure_rejects_a_partial_batch(block_key: BlockKey) -> None:
    runtime, _ = _runtime(block_key)
    partial = runtime.events[:2]
    with pytest.raises(TraceValidationError, match="truncates.*batch"):
        make_lifecycle_closure(partial)


def test_lifecycle_closure_requires_a_runtime_seal(block_key: BlockKey) -> None:
    runtime, _ = _runtime(block_key)
    with pytest.raises(TraceValidationError, match="final stream seal"):
        make_lifecycle_closure(runtime.events)

    seal = runtime.seal_lifecycle()
    closure = make_lifecycle_closure(runtime.events)
    assert closure.closed_through_ns == seal.timestamp_ns
    with pytest.raises(StateTransitionError, match="sealed"):
        runtime.register_workflow(
            WorkflowSpec(
                WorkflowKey("after-seal", 0),
                (WorkflowNode("node"),),
            )
        )


def test_lifecycle_artifact_rejects_zero_batch_size_without_hanging(
    block_key: BlockKey,
) -> None:
    runtime, _ = _runtime(block_key)
    runtime.seal_lifecycle()
    events = list(runtime.events)
    events[0] = replace(events[0], batch_size=0)
    tampered = tuple(events)
    with pytest.raises(TraceValidationError, match="batch"):
        ClosedLifecycleArtifact(
            schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            artifact_id="zero-batch",
            trace_pair_id="pair-1",
            run_id="run-1",
            phase="m3_c1b",
            source="dagkv.orchestrator",
            clock_domain=LIFECYCLE_CLOCK_DOMAIN,
            implementation_digest=_digest("implementation"),
            environment_digest=_digest("environment"),
            events=tampered,
            closure=make_lifecycle_closure(tampered),
        )


def test_lifecycle_artifact_rejects_batch_metadata_tampering(
    block_key: BlockKey,
) -> None:
    runtime, _ = _runtime(block_key)
    runtime.seal_lifecycle()
    events = list(runtime.events)
    events[1] = replace(events[1], batch_size=events[1].batch_size + 1)
    tampered = tuple(events)
    with pytest.raises(TraceValidationError, match="batch"):
        ClosedLifecycleArtifact(
            schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            artifact_id="tampered-batch",
            trace_pair_id="pair-1",
            run_id="run-1",
            phase="m3_c1b",
            source="dagkv.orchestrator",
            clock_domain=LIFECYCLE_CLOCK_DOMAIN,
            implementation_digest=_digest("implementation"),
            environment_digest=_digest("environment"),
            events=tampered,
            closure=make_lifecycle_closure(tampered),
        )


def test_lifecycle_artifact_rejects_block_state_tampering(
    block_key: BlockKey,
) -> None:
    runtime, _ = _runtime(block_key)
    runtime.seal_lifecycle()
    events = list(runtime.events)
    index = next(
        index
        for index, event in enumerate(events)
        if event.action == LedgerAction.BLOCK_STATE
    )
    state = events[index].block_state_after
    assert state is not None
    events[index] = replace(
        events[index],
        block_state_after=replace(
            state,
            location_version=state.location_version + 1,
        ),
    )
    tampered = tuple(events)
    with pytest.raises(TraceValidationError, match="location version"):
        ClosedLifecycleArtifact(
            schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            artifact_id="tampered-state",
            trace_pair_id="pair-1",
            run_id="run-1",
            phase="m3_c1b",
            source="dagkv.orchestrator",
            clock_domain=LIFECYCLE_CLOCK_DOMAIN,
            implementation_digest=_digest("implementation"),
            environment_digest=_digest("environment"),
            events=tampered,
            closure=make_lifecycle_closure(tampered),
        )


def test_lifecycle_artifact_rejects_noop_block_state_watermark(
    block_key: BlockKey,
) -> None:
    runtime, _ = _runtime(block_key)
    runtime.seal_lifecycle()
    events = list(runtime.events)
    seal = events.pop()
    prior = next(
        event for event in reversed(events) if event.action == LedgerAction.BLOCK_STATE
    )
    sequence = len(events)
    events.append(
        replace(
            prior,
            sequence=sequence,
            event_id=f"evt-{sequence:012d}",
            batch_id=f"bat-{sequence:012d}",
            batch_index=0,
            batch_size=1,
            timestamp_ns=seal.timestamp_ns,
        )
    )
    seal_sequence = sequence + 1
    events.append(
        replace(
            seal,
            sequence=seal_sequence,
            event_id=f"evt-{seal_sequence:012d}",
            batch_id=f"bat-{seal_sequence:012d}",
        )
    )
    tampered = tuple(events)
    with pytest.raises(TraceValidationError, match="no causal state change"):
        ClosedLifecycleArtifact(
            schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            artifact_id="noop-state",
            trace_pair_id="pair-1",
            run_id="run-1",
            phase="m3_c1b",
            source="dagkv.orchestrator",
            clock_domain=LIFECYCLE_CLOCK_DOMAIN,
            implementation_digest=_digest("implementation"),
            environment_digest=_digest("environment"),
            events=tampered,
            closure=make_lifecycle_closure(tampered),
        )


def test_coalesced_h2d_waiter_join_leave_and_terminal_are_replayable(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, _, first_waiter, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    first = runtime.ensure_h2d(
        block_key,
        target,
        (first_waiter,),
        transfer_id="actual-single-flight",
        timestamp_ns=8,
    )
    assert first is not None

    second_waiter = BindingHandle(
        workflow.key,
        "second-request",
        "second-request-binding",
    )
    runtime.bind_owner(
        second_waiter,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(
            workflow.key,
            "second-request",
            "second-sequence",
            0,
        ),
        timestamp_ns=9,
    )
    before_join = len(runtime.events)
    coalesced = runtime.ensure_h2d(
        block_key,
        target,
        (second_waiter,),
        transfer_id="ignored-proposal-id",
        timestamp_ns=10,
    )
    assert coalesced == first
    join_batch = runtime.events[before_join:]
    assert [event.action for event in join_batch] == [
        LedgerAction.BIND_STATE,
        LedgerAction.WAITER_JOIN,
    ]
    join = join_batch[-1]
    assert join.transfer_id == first.transfer_id
    assert join.waiter_binding_ids_after == tuple(
        sorted((first_waiter.binding_id, second_waiter.binding_id))
    )

    before_release = len(runtime.events)
    runtime.release_binding(second_waiter, timestamp_ns=11)
    release_batch = runtime.events[before_release:]
    assert [event.action for event in release_batch] == [
        LedgerAction.WAITER_LEAVE,
        LedgerAction.RELEASE,
    ]
    assert release_batch[0].waiter_binding_ids_after == (first_waiter.binding_id,)
    assert release_batch[1].binding_state_before == BindingState.WAITING
    assert release_batch[1].binding_state_after == BindingState.RELEASED

    runtime.complete_transfer(
        first.transfer_id,
        timestamp_ns=12,
        observed_bytes=first.byte_count,
        observed_digest=first.payload_digest,
    )
    terminal = next(
        event
        for event in runtime.events
        if event.transfer_id == first.transfer_id and event.status.value == "completed"
    )
    assert terminal.waiter_binding_ids_after == (first_waiter.binding_id,)
    assert runtime.audit().passed

    for event in runtime.events:
        batch_start = event.sequence - event.batch_index
        assert event.batch_id == f"bat-{batch_start:012d}"
    runtime.seal_lifecycle()
    ClosedLifecycleArtifact(
        schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
        artifact_id="h2d-lifecycle",
        trace_pair_id="h2d-pair",
        run_id="c1-runtime",
        phase="m3_c1b",
        source="dagkv.orchestrator",
        clock_domain=LIFECYCLE_CLOCK_DOMAIN,
        implementation_digest=_digest("implementation"),
        environment_digest=_digest("environment"),
        events=runtime.events,
        closure=make_lifecycle_closure(runtime.events),
    )


def test_initial_h2d_waiter_events_are_identity_sorted(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, _, first_waiter, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    earlier_waiter = BindingHandle(
        workflow.key,
        "earlier-request",
        "alpha-request-binding",
    )
    runtime.bind_owner(
        earlier_waiter,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(
            workflow.key,
            "earlier-request",
            "earlier-sequence",
            0,
        ),
        timestamp_ns=5,
    )
    _drop_to_cpu(runtime, block_key, gpu)

    event_count = len(runtime.events)
    command = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2),
        (first_waiter, earlier_waiter),
        transfer_id="identity-sorted-load",
        timestamp_ns=8,
    )
    assert command is not None
    join_events = [
        event
        for event in runtime.events[event_count:]
        if event.action == LedgerAction.WAITER_JOIN
    ]
    assert [event.binding_id for event in join_events] == [
        earlier_waiter.binding_id,
        first_waiter.binding_id,
    ]
    assert join_events[-1].waiter_binding_ids_after == tuple(
        sorted((earlier_waiter.binding_id, first_waiter.binding_id))
    )
    report = runtime.audit()
    assert report.passed, report.issues


def test_runtime_audit_reconciles_complete_waiter_history(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    command = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2),
        (request,),
        transfer_id="waiter-history-load",
        timestamp_ns=8,
    )
    assert command is not None
    runtime.fail_transfer(
        command.transfer_id,
        timestamp_ns=9,
        observed_bytes=0,
        observed_digest=None,
        error="injected DMA failure",
    )
    assert runtime.audit().passed

    runtime._transfer_waiter_history.clear()
    report = runtime.audit()
    assert not report.passed
    assert "transfer waiter history ledger/runtime mismatch" in report.issues


def test_lifecycle_artifact_rejects_forged_waiter_lineage(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    command = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2),
        (request,),
        transfer_id="lineage-load",
        timestamp_ns=8,
    )
    assert command is not None
    runtime.complete_transfer(
        command.transfer_id,
        timestamp_ns=9,
        observed_bytes=command.byte_count,
        observed_digest=command.payload_digest,
    )
    runtime.seal_lifecycle()
    events = list(runtime.events)
    index = next(
        index
        for index, event in enumerate(events)
        if event.action == LedgerAction.WAITER_JOIN
        and event.transfer_id == command.transfer_id
    )
    forged_workflow = WorkflowKey("forged-workflow", 0)
    events[index] = replace(
        events[index],
        workflow=forged_workflow,
        request_id="forged-request",
        node_id="forged-node",
        execution_ref=ExecutionRef(
            forged_workflow,
            "forged-request",
            "forged-sequence",
            0,
        ),
    )
    tampered = tuple(events)
    with pytest.raises(TraceValidationError, match="waiter binding lineage"):
        ClosedLifecycleArtifact(
            schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            artifact_id="forged-waiter-lineage",
            trace_pair_id="pair-1",
            run_id="c1-runtime",
            phase="m3_c1b",
            source="dagkv.orchestrator",
            clock_domain=LIFECYCLE_CLOCK_DOMAIN,
            implementation_digest=_digest("implementation"),
            environment_digest=_digest("environment"),
            events=tampered,
            closure=make_lifecycle_closure(tampered),
        )


def test_lifecycle_gate_accepts_a_post_intent_coalesced_waiter(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, retention, first_request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    command = runtime.ensure_h2d(
        block_key,
        target,
        (first_request,),
        transfer_id="shared-load",
        timestamp_ns=8,
    )
    assert command is not None
    second_request = BindingHandle(
        workflow.key,
        "second-request",
        "second-request-binding",
    )
    execution = ExecutionRef(
        workflow.key,
        second_request.request_id,
        "second-sequence",
        0,
    )
    runtime.bind_owner(
        second_request,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=execution,
        timestamp_ns=9,
    )
    pre_service_count = len(runtime.events)
    pre_service_events = runtime.events
    assert (
        runtime.ensure_h2d(
            block_key,
            target,
            (second_request,),
            transfer_id="ignored-coalesced-proposal",
            timestamp_ns=10,
        )
        == command
    )
    runtime.complete_transfer(
        command.transfer_id,
        timestamp_ns=11,
        observed_bytes=command.byte_count,
        observed_digest=command.payload_digest,
    )
    scheduled = next(
        event
        for event in runtime.events
        if event.transfer_id == command.transfer_id
        and event.status == LedgerStatus.SCHEDULED
    )
    join = next(
        event
        for event in runtime.events
        if event.action == LedgerAction.WAITER_JOIN
        and event.binding_id == second_request.binding_id
    )
    terminal = next(
        event
        for event in runtime.events
        if event.transfer_id == command.transfer_id
        and event.action in {LedgerAction.LOAD, LedgerAction.PREFETCH}
        and event.status == LedgerStatus.COMPLETED
    )
    exec_map = next(
        event
        for event in runtime.events
        if event.action == LedgerAction.EXEC_MAP
        and event.binding_id == second_request.binding_id
    )
    assert scheduled.sequence < pre_service_count <= join.sequence
    intent = DemandIntentPayload(
        schedule_event_id="second-schedule-event",
        scheduled_access_ns=10,
        claim_id="second-claim",
        retention_binding_id=retention.binding_id,
        request_binding_id=second_request.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=execution,
        block_key=block_key,
        reuse_epoch_id="second-epoch",
        pre_service_event_count=pre_service_count,
        pre_service_last_event_id=pre_service_events[-1].event_id,
        pre_service_last_timestamp_ns=pre_service_events[-1].timestamp_ns,
    )
    projection = _project_lifecycle(pre_service_events)
    CanonicalLifecycleEvidenceGate._verify_intent_bindings(
        intent,
        projection,
        cutoff_owner_ids={retention.binding_id},
    )
    service = H2DExecMapService(
        intent_record_id="second-intent",
        disposition=ServiceDisposition.H2D_EXEC_MAP,
        transfer_id=command.transfer_id,
        transfer_scheduled_event_id=scheduled.event_id,
        waiter_join_event_id=join.event_id,
        transfer_terminal_event_id=terminal.event_id,
        exec_map_event_id=exec_map.event_id,
        waiter_binding_ids=terminal.waiter_binding_ids_after or (),
    )
    CanonicalLifecycleEvidenceGate._verify_service(
        service,
        intent,
        {event.event_id: event for event in runtime.events},
        pre_service_projection=projection,
        label_available_ns=12,
    )


def test_lifecycle_gate_rejects_request_created_after_pre_service_prefix(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, retention, request, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    binding = runtime.binding_snapshot(request)
    assert binding.execution_ref is not None
    prefix = runtime.events[:-1]
    intent = DemandIntentPayload(
        schedule_event_id="schedule-event",
        scheduled_access_ns=5,
        claim_id="claim",
        retention_binding_id=retention.binding_id,
        request_binding_id=request.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=binding.execution_ref,
        block_key=block_key,
        reuse_epoch_id="epoch",
        pre_service_event_count=len(prefix),
        pre_service_last_event_id=prefix[-1].event_id,
        pre_service_last_timestamp_ns=prefix[-1].timestamp_ns,
    )
    with pytest.raises(TraceValidationError, match="request binding"):
        CanonicalLifecycleEvidenceGate._verify_intent_bindings(
            intent,
            _project_lifecycle(prefix),
            cutoff_owner_ids={retention.binding_id},
        )


def test_lifecycle_gate_rejects_request_with_prior_service_history(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, retention, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    assert (
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="first-resident-service",
            timestamp_ns=5,
        )
        is None
    )
    runtime.set_binding_state(request, BindingState.RETAINED, timestamp_ns=6)
    binding = runtime.binding_snapshot(request)
    assert binding.execution_ref is not None
    prefix = runtime.events
    intent = DemandIntentPayload(
        schedule_event_id="forged-second-schedule",
        scheduled_access_ns=7,
        claim_id="forged-second-claim",
        retention_binding_id=retention.binding_id,
        request_binding_id=request.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=binding.execution_ref,
        block_key=block_key,
        reuse_epoch_id="forged-second-epoch",
        pre_service_event_count=len(prefix),
        pre_service_last_event_id=prefix[-1].event_id,
        pre_service_last_timestamp_ns=prefix[-1].timestamp_ns,
    )

    with pytest.raises(TraceValidationError, match="prior service history"):
        CanonicalLifecycleEvidenceGate._verify_intent_bindings(
            intent,
            _project_lifecycle(prefix),
            cutoff_owner_ids={retention.binding_id},
        )


def test_lifecycle_gate_matches_failed_h2d_disposition_exactly(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, retention, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    pre_service_count = len(runtime.events)
    command = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2),
        (request,),
        transfer_id="failed-load",
        timestamp_ns=8,
    )
    assert command is not None
    runtime.fail_transfer(
        command.transfer_id,
        timestamp_ns=9,
        observed_bytes=0,
        observed_digest=None,
        error="injected DMA failure",
    )
    scheduled = next(
        event
        for event in runtime.events
        if event.transfer_id == command.transfer_id
        and event.status == LedgerStatus.SCHEDULED
    )
    terminal = next(
        event
        for event in runtime.events
        if event.transfer_id == command.transfer_id
        and event.status == LedgerStatus.FAILED
    )
    join = next(
        event
        for event in runtime.events
        if event.transfer_id == command.transfer_id
        and event.binding_id == request.binding_id
        and event.action == LedgerAction.WAITER_JOIN
    )
    binding = runtime.binding_snapshot(request)
    assert binding.execution_ref is not None
    intent = DemandIntentPayload(
        schedule_event_id="schedule-event",
        scheduled_access_ns=8,
        claim_id="claim",
        retention_binding_id=retention.binding_id,
        request_binding_id=request.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=binding.execution_ref,
        block_key=block_key,
        reuse_epoch_id="epoch",
        pre_service_event_count=pre_service_count,
        pre_service_last_event_id=runtime.events[pre_service_count - 1].event_id,
        pre_service_last_timestamp_ns=runtime.events[
            pre_service_count - 1
        ].timestamp_ns,
    )
    event_by_id = {event.event_id: event for event in runtime.events}
    correct = H2DFailedService(
        intent_record_id="intent",
        disposition=ServiceDisposition.H2D_FAILED,
        transfer_id=command.transfer_id,
        transfer_scheduled_event_id=scheduled.event_id,
        waiter_join_event_id=join.event_id,
        transfer_terminal_event_id=terminal.event_id,
        waiter_binding_ids=(request.binding_id,),
    )
    CanonicalLifecycleEvidenceGate._verify_service(
        correct,
        intent,
        event_by_id,
        pre_service_projection=_project_lifecycle(runtime.events[:pre_service_count]),
        label_available_ns=10,
    )

    wrong = replace(correct, disposition=ServiceDisposition.H2D_CANCELLED)
    with pytest.raises(TraceValidationError, match="terminal provenance"):
        CanonicalLifecycleEvidenceGate._verify_service(
            wrong,
            intent,
            event_by_id,
            pre_service_projection=_project_lifecycle(
                runtime.events[:pre_service_count]
            ),
            label_available_ns=10,
        )

    with pytest.raises(TraceValidationError, match="terminal provenance"):
        CanonicalLifecycleEvidenceGate._verify_service(
            correct,
            intent,
            event_by_id,
            pre_service_projection=_project_lifecycle(
                runtime.events[:pre_service_count]
            ),
            label_available_ns=8,
        )


def test_lifecycle_gate_rejects_release_before_demand_intent(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, retention, request, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    binding = runtime.binding_snapshot(request)
    assert binding.execution_ref is not None
    pre_service_projection = _project_lifecycle(runtime.events)
    runtime.release_binding(request, timestamp_ns=5)
    release = runtime.events[-1]
    assert release.action == LedgerAction.RELEASE
    intent = DemandIntentPayload(
        schedule_event_id="schedule-event",
        scheduled_access_ns=6,
        claim_id="claim",
        retention_binding_id=retention.binding_id,
        request_binding_id=request.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=binding.execution_ref,
        block_key=block_key,
        reuse_epoch_id="epoch",
        pre_service_event_count=release.sequence + 1,
        pre_service_last_event_id=release.event_id,
        pre_service_last_timestamp_ns=release.timestamp_ns,
    )
    service = RequestCancelledService(
        intent_record_id="intent",
        disposition=ServiceDisposition.REQUEST_CANCELLED,
        release_event_id=release.event_id,
    )
    with pytest.raises(TraceValidationError, match="cancellation event differs"):
        CanonicalLifecycleEvidenceGate._verify_service(
            service,
            intent,
            {release.event_id: release},
            pre_service_projection=pre_service_projection,
            label_available_ns=6,
        )
