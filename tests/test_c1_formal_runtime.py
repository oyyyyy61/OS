"""Formal sole-writer runtime integration for canonical C1 trace commits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv.c1_trace as trace_module
from dagkv.c1_commit import (
    AttemptPayload,
    CanonicalTraceCommitter,
    DemandCommitRequest,
    ObservationCloseRequest,
    ObservationTerminalSpec,
    TraceEnvelope,
    TracePreambleRequest,
)
from dagkv.c1_leases import (
    DependenceGroup,
    ForecastSource,
    JointOutcome,
    ReuseClaim,
    SharedLeaseForecast,
)
from dagkv.c1_schedule import ReplayScheduleClosure, ScheduleDemandEvent
from dagkv.c1_trace import (
    AtomicCutoffView,
    ForecastAttemptContext,
    ForecastAttemptStatus,
    PredictedAttemptPayload,
    ReplayScheduleWatermarkPayload,
    ResidentExecMapService,
    ServiceDisposition,
    TerminalReason,
    TerminalStatus,
    TraceHeaderPayload,
    TraceValidationError,
    WorkflowTopologyPayload,
    canonical_digest,
    canonical_json,
    load_trace_jsonl,
    validate_trace,
)
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
from tests.test_c1_schedule import _artifact, _schedule_epoch
from tests.test_c1_trace import _digest


@dataclass(frozen=True, slots=True)
class _FormalScenario:
    runtime: LifecycleOrchestrator
    committer: CanonicalTraceCommitter
    schedule_digest: str
    schedule_event: ScheduleDemandEvent
    request_handle: BindingHandle
    target_replica: ReplicaId
    demand_request: DemandCommitRequest
    attempt_factory: Callable[[AtomicCutoffView], AttemptPayload]


def _formal_scenario(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    *,
    trace_basename: str = "formal-trace.jsonl",
) -> _FormalScenario:
    run_id = "formal-run"
    workflow = WorkflowSpec(
        key=WorkflowKey("formal-workflow", 0),
        nodes=(WorkflowNode("agent"),),
    )
    retention = BindingHandle(workflow.key, "retention", "retention-binding")
    request_handle = BindingHandle(workflow.key, "request", "request-binding")
    execution_ref = ExecutionRef(workflow.key, "request", "sequence", 0)
    schedule_event = ScheduleDemandEvent(
        event_ordinal=0,
        schedule_event_id="schedule-event-1",
        scheduled_access_ns=8,
        claim_id="claim-1",
        retention_binding_id=retention.binding_id,
        request_binding_id=request_handle.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=execution_ref,
        block_key=block_key,
        reuse_epoch_id="epoch-1",
        source_record_id="plan-record-1",
        source_record_digest=_digest("plan-record-1"),
    )
    schedule = _artifact(
        (schedule_event,),
        (_schedule_epoch(schedule_event),),
        run_id=run_id,
        schedule_id="formal-schedule",
        schedule_case_id="formal-case",
        trace_pair_id="formal-pair",
    )
    schedule_digest = sha256(canonical_json(schedule)).hexdigest()
    grammar_digest = _digest("formal-grammar")
    topology = WorkflowTopologyPayload(
        workflow_spec=workflow,
        workflow_template_digest=_digest("formal-template"),
        source_case_digest=schedule.source_case_digest,
        split_component_id="formal-component",
        branch_grammar_digest=grammar_digest,
    )
    committer = CanonicalTraceCommitter(
        tmp_path / trace_basename,
        envelope=TraceEnvelope(
            trace_id="formal-trace",
            run_id=run_id,
            schedule_id=schedule.schedule_id,
            schedule_case_id=schedule.schedule_case_id,
        ),
        schedule=schedule,
        schedule_artifact_digest=schedule_digest,
    )
    committer.commit_preamble(
        TracePreambleRequest(
            operation_id="formal-preamble",
            header=TraceHeaderPayload(
                trace_pair_id=schedule.trace_pair_id,
                source_digest=schedule.source_artifact_digest,
                schedule_digest=schedule_digest,
                split_manifest_digest=_digest("formal-split"),
                branch_grammar_digest=grammar_digest,
                feature_contract_digest=_digest("formal-feature-contract"),
                implementation_digest=_digest("formal-implementation"),
                environment_digest=_digest("formal-environment"),
            ),
            topologies=(topology,),
        )
    )
    runtime = LifecycleOrchestrator(
        run_id=run_id,
        phase="m3_c1b",
        trace_required=True,
        formal_trace_committer=committer,
    )
    runtime.register_workflow(workflow)
    target_replica = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1)
    runtime.register_gpu_block(
        block_key,
        target_replica,
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=digest("formal-payload"),
        timestamp_ns=1,
    )
    runtime.start_node(workflow.key, "agent", timestamp_ns=2)
    runtime.bind_owner(
        retention,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=3,
    )
    runtime.bind_owner(
        request_handle,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=execution_ref,
        timestamp_ns=4,
    )

    def attempt_factory(view: AtomicCutoffView) -> AttemptPayload:
        context = ForecastAttemptContext(
            feature_view_digest=_digest("formal-feature-view"),
            information_cutoff_digest=view.view_digest,
            model_artifact_digest=_digest("formal-model"),
            predictor_digest=_digest("formal-predictor"),
            dependence_digest=_digest("formal-dependence"),
            outcome_catalog_digest=_digest("formal-outcome-catalog"),
            grouping_rules_digest=_digest("formal-grouping"),
            model_inputs_digest=_digest("formal-model-inputs"),
        )
        claim = ReuseClaim(
            claim_id=schedule_event.claim_id,
            binding_id=schedule_event.retention_binding_id,
            workflow=schedule_event.workflow,
            node_id=schedule_event.node_id,
            reuse_epoch_id=schedule_event.reuse_epoch_id,
            access_ns=schedule_event.scheduled_access_ns,
        )
        return PredictedAttemptPayload(
            status=ForecastAttemptStatus.PREDICTED,
            context=context,
            forecast=SharedLeaseForecast(
                forecast_id="formal-forecast",
                block_key=block_key,
                runtime_event_count=len(view.lifecycle_prefix),
                generated_ns=view.cutoff_ns,
                horizon_ns=view.deadline_ns,
                source=ForecastSource.PREDICTED,
                predictor_digest=context.predictor_digest,
                dependence_digest=context.dependence_digest,
                independence_basis="one frozen grammar group",
                groups=(
                    DependenceGroup(
                        group_id="formal-group",
                        outcomes=(
                            JointOutcome("demand", 500_000, (claim,)),
                            JointOutcome("no-demand", 500_000),
                        ),
                    ),
                ),
            ),
        )

    return _FormalScenario(
        runtime=runtime,
        committer=committer,
        schedule_digest=schedule_digest,
        schedule_event=schedule_event,
        request_handle=request_handle,
        target_replica=target_replica,
        demand_request=DemandCommitRequest(
            operation_id="formal-demand",
            observation_id="formal-observation",
            schedule_event_ids=(schedule_event.schedule_event_id,),
        ),
        attempt_factory=attempt_factory,
    )


def _commit_cutoff(scenario: _FormalScenario) -> None:
    committed = scenario.runtime.commit_shared_lease_cutoff_traced(
        scenario.schedule_event.block_key,
        cutoff_ns=5,
        horizon_duration_ns=10,
        operation_id="formal-cutoff",
        observation_id="formal-observation",
        attempt_factory=scenario.attempt_factory,
    )
    assert committed.receipt.commit.record_ids[-1] == (
        scenario.committer.records[-1].record_id
    )


def test_formal_runtime_returns_exact_receipts_and_event_free_replay(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    first_cutoff = scenario.runtime.commit_shared_lease_cutoff_traced(
        block_key,
        cutoff_ns=5,
        horizon_duration_ns=10,
        operation_id="formal-cutoff",
        observation_id="formal-observation",
        attempt_factory=scenario.attempt_factory,
    )
    replayed_cutoff = scenario.runtime.commit_shared_lease_cutoff_traced(
        block_key,
        cutoff_ns=5,
        horizon_duration_ns=10,
        operation_id="formal-cutoff",
        observation_id="formal-observation",
        attempt_factory=scenario.attempt_factory,
    )
    assert replayed_cutoff.receipt is first_cutoff.receipt

    first = scenario.runtime.ensure_h2d_traced(
        block_key,
        scenario.target_replica,
        (scenario.request_handle,),
        transfer_id="resident-demand",
        timestamp_ns=8,
        request=scenario.demand_request,
    )
    committed_events = scenario.runtime.events
    replay = scenario.runtime.ensure_h2d_traced(
        block_key,
        scenario.target_replica,
        (scenario.request_handle,),
        transfer_id="ignored-resident-replay",
        timestamp_ns=8,
        request=scenario.demand_request,
    )

    assert first.command is None
    assert replay.command is None
    assert not first.replayed and replay.replayed
    assert replay.receipt is first.receipt
    assert scenario.runtime.events == committed_events

    exec_map_event = next(
        event
        for event in reversed(committed_events)
        if event.action == LedgerAction.EXEC_MAP
        and event.binding_id == scenario.request_handle.binding_id
    )
    seal_event = scenario.runtime.seal_lifecycle()
    checkpoint = scenario.committer.schedule.checkpoints[-1]
    closure = scenario.committer.schedule.closure
    assert isinstance(closure, ReplayScheduleClosure)
    scenario.committer.close_observation(
        ObservationCloseRequest(
            operation_id="formal-close",
            observation_id="formal-observation",
            services=(
                ResidentExecMapService(
                    intent_record_id=first.receipt.commit.record_ids[0],
                    disposition=ServiceDisposition.RESIDENT_EXEC_MAP,
                    exec_map_event_id=exec_map_event.event_id,
                ),
            ),
            watermark=ReplayScheduleWatermarkPayload(
                producer_kind=scenario.committer.schedule.producer_kind,
                producer_id=scenario.committer.schedule.producer_id,
                producer_artifact_digest=closure.plan_event_digest,
                schedule_digest=scenario.schedule_digest,
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_digest=canonical_digest(checkpoint),
                consumed_event_count=checkpoint.consumed_event_count,
                last_schedule_event_id=checkpoint.last_schedule_event_id,
                max_closed_timestamp_ns=checkpoint.closed_through_ns,
                event_prefix_digest=checkpoint.event_prefix_digest,
                closed_epoch_count=checkpoint.closed_epoch_count,
                epoch_prefix_digest=checkpoint.epoch_prefix_digest,
            ),
            terminal=ObservationTerminalSpec(
                status=TerminalStatus.COMPLETE,
                reason=TerminalReason.WINDOW_COMPLETE,
                label_available_ns=max(
                    checkpoint.closed_through_ns,
                    seal_event.timestamp_ns,
                ),
                last_verified_event_count=len(scenario.runtime.events),
                last_verified_event_id=seal_event.event_id,
                last_verified_event_timestamp_ns=seal_event.timestamp_ns,
            ),
        )
    )
    sealed = scenario.committer.seal_trace()
    records = load_trace_jsonl(scenario.committer.path)

    assert sealed.closure.record_count == len(records)
    assert validate_trace(records).observations[0].observation_id == (
        "formal-observation"
    )


def test_formal_runtime_rejects_legacy_trace_endpoints(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)

    with pytest.raises(StateTransitionError, match="cutoff_traced"):
        scenario.runtime.commit_shared_lease_cutoff(
            block_key,
            cutoff_ns=5,
            horizon_duration_ns=10,
            committer=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(StateTransitionError, match="ensure_h2d_traced"):
        scenario.runtime.ensure_h2d(
            block_key,
            scenario.target_replica,
            (scenario.request_handle,),
            transfer_id="untyped-demand",
            timestamp_ns=8,
        )
    scenario.committer.abort()


def test_formal_runtime_rejects_schedule_drift_before_service(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    _commit_cutoff(scenario)
    before_events = scenario.runtime.events
    changed = replace(
        scenario.demand_request,
        schedule_event_ids=("unknown-schedule-event",),
    )

    with pytest.raises(TraceValidationError, match="unknown schedule"):
        scenario.runtime.ensure_h2d_traced(
            block_key,
            scenario.target_replica,
            (scenario.request_handle,),
            transfer_id="rejected-demand",
            timestamp_ns=8,
            request=changed,
        )

    assert scenario.runtime.events == before_events
    accepted = scenario.runtime.ensure_h2d_traced(
        block_key,
        scenario.target_replica,
        (scenario.request_handle,),
        transfer_id="accepted-demand",
        timestamp_ns=8,
        request=scenario.demand_request,
    )
    assert not accepted.replayed
    scenario.committer.abort()


def test_formal_trace_write_failure_poisons_runtime_before_service(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    _commit_cutoff(scenario)
    before_events = scenario.runtime.events

    def fail_write(*_args: object, **_kwargs: object) -> int:
        raise OSError("injected formal trace write failure")

    with monkeypatch.context() as context:
        context.setattr(trace_module.os, "write", fail_write)
        with pytest.raises(
            trace_module.TraceCommitIndeterminateError,
            match="indeterminate",
        ):
            scenario.runtime.ensure_h2d_traced(
                block_key,
                scenario.target_replica,
                (scenario.request_handle,),
                transfer_id="failed-demand",
                timestamp_ns=8,
                request=scenario.demand_request,
            )

    assert scenario.runtime.events == before_events
    with pytest.raises(StateTransitionError, match="runtime is poisoned"):
        scenario.runtime.seal_lifecycle()


def test_post_commit_receipt_failure_poisons_runtime(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    _commit_cutoff(scenario)
    before_count = len(scenario.runtime.events)

    def reject_receipt(*_args: object, **_kwargs: object) -> None:
        raise TraceValidationError("injected exact receipt rejection")

    with monkeypatch.context() as context:
        context.setattr(CanonicalTraceCommitter, "verify_receipt", reject_receipt)
        with pytest.raises(TraceValidationError, match="exact receipt"):
            scenario.runtime.ensure_h2d_traced(
                block_key,
                scenario.target_replica,
                (scenario.request_handle,),
                transfer_id="durable-demand",
                timestamp_ns=8,
                request=scenario.demand_request,
            )

    assert len(scenario.runtime.events) > before_count
    with pytest.raises(StateTransitionError, match="runtime is poisoned"):
        scenario.runtime.seal_lifecycle()


class _InjectedAbort(BaseException):
    pass


class _UnprintableAbort(BaseException):
    def __str__(self) -> str:
        raise RuntimeError("injected exception formatting failure")


def test_lifecycle_seal_response_loss_poisons_formal_attempt(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    original = scenario.runtime._ledger.append

    def committed_then_lost(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)  # type: ignore[arg-type]
        raise _UnprintableAbort()

    monkeypatch.setattr(scenario.runtime._ledger, "append", committed_then_lost)
    with pytest.raises(_UnprintableAbort):
        scenario.runtime.seal_lifecycle()

    assert scenario.runtime.events[-1].action == LedgerAction.STREAM_SEAL
    assert scenario.committer.poisoned_reason is not None
    with pytest.raises(StateTransitionError, match="runtime is poisoned"):
        scenario.runtime.seal_lifecycle()


def test_public_lifecycle_events_and_seal_are_detached(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    original_reason = scenario.runtime.events[0].reason
    exposed = scenario.runtime.events
    object.__setattr__(exposed[0], "reason", "forged-public-event")

    assert scenario.runtime.events[0].reason == original_reason
    seal = scenario.runtime.seal_lifecycle()
    object.__setattr__(seal, "reason", "forged-public-seal")
    assert scenario.runtime.events[-1].reason == "lifecycle_stream_closed"
    replayed = scenario.runtime.seal_lifecycle()
    assert replayed.reason == "lifecycle_stream_closed"
    scenario.committer.abort()


def test_demand_commit_response_loss_poisons_runtime_before_service(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    _commit_cutoff(scenario)
    before_events = scenario.runtime.events
    original = scenario.committer.commit_demands

    def durable_then_lost(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)  # type: ignore[arg-type]
        raise _InjectedAbort("typed demand receipt lost")

    monkeypatch.setattr(scenario.committer, "commit_demands", durable_then_lost)
    with pytest.raises(_InjectedAbort, match="receipt lost"):
        scenario.runtime.ensure_h2d_traced(
            block_key,
            scenario.target_replica,
            (scenario.request_handle,),
            transfer_id="lost-demand",
            timestamp_ns=8,
            request=scenario.demand_request,
        )

    assert len(scenario.committer.commits) == 3
    assert scenario.runtime.events == before_events
    assert scenario.committer.poisoned_reason is not None
    with pytest.raises(StateTransitionError, match="runtime is poisoned"):
        scenario.runtime.seal_lifecycle()


def test_mutated_caller_request_cannot_alias_formal_replay_cache(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    _commit_cutoff(scenario)
    first = scenario.runtime.ensure_h2d_traced(
        block_key,
        scenario.target_replica,
        (scenario.request_handle,),
        transfer_id="resident-demand",
        timestamp_ns=8,
        request=scenario.demand_request,
    )
    original_event_ids = scenario.demand_request.schedule_event_ids
    object.__setattr__(
        scenario.demand_request,
        "schedule_event_ids",
        ("unknown-schedule-event",),
    )

    with pytest.raises(IdentityError, match="reused"):
        scenario.runtime.ensure_h2d_traced(
            block_key,
            scenario.target_replica,
            (scenario.request_handle,),
            transfer_id="resident-demand",
            timestamp_ns=8,
            request=scenario.demand_request,
        )

    object.__setattr__(
        scenario.demand_request,
        "schedule_event_ids",
        original_event_ids,
    )
    replay = scenario.runtime.ensure_h2d_traced(
        block_key,
        scenario.target_replica,
        (scenario.request_handle,),
        transfer_id="resident-demand",
        timestamp_ns=8,
        request=scenario.demand_request,
    )
    assert replay.replayed
    assert replay.receipt is first.receipt
    scenario.committer.abort()


def test_attempt_factory_cannot_mutate_runtime_cutoff_prefix(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    before_events = scenario.runtime.events

    def mutate_detached_view(view: AtomicCutoffView) -> AttemptPayload:
        object.__setattr__(view.lifecycle_prefix[0], "reason", "forged-reason")
        return scenario.attempt_factory(view)

    with pytest.raises(StateTransitionError, match="detached cutoff view"):
        scenario.runtime.commit_shared_lease_cutoff_traced(
            block_key,
            cutoff_ns=5,
            horizon_duration_ns=10,
            operation_id="mutated-cutoff",
            observation_id="mutated-observation",
            attempt_factory=mutate_detached_view,
        )

    assert scenario.runtime.events == before_events
    _commit_cutoff(scenario)
    assert scenario.committer.commits[-1].operation_id == "formal-cutoff"
    scenario.committer.abort()


def test_cutoff_return_uses_the_canonical_attempt_snapshot(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    captured: list[PredictedAttemptPayload] = []
    original_commit = scenario.committer.commit_cutoff

    def capturing_factory(view: AtomicCutoffView) -> AttemptPayload:
        attempt = scenario.attempt_factory(view)
        assert isinstance(attempt, PredictedAttemptPayload)
        captured.append(attempt)
        return attempt

    def durable_then_mutate_source(*args: object, **kwargs: object) -> object:
        receipt = original_commit(*args, **kwargs)  # type: ignore[arg-type]
        source = captured[0]
        object.__setattr__(
            source,
            "context",
            replace(
                source.context,
                model_artifact_digest=_digest("mutated-after-commit"),
            ),
        )
        return receipt

    monkeypatch.setattr(
        scenario.committer,
        "commit_cutoff",
        durable_then_mutate_source,
    )
    committed = scenario.runtime.commit_shared_lease_cutoff_traced(
        block_key,
        cutoff_ns=5,
        horizon_duration_ns=10,
        operation_id="snapshot-cutoff",
        observation_id="snapshot-observation",
        attempt_factory=capturing_factory,
    )
    persisted = next(
        record.payload
        for record in scenario.committer.records
        if record.record_type == trace_module.TraceRecordType.FORECAST_ATTEMPT
    )

    assert committed.attempt == persisted
    assert committed.attempt != captured[0]
    assert scenario.committer.poisoned_reason is None
    scenario.committer.abort()


def test_formal_post_ledger_apply_failure_poisons_both_layers(
    tmp_path: Path,
    block_key: BlockKey,
    digest: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _formal_scenario(tmp_path, block_key, digest)
    _commit_cutoff(scenario)
    target = ReplicaId(Tier.CPU, "numa:0", "formal-save", 1)
    command = scenario.runtime.begin_d2h(
        block_key,
        target,
        transfer_id="formal-save",
        timestamp_ns=6,
    )
    assert command is not None

    def fail_slot_release(_replica: ReplicaId) -> None:
        raise _UnprintableAbort()

    monkeypatch.setattr(scenario.runtime, "_free_slot", fail_slot_release)
    with pytest.raises(_UnprintableAbort):
        scenario.runtime.fail_transfer(
            command.transfer_id,
            timestamp_ns=7,
            observed_bytes=0,
            observed_digest=None,
            error="injected DMA failure",
        )

    assert scenario.committer.poisoned_reason is not None
    with pytest.raises(StateTransitionError, match="runtime is poisoned"):
        scenario.runtime.register_workflow(
            WorkflowSpec(
                WorkflowKey("after-formal-poison", 0),
                (WorkflowNode("node"),),
            )
        )
    with pytest.raises(trace_module.TraceCommitIndeterminateError, match="poisoned"):
        scenario.committer.seal_trace()


def test_formal_committer_constructor_binding_is_fail_closed(
    block_key: BlockKey,
) -> None:
    template = _formal_scenario
    assert callable(template)
    with pytest.raises(StateTransitionError, match="configured canonical"):
        LifecycleOrchestrator(
            run_id="missing-formal-writer",
            trace_required=False,
        ).commit_shared_lease_cutoff_traced(
            block_key,
            cutoff_ns=1,
            horizon_duration_ns=1,
            operation_id="cutoff",
            observation_id="observation",
            attempt_factory=lambda _view: object(),  # type: ignore[return-value]
        )
