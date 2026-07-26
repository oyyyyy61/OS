"""C1-B0 closed trace schema and state-machine tests."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from dagkv.c1_leases import (
    DependenceGroup,
    ForecastSource,
    JointOutcome,
    LeaseOwnerSnapshot,
    ReuseClaim,
    SharedLeaseForecast,
    SharedLeasePolicySnapshot,
)
from dagkv.c1_trace import (
    TRACE_SCHEMA_VERSION,
    AbstainedAttemptPayload,
    AbstentionReason,
    CutoffPayload,
    DemandIntentPayload,
    DurableTraceWriter,
    EvidenceRole,
    ForecastAttemptContext,
    ForecastAttemptStatus,
    H2DFailedService,
    ObservationTerminalPayload,
    PredictedAttemptPayload,
    ReplayScheduleWatermarkPayload,
    RequestCancelledService,
    ResidentExecMapService,
    ReuseEpochPayload,
    ScheduleProducerKind,
    ServiceDisposition,
    TerminalReason,
    TerminalStatus,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceRecord,
    TraceRecordType,
    TraceStateMachine,
    TraceValidationError,
    TraceValidationReceipt,
    ValidatedTrace,
    WorkflowTopologyPayload,
    encode_trace_record,
    load_trace_jsonl,
    parse_trace_record,
    reconstruct_demand_label,
    trace_stream_digest,
    validate_trace,
    validate_trace_for_labels,
)
from dagkv.domain import (
    BlockKey,
    ExecutionRef,
    ResidencyState,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


class _LifecycleGate:
    def verify_lifecycle(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        header = trace.header.payload
        assert isinstance(header, TraceHeaderPayload)
        return TraceValidationReceipt(
            role=EvidenceRole.LIFECYCLE,
            trace_pair_id=header.trace_pair_id,
            trace_digest=trace_stream_digest(trace.records),
            artifact_digest=_digest("lifecycle-sidecar"),
            verifier_digest=_digest("lifecycle-verifier"),
            verified_observation_ids=("observation-1",),
        )


class _ScheduleGate:
    def verify_schedule(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        header = trace.header.payload
        assert isinstance(header, TraceHeaderPayload)
        return TraceValidationReceipt(
            role=EvidenceRole.SCHEDULE,
            trace_pair_id=header.trace_pair_id,
            trace_digest=trace_stream_digest(trace.records),
            artifact_digest=header.schedule_digest,
            verifier_digest=_digest("schedule-verifier"),
            verified_observation_ids=("observation-1",),
        )


def _authorized(records: tuple[TraceRecord, ...]) -> ValidatedTrace:
    return validate_trace_for_labels(
        records,
        lifecycle_gate=_LifecycleGate(),
        schedule_gate=_ScheduleGate(),
    )


def _record(
    sequence: int,
    record_type: TraceRecordType,
    payload: object,
    *,
    parent: str | None,
    observation_id: str | None = None,
) -> TraceRecord:
    return TraceRecord(
        schema_version=TRACE_SCHEMA_VERSION,
        record_type=record_type,
        trace_id="trace-1",
        run_id="run-1",
        schedule_id="schedule-1",
        schedule_case_id="case-1",
        sequence=sequence,
        record_id=f"record-{sequence:02d}",
        parent_record_id=parent,
        observation_id=observation_id,
        payload=payload,  # type: ignore[arg-type]
    )


def _trace(
    block_key: BlockKey,
    *,
    predicted: bool = True,
    service: str = "resident",
    terminal_status: TerminalStatus = TerminalStatus.COMPLETE,
) -> tuple[TraceRecord, ...]:
    workflow = WorkflowKey("workflow-1", 0)
    spec = WorkflowSpec(
        key=workflow,
        nodes=(WorkflowNode("root"), WorkflowNode("next", ("root",))),
    )
    grammar_digest = _digest("grammar")
    header = _record(
        0,
        TraceRecordType.TRACE_HEADER,
        TraceHeaderPayload(
            trace_pair_id="pair-1",
            source_digest=_digest("source"),
            schedule_digest=_digest("schedule"),
            split_manifest_digest=_digest("split"),
            branch_grammar_digest=grammar_digest,
            feature_contract_digest=_digest("feature-contract"),
            implementation_digest=_digest("implementation"),
            environment_digest=_digest("environment"),
        ),
        parent=None,
    )
    topology = _record(
        1,
        TraceRecordType.WORKFLOW_TOPOLOGY,
        WorkflowTopologyPayload(
            workflow_spec=spec,
            workflow_template_digest=_digest("template"),
            source_case_digest=_digest("case"),
            split_component_id="component-1",
            branch_grammar_digest=grammar_digest,
        ),
        parent=header.record_id,
    )
    owner = LeaseOwnerSnapshot(
        binding_id="retention-1",
        workflow=workflow,
        created_ns=1,
        eligible_node_ids=("next", "root"),
    )
    snapshot = SharedLeasePolicySnapshot(
        block_key=block_key,
        runtime_event_count=0,
        location_version=1,
        residency=ResidencyState.GPU_ONLY,
        owners=(owner,),
    )
    cutoff = _record(
        2,
        TraceRecordType.CUTOFF,
        CutoffPayload(
            topology_record_ids=(topology.record_id,),
            snapshot=snapshot,
            cutoff_ns=10,
            horizon_duration_ns=20,
            deadline_ns=30,
            lifecycle_event_count=0,
            last_event_id=None,
            last_event_timestamp_ns=None,
            atomic_cutoff_view_digest=_digest("atomic-cutoff-view"),
            feature_view_digest=_digest("feature-view"),
        ),
        parent=header.record_id,
        observation_id="observation-1",
    )
    context = ForecastAttemptContext(
        feature_view_digest=_digest("feature-view"),
        information_cutoff_digest=_digest("atomic-cutoff-view"),
        model_artifact_digest=_digest("model"),
        predictor_digest=_digest("predictor"),
        dependence_digest=_digest("dependence"),
        outcome_catalog_digest=_digest("catalog"),
        grouping_rules_digest=_digest("grouping"),
        model_inputs_digest=_digest("model-inputs"),
    )
    if predicted:
        claim = ReuseClaim(
            claim_id="claim-1",
            binding_id=owner.binding_id,
            workflow=workflow,
            node_id="next",
            reuse_epoch_id="epoch-1",
            access_ns=20,
        )
        forecast = SharedLeaseForecast(
            forecast_id="forecast-1",
            block_key=block_key,
            runtime_event_count=0,
            generated_ns=10,
            horizon_ns=30,
            source=ForecastSource.PREDICTED,
            predictor_digest=context.predictor_digest,
            dependence_digest=context.dependence_digest,
            independence_basis="one complete grammar group",
            groups=(
                DependenceGroup(
                    group_id="group-1",
                    outcomes=(
                        JointOutcome("demand", 500_000, (claim,)),
                        JointOutcome("none", 500_000),
                    ),
                ),
            ),
        )
        attempt_payload: object = PredictedAttemptPayload(
            status=ForecastAttemptStatus.PREDICTED,
            context=context,
            forecast=forecast,
        )
    else:
        attempt_payload = AbstainedAttemptPayload(
            status=ForecastAttemptStatus.ABSTAINED,
            context=context,
            reason=AbstentionReason.INSUFFICIENT_DATA,
        )
    attempt = _record(
        3,
        TraceRecordType.FORECAST_ATTEMPT,
        attempt_payload,
        parent=cutoff.record_id,
        observation_id="observation-1",
    )

    records = [header, topology, cutoff, attempt]
    if service != "none":
        intent = _record(
            4,
            TraceRecordType.DEMAND_INTENT,
            DemandIntentPayload(
                schedule_event_id="schedule-event-1",
                scheduled_access_ns=20,
                claim_id="claim-1",
                retention_binding_id=owner.binding_id,
                request_binding_id="request-binding-1",
                workflow=workflow,
                node_id="next",
                execution_ref=ExecutionRef(workflow, "request-1", "sequence-1", 0),
                block_key=block_key,
                reuse_epoch_id="epoch-1",
                pre_service_event_count=0,
                pre_service_last_event_id=None,
                pre_service_last_timestamp_ns=None,
            ),
            parent=attempt.record_id,
            observation_id="observation-1",
        )
        if service == "resident":
            terminal = ResidentExecMapService(
                intent_record_id=intent.record_id,
                disposition=ServiceDisposition.RESIDENT_EXEC_MAP,
                exec_map_event_id="event-exec-map",
            )
        elif service == "h2d_failed":
            terminal = H2DFailedService(
                intent_record_id=intent.record_id,
                disposition=ServiceDisposition.H2D_FAILED,
                transfer_id="transfer-1",
                transfer_scheduled_event_id="event-load-scheduled",
                transfer_terminal_event_id="event-load-failed",
                waiter_binding_ids=("request-binding-1",),
            )
        else:
            terminal = RequestCancelledService(
                intent_record_id=intent.record_id,
                disposition=ServiceDisposition.REQUEST_CANCELLED,
                release_event_id="event-request-release",
            )
        epoch = _record(
            5,
            TraceRecordType.REUSE_EPOCH,
            ReuseEpochPayload(
                reuse_epoch_id="epoch-1",
                access_ns=20,
                block_key=block_key,
                demand_intent_record_ids=(intent.record_id,),
                service_terminals=(terminal,),
            ),
            parent=intent.record_id,
            observation_id="observation-1",
        )
        records.extend((intent, epoch))

    previous = records[-1]
    if terminal_status == TerminalStatus.COMPLETE:
        watermark = _record(
            len(records),
            TraceRecordType.SCHEDULE_WATERMARK,
            ReplayScheduleWatermarkPayload(
                producer_kind=ScheduleProducerKind.REPLAY,
                producer_id="scheduler-1",
                producer_artifact_digest=_digest("scheduler"),
                schedule_digest=_digest("schedule"),
                consumed_event_count=1,
                last_schedule_event_id="schedule-event-1",
                max_closed_timestamp_ns=31,
            ),
            parent=previous.record_id,
            observation_id="observation-1",
        )
        records.append(watermark)
        previous = watermark
        terminal_payload = ObservationTerminalPayload(
            status=TerminalStatus.COMPLETE,
            reason=TerminalReason.WINDOW_COMPLETE,
            label_available_ns=31,
            schedule_watermark_record_id=watermark.record_id,
            last_verified_event_count=0,
            last_verified_event_id=None,
            last_verified_event_timestamp_ns=None,
            unresolved_demand_intent_record_ids=(),
        )
    else:
        terminal_payload = ObservationTerminalPayload(
            status=TerminalStatus.CENSORED,
            reason=TerminalReason.TRACE_TRUNCATED,
            label_available_ns=None,
            schedule_watermark_record_id=None,
            last_verified_event_count=0,
            last_verified_event_id=None,
            last_verified_event_timestamp_ns=None,
            unresolved_demand_intent_record_ids=(),
        )
    records.append(
        _record(
            len(records),
            TraceRecordType.OBSERVATION_TERMINAL,
            terminal_payload,
            parent=previous.record_id,
            observation_id="observation-1",
        )
    )
    return tuple(records)


def test_predicted_trace_round_trip_and_reconstructs_resident_demand(
    block_key: BlockKey,
) -> None:
    records = _trace(block_key)
    parsed = tuple(
        parse_trace_record(encode_trace_record(record)) for record in records
    )
    validated = _authorized(parsed)

    assert parsed == records
    assert reconstruct_demand_label(validated, "observation-1").first_demand == 1
    assert reconstruct_demand_label(validated, "observation-1").epoch_count == 1


def test_failed_h2d_remains_positive_demand(block_key: BlockKey) -> None:
    validated = _authorized(_trace(block_key, service="h2d_failed"))
    label = reconstruct_demand_label(validated, "observation-1")

    assert label.first_demand == 1
    assert label.epoch_count == 1
    assert label.repeat_count == 0


def test_abstention_reaches_complete_attempt_accounting(block_key: BlockKey) -> None:
    records = _trace(block_key, predicted=False, service="none")
    validated = _authorized(records)

    assert isinstance(
        validated.observations[0].attempt.payload, AbstainedAttemptPayload
    )
    assert reconstruct_demand_label(validated, "observation-1").first_demand == 0


def test_noncomplete_observation_never_exposes_a_label(block_key: BlockKey) -> None:
    validated = validate_trace(
        _trace(block_key, service="h2d_failed", terminal_status=TerminalStatus.CENSORED)
    )

    with pytest.raises(TraceValidationError, match="unavailable"):
        reconstruct_demand_label(validated, "observation-1")


def test_complete_trace_requires_watermark_beyond_deadline(block_key: BlockKey) -> None:
    records = list(_trace(block_key, service="none"))
    watermark = records[-2]
    payload = watermark.payload
    assert isinstance(payload, ReplayScheduleWatermarkPayload)
    records[-2] = replace(
        watermark,
        payload=replace(payload, max_closed_timestamp_ns=30),
    )

    with pytest.raises(TraceValidationError, match="does not exceed"):
        validate_trace(tuple(records))


def test_epoch_then_new_intent_fails_closed(block_key: BlockKey) -> None:
    records = list(_trace(block_key))
    original_intent = records[4]
    epoch = records[5]
    extra = replace(
        original_intent,
        sequence=6,
        record_id="record-extra",
        parent_record_id=epoch.record_id,
        payload=replace(
            original_intent.payload,
            schedule_event_id="schedule-event-2",
            claim_id="claim-2",
            request_binding_id="request-binding-2",
            execution_ref=replace(
                original_intent.payload.execution_ref,
                request_id="request-2",
            ),
            reuse_epoch_id="epoch-2",
            scheduled_access_ns=21,
        ),
    )
    prefix = tuple(records[:6]) + (extra,)
    machine = TraceStateMachine()
    with pytest.raises(TraceValidationError, match="order"):
        for record in prefix:
            machine.push(record)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "float", "whitespace", "unknown"],
)
def test_parser_rejects_noncanonical_or_open_schema_bytes(
    block_key: BlockKey,
    mutation: str,
) -> None:
    raw = encode_trace_record(_trace(block_key)[0])
    if mutation == "duplicate":
        raw = b'{"record_id":"duplicate",' + raw[1:]
    elif mutation == "float":
        value = json.loads(raw)
        value["sequence"] = 0.0
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    elif mutation == "whitespace":
        raw = b" " + raw
    else:
        value = json.loads(raw)
        value["unknown"] = "field"
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()

    with pytest.raises(TraceValidationError):
        parse_trace_record(raw)


def test_durable_writer_is_create_only_and_replayable(
    tmp_path,
    block_key: BlockKey,
) -> None:
    path = (tmp_path / "trace.jsonl").resolve()
    records = _trace(block_key)
    with DurableTraceWriter(path) as writer:
        first = writer.append_durable(
            records[:4],
            event_count=0,
            view_digest=_digest("cutoff-view"),
        )
        second = writer.append_durable(
            records[4:],
            event_count=0,
            view_digest=_digest("terminal-view"),
        )

    assert first.record_ids == tuple(record.record_id for record in records[:4])
    assert first.batch_digest != second.batch_digest
    assert load_trace_jsonl(path) == records
    with pytest.raises(OSError, match="create-only"):
        DurableTraceWriter(path)


def test_writer_fsync_failure_poisoning_is_fail_closed(
    tmp_path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "trace.jsonl").resolve()
    writer = DurableTraceWriter(path)

    def fail_fsync(descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr("dagkv.c1_trace.os.fsync", fail_fsync)
    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        writer.append_durable(
            (_trace(block_key)[0],),
            event_count=0,
            view_digest=_digest("view"),
        )
    with pytest.raises(TraceCommitIndeterminateError, match="poisoned"):
        writer.append_durable(
            (_trace(block_key)[0],),
            event_count=0,
            view_digest=_digest("view"),
        )
    monkeypatch.undo()
    writer.close(finalize=False)
